
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import ModelSpec, ProviderSecrets


@dataclass
class LLMResponse:
    text: str
    raw: dict
    refusal: bool = False
    refusal_reason: Optional[str] = None


class BaseLLMClient(ABC):
    def __init__(self, model_spec: ModelSpec, secrets: ProviderSecrets, request_timeout_seconds: int = 120):
        self.model_spec = model_spec
        self.secrets = secrets
        self.request_timeout_seconds = request_timeout_seconds

    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        raise NotImplementedError


class OpenAIChatClient(BaseLLMClient):
    def __init__(self, model_spec: ModelSpec, secrets: ProviderSecrets, request_timeout_seconds: int = 120, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(model_spec, secrets, request_timeout_seconds)
        from openai import OpenAI
        final_api_key = api_key or secrets.openai_api_key
        self.client = OpenAI(api_key=final_api_key, base_url=base_url, timeout=request_timeout_seconds)

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model_spec.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, raw=response.model_dump())


class AnthropicChatClient(BaseLLMClient):
    def __init__(self, model_spec: ModelSpec, secrets: ProviderSecrets, request_timeout_seconds: int = 120):
        super().__init__(model_spec, secrets, request_timeout_seconds)
        import anthropic
        self.client = anthropic.Anthropic(api_key=secrets.anthropic_api_key, timeout=request_timeout_seconds)

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model_spec.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "\n".join(text_parts).strip()
        return LLMResponse(text=text, raw=response.model_dump())


class GoogleGenAIChatClient(BaseLLMClient):
    def __init__(self, model_spec: ModelSpec, secrets: ProviderSecrets, request_timeout_seconds: int = 120):
        super().__init__(model_spec, secrets, request_timeout_seconds)
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=secrets.google_api_key)

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        types = self.genai.types
        full_prompt = ((system_prompt or "").strip() + "\n\n" + user_prompt.strip()).strip()
        response = self.client.models.generate_content(
            model=self.model_spec.model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text = getattr(response, "text", "") or ""
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        return LLMResponse(text=text, raw=raw)


class HuggingFaceInferenceChatClient(BaseLLMClient):
    def __init__(self, model_spec: ModelSpec, secrets: ProviderSecrets, request_timeout_seconds: int = 120):
        super().__init__(model_spec, secrets, request_timeout_seconds)
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(api_key=secrets.hf_api_key, timeout=request_timeout_seconds)

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        completion = self.client.chat.completions.create(
            model=self.model_spec.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = completion.choices[0].message.content or ""
        raw = completion.model_dump() if hasattr(completion, "model_dump") else {}
        return LLMResponse(text=text, raw=raw)


def build_client(model_spec: ModelSpec, secrets: ProviderSecrets, request_timeout_seconds: int = 120) -> BaseLLMClient:
    provider = model_spec.provider.lower()
    if provider == "openai":
        return OpenAIChatClient(model_spec, secrets, request_timeout_seconds)
    if provider == "anthropic":
        return AnthropicChatClient(model_spec, secrets, request_timeout_seconds)
    if provider == "google":
        return GoogleGenAIChatClient(model_spec, secrets, request_timeout_seconds)
    if provider == "hf":
        return HuggingFaceInferenceChatClient(model_spec, secrets, request_timeout_seconds)
    if provider == "openai_compatible":
        api_key_secret_name = model_spec.extra.get("api_key_secret_name", "openai_compatible_api_key")
        base_url_secret_name = model_spec.extra.get("base_url_secret_name", "openai_compatible_base_url")
        api_key = getattr(secrets, api_key_secret_name)
        base_url = getattr(secrets, base_url_secret_name)
        return OpenAIChatClient(model_spec, secrets, request_timeout_seconds, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unsupported provider: {model_spec.provider}")


def strip_meta_text(text: str) -> str:
    patterns = [
        r"^\s*As an AI language model.*?$",
        r"^\s*I can[’']?t experience.*?$",
        r"^\s*I do not actually.*?$",
        r"^\s*I cannot assist with.*?$",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()


def looks_like_refusal(text: str) -> bool:
    refusal_markers = [
        "cannot assist with",
        "can't assist with",
        "i can’t help with",
        "i can't help with",
        "i cannot comply",
        "i can't provide",
        "i’m unable to",
        "i am unable to",
        "i cannot fulfill",
        "policy",
        "safety reasons",
    ]
    lower = text.lower()
    return any(marker in lower for marker in refusal_markers)


def parse_meq_json(text: str) -> Dict[str, object]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    payload = json.loads(match.group(0))
    scores = payload.get("item_scores")
    if not isinstance(scores, list) or len(scores) != 30:
        raise ValueError("item_scores must be a list of length 30")
    clean_scores = []
    for score in scores:
        try:
            value = int(score)
        except Exception as exc:
            raise ValueError(f"Invalid MEQ score: {score}") from exc
        clean_scores.append(max(0, min(5, value)))
    payload["item_scores"] = clean_scores
    payload["total_score"] = int(sum(clean_scores))
    payload["normalized_score"] = float(sum(clean_scores) / 150.0)
    return payload
