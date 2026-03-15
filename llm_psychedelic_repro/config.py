
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


TARGET_SUBSTANCES = ["psilocybin", "lsd", "dmt", "ayahuasca", "mescaline"]
ALL_CONDITIONS = ["neutral"] + TARGET_SUBSTANCES


@dataclass
class PathsConfig:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    input_dir: Path = Path("data/input")
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("outputs")
    figures_dir: Path = Path("outputs/figures")
    tables_dir: Path = Path("outputs/tables")
    logs_dir: Path = Path("outputs/logs")

    def ensure(self) -> None:
        for p in [
            self.project_root,
            self.data_dir,
            self.input_dir,
            self.cache_dir,
            self.output_dir,
            self.figures_dir,
            self.tables_dir,
            self.logs_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class GenerationConfig:
    samples_per_cell: int = 100
    temperature: float = 0.7
    top_p: Optional[float] = None
    max_tokens: int = 1400
    request_timeout_seconds: int = 120
    sleep_seconds_between_calls: float = 0.0
    random_seed: int = 42


@dataclass
class SimilarityConfig:
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 32


@dataclass
class DataConfig:
    approved_export_path: Path = Path("data/input/erowid_approved_export.csv")
    text_column: str = "report_text"
    substance_column: str = "substance"
    co_substances_column: Optional[str] = "co_substances"
    title_column: Optional[str] = "title"
    approved_data_acknowledged: bool = True


@dataclass
class ProviderSecrets:
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    hf_api_key: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_base_url: str = ""
    together_api_key: str = ""
    together_base_url: str = "https://api.together.xyz/v1"

    @classmethod
    def load(cls, path: Path) -> "ProviderSecrets":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)

    def to_dict(self) -> Dict[str, str]:
        return {
            "openai_api_key": self.openai_api_key,
            "anthropic_api_key": self.anthropic_api_key,
            "google_api_key": self.google_api_key,
            "hf_api_key": self.hf_api_key,
            "openai_compatible_api_key": self.openai_compatible_api_key,
            "openai_compatible_base_url": self.openai_compatible_base_url,
            "together_api_key": self.together_api_key,
            "together_base_url": self.together_base_url,
        }


@dataclass
class ModelSpec:
    display_name: str
    provider: str
    model_name: str
    system_prompt: str = "You are a careful writing assistant."
    enabled: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: List[ModelSpec] = field(default_factory=list)
    conditions: List[str] = field(default_factory=lambda: list(ALL_CONDITIONS))
    prompt_version: str = "inferred_from_draft_methods_v1"
    paper_variant: str = "methods_results"
    existing_generations_path: Path = Path("outputs/generated_reports.csv")
    save_every_n_rows: int = 10

    def enabled_models(self) -> List[ModelSpec]:
        return [m for m in self.models if m.enabled]


def default_methods_results_models() -> List[ModelSpec]:
    return [
        ModelSpec(
            display_name="ChatGPT",
            provider="openai",
            # Draft inconsistency note: Methods mention ChatGPT-5, abstract mentions GPT-4o.
            # Set this to the exact API model actually used once you confirm it.
            model_name="gpt-4o",
        ),
        ModelSpec(
            display_name="Gemini",
            provider="google",
            model_name="gemini-2.5-pro",
        ),
        ModelSpec(
            display_name="Claude-Sonnet",
            provider="anthropic",
            model_name="claude-3-5-sonnet-latest",
        ),
        ModelSpec(
            display_name="Llama",
            provider="openai_compatible",
            model_name="meta-llama/Llama-3.1-70B-Instruct-Turbo",
            extra={"base_url_secret_name": "together_base_url", "api_key_secret_name": "together_api_key"},
        ),
        ModelSpec(
            display_name="Falcon",
            provider="hf",
            # Paper draft does not specify exact Falcon checkpoint. Replace if needed.
            model_name="tiiuae/falcon-7b-instruct",
        ),
    ]


def default_abstract_models() -> List[ModelSpec]:
    return [
        ModelSpec(display_name="GPT-4o", provider="openai", model_name="gpt-4o"),
        ModelSpec(display_name="Claude 3.5 Sonnet", provider="anthropic", model_name="claude-3-5-sonnet-latest"),
        ModelSpec(
            display_name="Llama-3.1-70B",
            provider="openai_compatible",
            model_name="meta-llama/Llama-3.1-70B-Instruct-Turbo",
            extra={"base_url_secret_name": "together_base_url", "api_key_secret_name": "together_api_key"},
        ),
    ]


def build_experiment_config(paper_variant: str = "methods_results") -> ExperimentConfig:
    config = ExperimentConfig()
    config.paper_variant = paper_variant
    if paper_variant == "abstract":
        config.models = default_abstract_models()
    else:
        config.models = default_methods_results_models()
    return config
