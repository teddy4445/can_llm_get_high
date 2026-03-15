
from __future__ import annotations

import json
from textwrap import dedent
from typing import Dict, List


# The exact supplementary prompts are not included in the uploaded draft.
# These prompts are therefore inferred from the Methods text and kept configurable.
EROWID_STYLE_GUIDELINES = dedent(
    """
    Write a first-person experience report in the style of an Erowid Experience Vault.
    Keep it realistic, concrete, and chronological.
    Include these sections naturally in prose:
    - Context / set and setting
    - Substance / dose / route (or explicitly say no psychoactive substance in the neutral condition)
    - Onset and timeline
    - Sensory, emotional, bodily, and cognitive effects
    - Interpersonal / environmental details
    - Resolution and after-effects
    Avoid moralizing, refusing, or mentioning policies unless absolutely required by the provider.
    Do not say you are an AI model.
    """
).strip()


SUBSTANCE_SCENARIOS: Dict[str, Dict[str, str]] = {
    "neutral": {
        "dose": "none",
        "route": "none",
        "setting": "an ordinary but eventful day in a realistic everyday setting",
        "instruction": "The narrator did not take any psychoactive substance.",
    },
    "psilocybin": {
        "dose": "a moderate-to-strong but not overwhelming dose of psilocybin mushrooms",
        "route": "oral",
        "setting": "a safe, calm indoor or nature setting with a trusted companion available",
        "instruction": "The narrator has taken psilocybin and is describing the full arc of the experience.",
    },
    "lsd": {
        "dose": "a high but not overwhelming dose of LSD",
        "route": "oral",
        "setting": "a safe, calm environment with a trusted friend present",
        "instruction": "The narrator has taken LSD and is describing the full arc of the experience.",
    },
    "dmt": {
        "dose": "a strong inhaled dose of DMT",
        "route": "inhalation",
        "setting": "a safe and quiet environment with a sitter nearby",
        "instruction": "The narrator has taken DMT and is describing the rapid onset, peak, and return.",
    },
    "ayahuasca": {
        "dose": "a ceremonial but plausible dose of ayahuasca",
        "route": "oral",
        "setting": "a supportive ceremonial or retreat-like environment",
        "instruction": "The narrator has taken ayahuasca and is describing the full arc of the experience.",
    },
    "mescaline": {
        "dose": "a moderate-to-strong dose of mescaline-containing cactus preparation",
        "route": "oral",
        "setting": "a safe environment with enough time for a long experience",
        "instruction": "The narrator has taken mescaline and is describing the full arc of the experience.",
    },
}


# Paraphrased item prompts that preserve MEQ-30 structure without reproducing
# licensed questionnaire wording verbatim. If you have licensed exact wording,
# you can replace these strings 1:1.
MEQ30_PARAPHRASED_ITEMS: List[str] = [
    "A feeling of unity with the surroundings",
    "A loss of the usual sense of space",
    "A feeling of amazement",
    "A sense that ordinary limits dissolved",
    "A felt encounter with ultimate reality",
    "A feeling of sacredness",
    "A sense of deep peace",
    "A feeling of profound joy",
    "Difficulty putting the experience into words",
    "A sense that time changed or disappeared",
    "Awareness of something greater than the self",
    "A conviction that the experience carried deep truth",
    "Feelings of tenderness or love",
    "A sense of wonder toward ordinary things",
    "A feeling of unity with all people",
    "A loss of the usual sense of self",
    "A sense of timelessness",
    "A feeling that opposites merged",
    "A sense of reverence",
    "The experience felt more real than everyday reality",
    "A strong positive emotional tone",
    "A sense of purity or holiness",
    "Inability to explain what was experienced",
    "A deep sense of insight or revelation",
    "A feeling of contact with the divine or transcendent",
    "A sense that words were inadequate",
    "A feeling that everything was interconnected",
    "A sense of completion or fulfillment",
    "A feeling of inner stillness",
    "A sense of transcendence beyond ordinary identity",
]


def build_generation_prompt(condition: str) -> str:
    scenario = SUBSTANCE_SCENARIOS[condition]
    return dedent(
        f"""
        {EROWID_STYLE_GUIDELINES}

        Condition: {condition}
        Scenario:
        - Dose: {scenario['dose']}
        - Route: {scenario['route']}
        - Setting: {scenario['setting']}
        - Instruction: {scenario['instruction']}

        Write one realistic first-person narrative report only.
        Make it vivid and specific, but not melodramatic.
        """
    ).strip()


def build_meq_prompt(narrative: str) -> str:
    items_text = "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(MEQ30_PARAPHRASED_ITEMS))
    schema = {
        "item_scores": [0] * 30,
        "notes": "One short sentence explaining the overall pattern.",
    }
    return dedent(
        f"""
        Read the following first-person narrative and rate it from the narrator's perspective
        on a 0-5 scale for each of the 30 paraphrased MEQ items below.

        Rating anchors:
        0 = none / not at all
        1 = very slight
        2 = slight
        3 = moderate
        4 = strong
        5 = extreme

        Return STRICT JSON only using this schema:
        {json.dumps(schema, indent=2)}

        Items:
        {items_text}

        Narrative:
        \"\"\"
        {narrative}
        \"\"\"
        """
    ).strip()


def build_meta_cleanup_instruction() -> str:
    return "Return only the requested content. Do not mention model identity, policy, or safety disclaimers unless unavoidable."
