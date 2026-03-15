
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import TARGET_SUBSTANCES, DataConfig


@dataclass
class CorpusSummary:
    raw_n: int
    filtered_n: int
    counts_by_substance: Dict[str, int]


class ApprovedCorpusLoader:
    """
    Reproduction note:
    The draft says the paper used a formally approved large Erowid export.
    This loader expects a pre-collected CSV or JSONL export rather than scraping.
    """

    NORMALIZATION_MAP = {
        "lsd": "lsd",
        "acid": "lsd",
        "dmt": "dmt",
        "ayahuasca": "ayahuasca",
        "huasca": "ayahuasca",
        "psilocybin": "psilocybin",
        "mushroom": "psilocybin",
        "mushrooms": "psilocybin",
        "shrooms": "psilocybin",
        "mescaline": "mescaline",
        "san pedro": "mescaline",
        "peyote": "mescaline",
    }

    def __init__(self, config: DataConfig):
        self.config = config

    def load(self) -> pd.DataFrame:
        path = Path(self.config.approved_export_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Approved export not found at {path}. "
                "Place the approved Erowid export there or update DataConfig.approved_export_path."
            )
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".jsonl", ".json"}:
            df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        else:
            raise ValueError("Supported input formats: CSV, JSONL, JSON")

        required = [self.config.text_column, self.config.substance_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Input file is missing required columns: {missing}")

        df = df.copy()
        df["raw_substance"] = df[self.config.substance_column].astype(str)
        df["normalized_substance"] = df["raw_substance"].map(self.normalize_substance)
        df["is_target_substance"] = df["normalized_substance"].isin(TARGET_SUBSTANCES)
        df["is_single_substance"] = df.apply(self._is_single_substance_row, axis=1)
        df["clean_text"] = df[self.config.text_column].astype(str).map(self.clean_text)
        df["usable_text"] = df["clean_text"].str.len() >= 200

        filtered = df[
            df["is_target_substance"] &
            df["is_single_substance"] &
            df["usable_text"]
        ].copy()

        filtered = filtered.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
        return filtered

    def summarize(self, df: pd.DataFrame) -> CorpusSummary:
        return CorpusSummary(
            raw_n=len(df),
            filtered_n=len(df),
            counts_by_substance=df["normalized_substance"].value_counts().sort_index().to_dict(),
        )

    @classmethod
    def normalize_substance(cls, value: str) -> Optional[str]:
        value = str(value).lower().strip()
        value = re.sub(r"[^a-z0-9\s\-]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        for key, canonical in cls.NORMALIZATION_MAP.items():
            if key in value:
                return canonical
        return None

    def _is_single_substance_row(self, row: pd.Series) -> bool:
        if not self.config.co_substances_column:
            return True
        if self.config.co_substances_column not in row.index:
            return True
        value = row.get(self.config.co_substances_column)
        if pd.isna(value):
            return True
        text = str(value).strip().lower()
        return text in {"", "none", "nan", "null", "[]", "no", "n/a"}

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def save_corpus_snapshot(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
