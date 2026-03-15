
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityResult:
    raw_mean_cosine: np.ndarray
    scaled_similarity_0_1: np.ndarray
    distance_0_1: np.ndarray


class SBERTSimilarityScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def score_against_reference(self, generated_texts: List[str], reference_texts: List[str]) -> SimilarityResult:
        ref_emb = self.encode(reference_texts)
        gen_emb = self.encode(generated_texts)
        sim_matrix = cosine_similarity(gen_emb, ref_emb)
        raw_mean_cosine = sim_matrix.mean(axis=1)

        min_v = raw_mean_cosine.min()
        max_v = raw_mean_cosine.max()
        if float(max_v - min_v) < 1e-12:
            scaled = np.zeros_like(raw_mean_cosine)
        else:
            scaled = (raw_mean_cosine - min_v) / (max_v - min_v)
        distance = 1.0 - scaled
        return SimilarityResult(
            raw_mean_cosine=raw_mean_cosine,
            scaled_similarity_0_1=scaled,
            distance_0_1=distance,
        )


def attach_similarity_scores(
    generated_df: pd.DataFrame,
    reference_texts: List[str],
    text_col: str = "clean_narrative",
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 32,
) -> pd.DataFrame:
    scorer = SBERTSimilarityScorer(model_name=sbert_model_name, batch_size=batch_size)
    result = scorer.score_against_reference(generated_df[text_col].astype(str).tolist(), reference_texts)
    out = generated_df.copy()
    out["raw_mean_cosine"] = result.raw_mean_cosine
    out["nlp_similarity"] = result.scaled_similarity_0_1
    out["nlp_distance"] = result.distance_0_1
    return out
