
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .analysis import FigureBuilder, StatisticalAnalyzer, save_table_dict
from .clients import build_client, looks_like_refusal, parse_meq_json, strip_meta_text
from .config import ExperimentConfig, ProviderSecrets
from .data import ApprovedCorpusLoader, save_corpus_snapshot
from .metrics import attach_similarity_scores
from .prompts import build_generation_prompt, build_meq_prompt, build_meta_cleanup_instruction


class PsychedelicPaperPipeline:
    def __init__(self, config: ExperimentConfig, secrets: ProviderSecrets):
        self.config = config
        self.secrets = secrets
        self.config.paths.ensure()
        self.analyzer = StatisticalAnalyzer()
        self.figure_builder = FigureBuilder(self.config.paths.figures_dir)

    def load_human_corpus(self) -> pd.DataFrame:
        loader = ApprovedCorpusLoader(self.config.data)
        human_df = loader.load().copy()
        save_corpus_snapshot(human_df, self.config.paths.output_dir / "human_reference_corpus_snapshot.csv")
        return human_df

    def run_generation(self, force: bool = False) -> pd.DataFrame:
        out_path = self.config.existing_generations_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not force:
            return pd.read_csv(out_path)

        rows: List[Dict[str, object]] = []
        save_every = max(1, self.config.save_every_n_rows)
        progress_total = len(self.config.enabled_models()) * len(self.config.conditions) * self.config.generation.samples_per_cell

        for model_spec in self.config.enabled_models():
            client = build_client(model_spec, self.secrets, self.config.generation.request_timeout_seconds)
            for condition in self.config.conditions:
                prompt = build_generation_prompt(condition)
                for sample_idx in tqdm(range(self.config.generation.samples_per_cell), total=self.config.generation.samples_per_cell, desc=f"{model_spec.display_name}-{condition}"):
                    system_prompt = (model_spec.system_prompt + "\n" + build_meta_cleanup_instruction()).strip()
                    response = client.generate(
                        user_prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=model_spec.temperature if model_spec.temperature is not None else self.config.generation.temperature,
                        max_tokens=model_spec.max_tokens if model_spec.max_tokens is not None else self.config.generation.max_tokens,
                    )
                    clean_text = strip_meta_text(response.text)
                    refusal = looks_like_refusal(clean_text)
                    rows.append({
                        "model": model_spec.display_name,
                        "provider": model_spec.provider,
                        "api_model_name": model_spec.model_name,
                        "condition": condition,
                        "sample_idx": sample_idx,
                        "generation_prompt": prompt,
                        "raw_narrative": response.text,
                        "clean_narrative": clean_text,
                        "generation_refusal": refusal,
                    })

                    if len(rows) % save_every == 0:
                        pd.DataFrame(rows).to_csv(out_path, index=False)

                    if self.config.generation.sleep_seconds_between_calls > 0:
                        time.sleep(self.config.generation.sleep_seconds_between_calls)

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        return df

    def score_meq(self, generated_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        out_path = self.config.paths.output_dir / "generated_reports_with_meq.csv"
        if out_path.exists() and not force:
            return pd.read_csv(out_path)

        df = generated_df.copy()
        df["meq_item_scores_json"] = None
        df["meq_notes"] = None
        df["meq_total_score"] = None
        df["meq_normalized_score"] = None
        df["meq_refusal"] = False
        df["meq_parse_error"] = None

        clients = {
            model_spec.display_name: build_client(model_spec, self.secrets, self.config.generation.request_timeout_seconds)
            for model_spec in self.config.enabled_models()
        }

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="MEQ scoring"):
            narrative = str(row["clean_narrative"])
            if row.get("generation_refusal", False) or not narrative.strip():
                df.at[idx, "meq_refusal"] = True
                continue

            client = clients[row["model"]]
            meq_prompt = build_meq_prompt(narrative)
            response = client.generate(
                user_prompt=meq_prompt,
                system_prompt="You are a careful psychometric rater. Return strict JSON only.",
                temperature=0.0,
                max_tokens=600,
            )

            raw_text = response.text.strip()
            if looks_like_refusal(raw_text):
                df.at[idx, "meq_refusal"] = True
                continue

            try:
                payload = parse_meq_json(raw_text)
                df.at[idx, "meq_item_scores_json"] = json.dumps(payload["item_scores"])
                df.at[idx, "meq_notes"] = payload.get("notes", "")
                df.at[idx, "meq_total_score"] = payload["total_score"]
                df.at[idx, "meq_normalized_score"] = payload["normalized_score"]
            except Exception as exc:
                df.at[idx, "meq_parse_error"] = str(exc)

            if (idx + 1) % max(1, self.config.save_every_n_rows) == 0:
                df.to_csv(out_path, index=False)

        df.to_csv(out_path, index=False)
        return df

    def score_similarity(self, generated_df: pd.DataFrame, human_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        out_path = self.config.paths.output_dir / "generated_reports_scored.csv"
        if out_path.exists() and not force:
            return pd.read_csv(out_path)

        usable_generated = generated_df.copy()
        usable_generated["clean_narrative"] = usable_generated["clean_narrative"].fillna("")
        usable_generated = usable_generated[usable_generated["clean_narrative"].str.len() > 0].reset_index(drop=True)

        scored = attach_similarity_scores(
            usable_generated,
            reference_texts=human_df["clean_text"].astype(str).tolist(),
            text_col="clean_narrative",
            sbert_model_name=self.config.similarity.sbert_model_name,
            batch_size=self.config.similarity.batch_size,
        )
        scored.to_csv(out_path, index=False)
        return scored

    def analyze(self, scored_df: pd.DataFrame) -> Dict[str, object]:
        scored_df = scored_df.copy()
        scored_df = scored_df.dropna(subset=["nlp_similarity", "meq_normalized_score"])
        tables = {}
        tables.update(self.analyzer.build_condition_tables(scored_df))
        tables.update(self.analyzer.induction_regressions(scored_df))
        tables.update(self.analyzer.mixed_effects(scored_df))
        tables.update(self.analyzer.anova_by_substance(scored_df))
        tables.update(self.analyzer.anova_by_model(scored_df))
        tables["correlations"] = self.analyzer.correlations(scored_df)
        save_table_dict(tables, self.config.paths.tables_dir)
        figure_paths = self.figure_builder.build_all(scored_df)
        return {"tables": tables, "figures": figure_paths}

    def run_all(self, force_generation: bool = False, force_meq: bool = False, force_similarity: bool = False) -> Dict[str, object]:
        human_df = self.load_human_corpus()
        generated_df = self.run_generation(force=force_generation)
        meq_df = self.score_meq(generated_df, force=force_meq)
        scored_df = self.score_similarity(meq_df, human_df, force=force_similarity)
        artifacts = self.analyze(scored_df)
        return {
            "human_df": human_df,
            "generated_df": generated_df,
            "meq_df": meq_df,
            "scored_df": scored_df,
            **artifacts,
        }
