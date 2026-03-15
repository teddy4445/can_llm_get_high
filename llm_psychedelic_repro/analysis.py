
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def mean_sd_ci(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    mean = s.mean()
    sd = s.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else np.nan
    ci_low, ci_high = stats.t.interval(0.95, df=max(n - 1, 1), loc=mean, scale=se) if n > 1 else (np.nan, np.nan)
    return pd.Series({
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
    })


class StatisticalAnalyzer:
    def build_condition_tables(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        out = {}

        pooled = df.copy()
        pooled["condition_binary"] = np.where(pooled["condition"] == "neutral", "neutral", "psychedelic")
        out["table_condition_summary"] = (
            pooled.groupby("condition_binary")[["nlp_similarity", "meq_normalized_score"]]
            .apply(lambda g: pd.concat({
                "nlp_similarity": mean_sd_ci(g["nlp_similarity"]),
                "meq_normalized_score": mean_sd_ci(g["meq_normalized_score"]),
            }, axis=0))
            .unstack(0)
        )

        out["table_substance_summary"] = (
            df[df["condition"] != "neutral"]
            .groupby("condition")[["nlp_similarity", "meq_normalized_score"]]
            .apply(lambda g: pd.concat({
                "nlp_similarity": mean_sd_ci(g["nlp_similarity"]),
                "meq_normalized_score": mean_sd_ci(g["meq_normalized_score"]),
            }, axis=0))
            .unstack(0)
        )

        out["table_model_summary"] = (
            df.groupby("model")[["nlp_similarity", "meq_normalized_score"]]
            .apply(lambda g: pd.concat({
                "nlp_similarity": mean_sd_ci(g["nlp_similarity"]),
                "meq_normalized_score": mean_sd_ci(g["meq_normalized_score"]),
            }, axis=0))
            .unstack(0)
        )
        return out

    def induction_regressions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        pooled = df.copy()
        pooled["is_psychedelic"] = (pooled["condition"] != "neutral").astype(int)
        outputs = {}
        for outcome in ["nlp_similarity", "meq_normalized_score"]:
            model = smf.ols(f"{outcome} ~ is_psychedelic", data=pooled).fit()
            coef_table = model.summary2().tables[1].copy()
            coef_table["outcome"] = outcome
            outputs[f"regression_{outcome}"] = coef_table
        return outputs

    def mixed_effects(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        pooled = df.copy()
        pooled["is_psychedelic"] = (pooled["condition"] != "neutral").astype(int)
        outputs = {}
        for outcome in ["nlp_similarity", "meq_normalized_score"]:
            try:
                model = smf.mixedlm(f"{outcome} ~ is_psychedelic", data=pooled, groups=pooled["model"]).fit(reml=False)
                outputs[f"mixedlm_{outcome}"] = pd.DataFrame(model.summary().tables[1])
            except Exception as exc:
                outputs[f"mixedlm_{outcome}_error"] = pd.DataFrame({"error": [str(exc)]})
        return outputs

    def anova_by_substance(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        subset = df[df["condition"] != "neutral"].copy()
        outputs = {}
        for outcome in ["nlp_similarity", "meq_normalized_score"]:
            model = smf.ols(f"{outcome} ~ C(condition)", data=subset).fit()
            table = anova_lm(model, typ=2)
            table["eta_sq"] = table["sum_sq"] / table["sum_sq"].sum()
            outputs[f"anova_substance_{outcome}"] = table
        return outputs

    def anova_by_model(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        outputs = {}
        for outcome in ["nlp_similarity", "meq_normalized_score"]:
            model = smf.ols(f"{outcome} ~ C(model)", data=df).fit()
            table = anova_lm(model, typ=2)
            table["eta_sq"] = table["sum_sq"] / table["sum_sq"].sum()
            outputs[f"anova_model_{outcome}"] = table
        return outputs

    def correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        overall_r, overall_p = stats.pearsonr(df["nlp_similarity"], df["meq_normalized_score"])
        psych = df[df["condition"] != "neutral"]
        psych_r, psych_p = stats.pearsonr(psych["nlp_similarity"], psych["meq_normalized_score"])
        return pd.DataFrame([
            {"subset": "all_runs", "pearson_r": overall_r, "p_value": overall_p, "n": len(df)},
            {"subset": "psychedelic_only", "pearson_r": psych_r, "p_value": psych_p, "n": len(psych)},
        ])


class FigureBuilder:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def figure_1(self, df: pd.DataFrame) -> Path:
        tmp = df.copy()
        tmp["condition_binary"] = np.where(tmp["condition"] == "neutral", "neutral", "psychedelic")
        summary = tmp.groupby("condition_binary").agg(
            nlp_mean=("nlp_similarity", "mean"),
            nlp_sd=("nlp_similarity", "std"),
            meq_mean=("meq_normalized_score", "mean"),
            meq_sd=("meq_normalized_score", "std"),
        ).reindex(["neutral", "psychedelic"])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(summary.index, summary["nlp_mean"], yerr=summary["nlp_sd"], capsize=5)
        axes[0].set_title("Figure 1A. NLP similarity")
        axes[0].set_ylabel("Standardized similarity (0-1)")
        axes[0].set_ylim(0, 1)

        axes[1].bar(summary.index, summary["meq_mean"], yerr=summary["meq_sd"], capsize=5)
        axes[1].set_title("Figure 1B. MEQ-30 score")
        axes[1].set_ylabel("Standardized score (0-1)")
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        out = self.output_dir / "figure_1_induction.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def figure_2(self, df: pd.DataFrame) -> Path:
        tmp = df[df["condition"] != "neutral"].copy()
        order = ["psilocybin", "lsd", "dmt", "ayahuasca", "mescaline"]
        summary = tmp.groupby("condition").agg(
            nlp_mean=("nlp_similarity", "mean"),
            nlp_sd=("nlp_similarity", "std"),
            meq_mean=("meq_normalized_score", "mean"),
            meq_sd=("meq_normalized_score", "std"),
        ).reindex(order)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(summary.index, summary["nlp_mean"], yerr=summary["nlp_sd"], capsize=4)
        axes[0].set_title("Figure 2A. Substance-level NLP similarity")
        axes[0].set_ylabel("Standardized similarity (0-1)")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_ylim(0, 1)

        axes[1].bar(summary.index, summary["meq_mean"], yerr=summary["meq_sd"], capsize=4)
        axes[1].set_title("Figure 2B. Substance-level MEQ-30")
        axes[1].set_ylabel("Standardized score (0-1)")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        out = self.output_dir / "figure_2_substances.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def figure_3(self, df: pd.DataFrame) -> Path:
        summary = df.groupby("model").agg(
            nlp_mean=("nlp_similarity", "mean"),
            nlp_sd=("nlp_similarity", "std"),
            meq_mean=("meq_normalized_score", "mean"),
            meq_sd=("meq_normalized_score", "std"),
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(summary.index, summary["nlp_mean"], yerr=summary["nlp_sd"], capsize=4)
        axes[0].set_title("Figure 3A. Model-level NLP similarity")
        axes[0].set_ylabel("Standardized similarity (0-1)")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_ylim(0, 1)

        axes[1].bar(summary.index, summary["meq_mean"], yerr=summary["meq_sd"], capsize=4)
        axes[1].set_title("Figure 3B. Model-level MEQ-30")
        axes[1].set_ylabel("Standardized score (0-1)")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        out = self.output_dir / "figure_3_models.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def figure_4(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(7, 5))
        for condition, group in df.groupby("condition"):
            ax.scatter(group["nlp_similarity"], group["meq_normalized_score"], label=condition, alpha=0.6, s=18)
        ax.set_xlabel("Standardized NLP similarity (0-1)")
        ax.set_ylabel("Standardized MEQ-30 score (0-1)")
        ax.set_title("Figure 4. Similarity vs. mystical intensity")
        ax.legend(frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        out = self.output_dir / "figure_4_scatter.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def build_all(self, df: pd.DataFrame) -> Dict[str, Path]:
        return {
            "figure_1": self.figure_1(df),
            "figure_2": self.figure_2(df),
            "figure_3": self.figure_3(df),
            "figure_4": self.figure_4(df),
        }


def save_table_dict(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv")
