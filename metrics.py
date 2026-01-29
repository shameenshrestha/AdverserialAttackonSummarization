"""Evaluation helpers (ROUGE, SBERT similarity, aggregation)."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import util

from . import models

SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge_fmeasure(summary: str, reference: str, rouge_type: str = "rouge1") -> float:
    if not summary or not reference:
        return 0.0
    return float(SCORER.score(reference, summary)[rouge_type].fmeasure)


def semantic_similarity(a: str, b: str, model_name: str = "default") -> float:
    if not a or not b:
        return 0.0
    sbert = models.load_sentence_transformer(model_name)
    emb_a = sbert.encode(a, convert_to_tensor=True)
    emb_b = sbert.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).item())


def results_to_df(results: Iterable[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for ex in results:
        article = ex.get("article", "")
        orig = ex.get("orig_summary", "")
        adv = ex.get("adv_summary", "")
        pert = ex.get("perturbation_type", "NONE")
        model_name = ex.get("model", "MODEL")
        dataset = ex.get("dataset", "DATASET")
        success = bool(ex.get("success", False))
        num_changes = int(ex.get("num_changes", 0))
        queries = ex.get("queries", np.nan)

        r1_before = rouge_fmeasure(orig, article, "rouge1")
        r1_after = rouge_fmeasure(adv, article, "rouge1")
        r2_before = rouge_fmeasure(orig, article, "rouge2")
        r2_after = rouge_fmeasure(adv, article, "rouge2")
        rl_before = rouge_fmeasure(orig, article, "rougeL")
        rl_after = rouge_fmeasure(adv, article, "rougeL")
        sim = semantic_similarity(orig, adv)

        num_tokens = len(orig.split()) if orig else 0
        changed_rate = (num_changes / num_tokens) if num_tokens > 0 else 0.0

        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "perturb": pert,
                "success": success,
                "num_changes": num_changes,
                "queries": queries,
                "article": article,
                "orig_summary": orig,
                "adv_summary": adv,
                "r1_before": r1_before,
                "r1_after": r1_after,
                "r2_before": r2_before,
                "r2_after": r2_after,
                "rl_before": rl_before,
                "rl_after": rl_after,
                "sim_orig_adv": sim,
                "changed_rate": changed_rate,
            }
        )
    df = pd.DataFrame(rows)
    return add_attack_success_rate(df)


def aggregate_word_level(df: pd.DataFrame, metric: str = "attack") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset", "model"])
    group_cols = ["dataset", "model"]
    pert_list = sorted(df["perturb"].unique())
    out_rows = []
    for (dataset, model_name), group in df.groupby(group_cols):
        row = {"dataset": dataset, "model": model_name}
        for perturb in pert_list:
            gp = group[group["perturb"] == perturb]
            if gp.empty:
                row[perturb] = np.nan
                continue
            if metric == "attack":
                val = gp["success"].mean()
            elif metric == "rouge":
                val = gp["r1_after"].mean()
            elif metric == "semantic":
                val = gp["sim_orig_adv"].mean()
            elif metric == "changed_rate":
                val = gp["changed_rate"].mean()
            elif metric == "avg_queries":
                val = gp["queries"].mean()
            else:
                val = np.nan
            row[perturb] = val
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def aggregated_summary_row(df: pd.DataFrame, dataset: str, model: str, perturb: str):
    group = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["perturb"] == perturb)]
    if group.empty:
        return None
    return {
        "dataset": dataset,
        "model": model,
        "perturb": perturb,
        "attack_success_rate": float(group["success"].mean()),
        "avg_changed_rate": float(group["changed_rate"].mean()),
        "avg_queries": float(group["queries"].mean()),
        "avg_sim": float(group["sim_orig_adv"].mean()),
        "r1_before": float(group["r1_before"].mean()),
        "r1_after": float(group["r1_after"].mean()),
    }


def build_cnn_dm_asr_table(results: Iterable[dict], pert_order: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(results).rename(columns={"perturbation_type": "perturb"})
    if df.empty:
        return pd.DataFrame()
    df["dataset"] = df.get("dataset", "CNN/DailyMail")

    base_mask = df["perturb"].isin(["NONE", "orig", "original"])
    baselines = (
        df[base_mask]
        .groupby(["dataset", "model"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "Before Perturbation"})
    )

    asr_after = (
        df[~base_mask]
        .groupby(["dataset", "model", "perturb"])["success"]
        .mean()
        .reset_index()
    )

    asr_wide = (
        asr_after.pivot_table(index=["dataset", "model"], columns="perturb", values="success")
        .reset_index()
        .fillna(0.0)
    )

    table = pd.merge(baselines, asr_wide, on=["dataset", "model"], how="outer")

    for col in table.columns:
        if col not in ["dataset", "model"]:
            table[col] = (table[col] * 100).round(2)

    if pert_order:
        ordered_cols = ["dataset", "model", "Before Perturbation"] + [p for p in pert_order if p in table.columns]
        table = table[ordered_cols]
    return table.sort_values(["dataset", "model"]).reset_index(drop=True)


def add_attack_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate per-sample rows with the attack success rate of their (dataset, model, perturb) group.
    """
    if df.empty:
        df["attack_success_rate"] = pd.Series(dtype=float)
        return df
    rates = (
        df.groupby(["dataset", "model", "perturb"])["success"]
        .mean()
        .rename("attack_success_rate")
        .reset_index()
    )
    return df.merge(rates, on=["dataset", "model", "perturb"], how="left")
