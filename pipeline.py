"""CLI pipeline for running adversarial summarization attacks."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List

import editdistance
from tqdm import tqdm

from . import attacks, data, metrics, models
from .nlp_utils import ensure_nltk_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial perturbations on summarization models.")
    parser.add_argument("--dataset", default="cnn_dailymail", help="Dataset key (default: cnn_dailymail).")
    parser.add_argument("--split", default="test[:20]", help="Dataset split expression.")
    parser.add_argument("--n-samples", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument("--model", default="t5-small", help="Summarization model identifier.")
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=["word:synonym", "word:delete", "sentence:paraphrase"],
        help="List of perturbation specs (e.g., word:synonym).",
    )
    parser.add_argument("--output-dir", default="artifacts/run", help="Directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--success-drop",
        type=float,
        default=0.05,
        help="Minimum ROUGE-1 drop to count attack as success.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    ensure_nltk_data()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    dataset_config = data.DatasetConfig(
        name=args.dataset,
        split=args.split,
        sample_limit=args.n_samples,
    )
    dataset = data.load_dataset_split(dataset_config)
    samples = list(data.iter_samples(dataset, dataset_config))
    logging.info("Loaded %d samples from %s split %s", len(samples), args.dataset, args.split)

    device = models.default_device()
    sbert_model = models.load_sentence_transformer()

    results: List[dict] = []

    for sample in tqdm(samples, desc="Attacking samples"):
        article = sample["article"]
        reference = sample["reference"]
        baseline_summary = models.generate_summary(article, model_name=args.model, device=device)
        baseline_r1 = metrics.rouge_fmeasure(baseline_summary, reference, "rouge1")

        for perturbation in args.perturbations:
            attack_result = attacks.apply_perturbation(
                {"article": article, "reference": reference},
                baseline_summary,
                perturbation,
                rng=rng,
                sbert_model=sbert_model,
            )
            adv_input = attack_result.display_document["article"]
            adv_summary = models.generate_summary(adv_input, model_name=args.model, device=device)
            adv_r1 = metrics.rouge_fmeasure(adv_summary, reference, "rouge1")
            success = adv_r1 <= (baseline_r1 - args.success_drop)

            num_changes = editdistance.eval(baseline_summary, adv_summary)

            results.append(
                {
                    "sample_id": sample["id"],
                    "article": article,
                    "reference": reference,
                    "orig_summary": baseline_summary,
                    "adv_summary": adv_summary,
                    "perturbation_type": perturbation,
                    "model": args.model,
                    "dataset": args.dataset,
                    "success": success,
                    "num_changes": num_changes,
                    "queries": 2,  # baseline + adversarial generation
                    "original_element": attack_result.original_element,
                    "perturbed_element": attack_result.perturbed_element,
                }
            )

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    logging.info("Wrote %s", results_path)

    df = metrics.results_to_df(results)
    metrics_path = output_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    logging.info("Saved metrics table to %s", metrics_path)

    attack_table = metrics.aggregate_word_level(df, metric="attack")
    attack_table_percent = attack_table.copy()
    for col in attack_table_percent.columns:
        if col not in ["dataset", "model"]:
            attack_table_percent[col] = (attack_table_percent[col] * 100).round(2)
    attack_table_path = output_dir / "attack_success_table_percent.csv"
    attack_table_percent.to_csv(attack_table_path, index=False)
    logging.info("Saved attack success table to %s", attack_table_path)

    cnn_dm_table = metrics.build_cnn_dm_asr_table(results, pert_order=args.perturbations)
    cnn_table_path = output_dir / "cnn_dm_asr_table.csv"
    cnn_dm_table.to_csv(cnn_table_path, index=False)
    logging.info("Saved CNN/DM ASR table to %s", cnn_table_path)


if __name__ == "__main__":
    run()
