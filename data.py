"""Dataset utilities for adversarial summarization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class DatasetConfig:
    """Simple configuration for selecting a dataset split."""

    name: str = "cnn_dailymail"
    split: str = "test[:100]"
    sample_limit: Optional[int] = None
    subset: Optional[str] = "3.0.0"


def _load_cnn_dailymail(split: str, subset: str | None) -> Dataset:
    subset_name = subset or "3.0.0"
    return load_dataset("cnn_dailymail", subset_name, split=split)


DATASET_LOADERS: Dict[str, callable] = {
    "cnn_dailymail": _load_cnn_dailymail,
}


def load_dataset_split(config: DatasetConfig) -> Dataset:
    """Return a Hugging Face Dataset for the requested config."""
    if config.name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset '{config.name}'. Known: {sorted(DATASET_LOADERS)}")
    loader = DATASET_LOADERS[config.name]
    return loader(config.split, config.subset)


def iter_samples(dataset: Dataset, config: DatasetConfig) -> Iterator[Dict[str, str]]:
    """
    Yield `{"article": ..., "reference": ..., "id": ...}` dicts honoring sample_limit.
    """
    limit = config.sample_limit if config.sample_limit is not None else len(dataset)
    for idx, record in enumerate(dataset):
        if idx >= limit:
            break
        yield {
            "id": record.get("id", idx),
            "article": record["article"],
            "reference": record.get("highlights", record.get("summary", "")),
        }
