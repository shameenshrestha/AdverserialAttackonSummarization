"""Model loader utilities (summarizers, paraphrasers, embedders)."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ALIASES: Dict[str, str] = {
    "t5-small": "t5-small",
    "bart": "facebook/bart-large-cnn",
    "pegasus": "google/pegasus-cnn_dailymail",
    "distilbart": "sshleifer/distilbart-cnn-12-6",
}

PARAPHRASER_ALIAS = {
    "dipper": "kalpeshk2011/dipper-paraphraser-xxl",
    "pegasus_paraphrase": "tuner007/pegasus_paraphrase",
}

SBERT_ALIAS = {
    "default": "all-MiniLM-L6-v2",
}


def _resolve_model_name(name: str, mapping: Dict[str, str]) -> str:
    return mapping.get(name, name)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=None)
def load_seq2seq(model_name: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    resolved = _resolve_model_name(model_name, MODEL_ALIASES)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForSeq2SeqLM.from_pretrained(resolved)
    return tokenizer, model


def generate_summary(
    text: str,
    model_name: str = "t5-small",
    device: torch.device | None = None,
    max_length: int = 150,
    min_length: int = 40,
    num_beams: int = 4,
    length_penalty: float = 2.0,
    early_stopping: bool = True,
    prefix: str = "summarize: ",
    **generate_kwargs,
) -> str:
    """Generate a summary for the provided text."""
    tokenizer, model = load_seq2seq(model_name)
    device = device or default_device()
    model = model.to(device)
    input_ids = tokenizer.encode(
        prefix + text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(device)
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        **generate_kwargs,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@lru_cache(maxsize=None)
def load_paraphraser(model_name: str = "pegasus_paraphrase") -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    resolved = _resolve_model_name(model_name, PARAPHRASER_ALIAS)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForSeq2SeqLM.from_pretrained(resolved)
    return tokenizer, model


def paraphrase(
    sentence: str,
    model_name: str = "pegasus_paraphrase",
    device: torch.device | None = None,
    max_length: int = 512,
    num_beams: int = 5,
) -> str:
    """Return a paraphrased version of the given sentence (best-effort)."""
    tokenizer, model = load_paraphraser(model_name)
    device = device or default_device()
    model = model.to(device)
    inputs = tokenizer(sentence, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1, num_beams=num_beams)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@lru_cache(maxsize=None)
def load_sentence_transformer(model_name: str = "default") -> SentenceTransformer:
    resolved = _resolve_model_name(model_name, SBERT_ALIAS)
    return SentenceTransformer(resolved)
