"""Attack orchestration utilities."""

from __future__ import annotations

import dataclasses
import logging
import random
import re
from typing import Dict, Mapping, Optional

import nltk

from .nlp_utils import ensure_nltk_data
from .perturbations import character as char_p
from .perturbations import sentence as sent_p
from .perturbations import word as word_p

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class PerturbationResult:
    original_document: Mapping[str, str]
    perturbed_document: Mapping[str, str]
    display_document: Mapping[str, str]
    original_element: Optional[str]
    perturbed_element: Optional[str]
    perturbation_type: str


def _split_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        return spec.split(":", 1)
    if "_" in spec:
        return spec.split("_", 1)
    raise ValueError(f"Invalid perturbation spec '{spec}'. Expected format 'level:method'.")


def apply_perturbation(
    document: Mapping[str, str],
    generated_summary: str,
    perturbation_type: str,
    rng: Optional[random.Random] = None,
    sbert_model=None,
) -> PerturbationResult:
    """Apply the requested perturbation to the provided document."""
    ensure_nltk_data(["punkt"])
    generator = rng or random
    document_text = document["article"]
    original_document = dict(document)
    perturbed_document = dict(document)
    display_document = dict(document)

    main_type, method = _split_spec(perturbation_type)

    if main_type == "document":
        perturbed_text = sent_p.document_sentence_reorder(document_text, proportion=1.0, rng=generator)
        perturbed_document["article"] = perturbed_text
        display_document["article"] = perturbed_text
        return PerturbationResult(
            original_document=original_document,
            perturbed_document=perturbed_document,
            display_document=display_document,
            original_element=None,
            perturbed_element=perturbed_text,
            perturbation_type=perturbation_type,
        )

    important_elements = word_p.identify_important_words_and_phrases(document_text, generated_summary)

    if not important_elements:
        LOGGER.warning("No important elements identified for perturbation.")
        return PerturbationResult(
            original_document=original_document,
            perturbed_document=original_document,
            display_document=original_document,
            original_element=None,
            perturbed_element=None,
            perturbation_type=perturbation_type,
        )

    element_to_perturb = generator.choice(important_elements)
    LOGGER.info("Element chosen for perturbation: %s", element_to_perturb)

    sentences = nltk.sent_tokenize(document_text)
    target_sentence = next((s for s in sentences if element_to_perturb.lower() in s.lower()), None)
    if not target_sentence:
        LOGGER.warning("Sentence containing '%s' not found.", element_to_perturb)
        return PerturbationResult(
            original_document=original_document,
            perturbed_document=original_document,
            display_document=original_document,
            original_element=None,
            perturbed_element=None,
            perturbation_type=perturbation_type,
        )

    perturbed_word = None
    display_text = document_text
    perturbed_text = document_text
    if main_type == "character":
        perturbed_word = char_p.perturb(element_to_perturb, method, rng=generator)
        pattern = re.compile(re.escape(element_to_perturb), re.IGNORECASE)
        perturbed_sentence = pattern.sub(perturbed_word, target_sentence)
        display_text = document_text.replace(target_sentence, perturbed_sentence, 1)
        perturbed_text = document_text.replace(target_sentence, "", 1)
    elif main_type == "word":
        perturbed_word = word_p.perturb(
            element_to_perturb,
            method,
            rng=generator,
            original_text=document_text,
            generated_summary=generated_summary,
            sbert_model=sbert_model,
        )
        pattern = re.compile(r"\b" + re.escape(element_to_perturb) + r"\b", re.IGNORECASE)
        replacement = perturbed_word if perturbed_word else ""
        perturbed_sentence = pattern.sub(replacement, target_sentence)
        display_text = document_text.replace(target_sentence, perturbed_sentence, 1)
        perturbed_text = document_text.replace(target_sentence, "", 1)
    elif main_type == "sentence":
        perturbed_sentence = sent_p.perturb(target_sentence, method, rng=generator)
        display_text = document_text.replace(target_sentence, perturbed_sentence, 1)
        perturbed_text = document_text.replace(target_sentence, "", 1)
    else:
        LOGGER.warning("Perturbation type '%s' not recognized.", main_type)
        return PerturbationResult(
            original_document=original_document,
            perturbed_document=original_document,
            display_document=original_document,
            original_element=None,
            perturbed_element=None,
            perturbation_type=perturbation_type,
        )

    LOGGER.info("Original sentence: %s", target_sentence)
    LOGGER.info("Perturbed sentence: %s", perturbed_sentence)

    perturbed_document["article"] = perturbed_text
    display_document["article"] = display_text

    return PerturbationResult(
        original_document=original_document,
        perturbed_document=perturbed_document,
        display_document=display_document,
        original_element=element_to_perturb,
        perturbed_element=perturbed_word if perturbed_word else perturbed_sentence,
        perturbation_type=perturbation_type,
    )


PERTURBATION_REGISTRY: Dict[str, str] = {
    "CI": "character:insert",
    "CD": "character:delete",
    "CR": "character:swap",
    "CS": "character:homoglyph",
    "WD": "word:delete",
    "WRH": "word:homoglyph",
    "WRS": "word:synonym",
    "SR": "sentence:reorder",
    "SRH": "sentence:homoglyph",
    "SRP": "sentence:paraphrase",
    "DR": "document:reorder",  # placeholder for potential doc-level operations
}
