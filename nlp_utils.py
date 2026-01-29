"""General NLP helpers (NLTK downloads, stop words cache)."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Set

import nltk
from nltk.corpus import stopwords

REQUIRED_NLTK_PACKAGES = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "stopwords": "corpora/stopwords",
}


def ensure_nltk_data(packages: Iterable[str] = REQUIRED_NLTK_PACKAGES.keys()) -> None:
    """Download required corpora if they are missing."""
    for pkg in packages:
        resource_path = REQUIRED_NLTK_PACKAGES.get(pkg, pkg)
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def get_stopwords(language: str = "english") -> Set[str]:
    ensure_nltk_data(["stopwords"])
    return set(stopwords.words(language))
