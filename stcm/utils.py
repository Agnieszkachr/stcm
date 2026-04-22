"""
stcm/utils.py
=============
Shared helper functions used across the STCM pipeline.

Includes:
- Unicode normalisation for Ancient Greek
- Verse-reference parsing for SBLGNT line format
- Logging helpers
- Pickle / JSON I/O
- Cosine similarity / angular distance
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import pickle
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise_greek(text: str) -> str:
    """
    Normalise a Greek string for embedding:
    1. NFC normalisation (canonical composition).
    2. Strip soft-hyphens and zero-width characters.
    3. Collapse multiple whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u00AD\u200B\u200C\u200D\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_accents(text: str) -> str:
    """
    Remove diacritics from Greek text (NFD decompose → filter combining chars).
    Used for approximate matching / fallback n-gram embedder.
    """
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------------------
# SBLGNT line-format parsing
# ---------------------------------------------------------------------------
#
# Each line in a SBLGNT .txt file has the format:
#   BOOK CHAPTER:VERSE word1 word2 word3 …
# E.g.: Matt 1:1 Βίβλος γενέσεως Ἰησοῦ Χριστοῦ υἱοῦ Δαυὶδ υἱοῦ Ἀβραάμ.

_VERSE_RE = re.compile(
    r"^(?P<book>\S+)\s+(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.+)$"
)


def parse_sblgnt_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse one SBLGNT verse line.

    Returns a dict with keys: book, chapter (int), verse (int), text (str).
    Returns None if the line does not match the expected format.
    """
    m = _VERSE_RE.match(line.strip())
    if not m:
        return None
    return {
        "book": m.group("book"),
        "chapter": int(m.group("chapter")),
        "verse": int(m.group("verse")),
        "text": normalise_greek(m.group("text")),
    }


def load_sblgnt(path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Load an entire SBLGNT book file and return a list of verse dicts.

    Parameters
    ----------
    path : pathlib.Path
        Path to the .txt file (e.g. data/raw/matthew.txt).

    Returns
    -------
    List[Dict]
        List of dicts with keys book, chapter, verse, text.
    """
    verses: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = parse_sblgnt_line(line)
        if parsed:
            verses.append(parsed)
        else:
            log.debug("Skipping malformed line: %r", line[:60])
    log.info("Loaded %d verses from %s", len(verses), path.name)
    return verses


def verses_to_pericope(
    verses: List[Dict[str, Any]],
    chapter: int,
    verse_start: int,
    verse_end: int,
) -> str:
    """
    Concatenate the text of verses in a given range (inclusive) into one string.

    Parameters
    ----------
    verses      : full list of verse dicts for the book
    chapter     : chapter number
    verse_start : first verse (inclusive)
    verse_end   : last verse (inclusive)

    Returns
    -------
    str : concatenated, normalised Greek text
    """
    selected = [
        v["text"]
        for v in verses
        if v["chapter"] == chapter and verse_start <= v["verse"] <= verse_end
    ]
    return " ".join(selected)


# ---------------------------------------------------------------------------
# Cosine similarity / angular distance
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Angular distance in [0, 1] derived from cosine similarity.
    0 = identical direction, 1 = opposite directions.
    """
    cos = np.clip(cosine_similarity(a, b), -1.0, 1.0)
    return float(np.arccos(cos) / np.pi)


def residual_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Component of *a* orthogonal to *b* (residual after projecting a onto b).
    """
    b_unit = b / (np.linalg.norm(b) + 1e-12)
    return a - np.dot(a, b_unit) * b_unit


# ---------------------------------------------------------------------------
# Hashing / cache keys
# ---------------------------------------------------------------------------

def text_hash(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string (first 16 chars for brevity)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def array_hash(arr: np.ndarray) -> str:
    """SHA-256 of numpy array bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, path: pathlib.Path) -> None:
    """Pickle *obj* to *path*, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved pickle: %s", path)


def load_pickle(path: pathlib.Path) -> Any:
    """Load and return a pickled object from *path*."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    log.info("Loaded pickle: %s", path)
    return obj


def save_json(obj: Any, path: pathlib.Path, indent: int = 2) -> None:
    """Serialise *obj* to JSON at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
    log.info("Saved JSON: %s", path)


def load_json(path: pathlib.Path) -> Any:
    """Load JSON from *path*."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: np.ndarray,
    statistic: Any = np.mean,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data        : 1-D array
    statistic   : callable, default np.mean
    n_bootstrap : number of bootstrap samples
    ci_level    : e.g. 0.95 for 95 % CI
    rng         : optional numpy Generator for reproducibility

    Returns
    -------
    (lower, upper) floats
    """
    if rng is None:
        rng = np.random.default_rng(42)
    samples = [
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - ci_level
    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lower, upper
