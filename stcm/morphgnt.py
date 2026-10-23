"""
stcm/morphgnt.py
================
Lemmas for the pericope table from MorphGNT SBLGNT.

MorphGNT (https://github.com/morphgnt/sblgnt) gives, for every word of the
SBL Greek New Testament, its verse reference, part of speech, parsing code,
surface text, word, normalised word and lemma.  The morphological analysis is
licensed CC BY-SA 3.0 (attribution: MorphGNT, James K. Tauber and
contributors); the underlying text is subject to the SBLGNT EULA.

The data are pinned to one commit (MORPHGNT_COMMIT) and are downloaded by
``download_morphgnt.py`` into ``data/raw/morphgnt/``.

Each lemma is normalised exactly as word forms are for the word-form Jaccard
(``stcm.utils.word_tokens``): NFC, accents stripped, lower-cased, punctuation
removed.  Two consequences, both shared with the word-form measure: a lemma
written with an optional final letter, such as οὕτω(ς) or ἔξεστι(ν), is kept
as its first part (ουτω, εξεστι); and accent stripping merges a few lemmas,
for instance εἰς and εἷς, or τίς and τις (with unstripped lemmas the
residual of Section 4.3 is +0.2624 instead of +0.2623).
"""
from __future__ import annotations

import pathlib
import unicodedata
from typing import Dict, List, Optional, Tuple

from stcm.config import default_config
from stcm.data_loader import ranges
from stcm.utils import word_tokens

MORPHGNT_COMMIT = "aaed91e57c8e4a8dc9a2383e129ca5e75fe6393d"
BOOK_FILES = {"matthew": "61-Mt-morphgnt.txt", "mark": "62-Mk-morphgnt.txt", "luke": "63-Lk-morphgnt.txt"}


def morphgnt_dir() -> pathlib.Path:
    return default_config.paths.data_raw / "morphgnt"


def load_book(gospel: str, root: Optional[pathlib.Path] = None) -> Dict[Tuple[int, int], List[Dict[str, str]]]:
    """
    {(chapter, verse): [ {word, norm, lemma, pos}, ... ]} for one gospel.
    MorphGNT columns: BBCCVV, POS, parse, text, word, normalised, lemma.
    """
    root = root or morphgnt_dir()
    out: Dict[Tuple[int, int], List[Dict[str, str]]] = {}
    for line in (root / BOOK_FILES[gospel]).read_text(encoding="utf-8").splitlines():
        cols = line.split()
        if len(cols) < 7:
            continue
        ref = cols[0]
        ch, vs = int(ref[2:4]), int(ref[4:6])
        out.setdefault((ch, vs), []).append(
            dict(text=cols[3], word=cols[4], norm=cols[5], lemma=cols[6], pos=cols[1]))
    return out


def normalise_lemma(lemma: str) -> str:
    """Normalise a lemma as the word-form Jaccard normalises forms; returns '' if nothing is left."""
    toks = word_tokens(unicodedata.normalize("NFC", lemma))
    return toks[0] if toks else ""


def unit_lemmas(book: Dict[Tuple[int, int], List[Dict[str, str]]], ref) -> List[str]:
    """Normalised lemmas of all words in the verse ranges of a unit (union of ranges)."""
    out: List[str] = []
    for ch, a, b in ranges(ref):
        for v in range(a, b + 1):
            for w in book.get((ch, v), []):
                lem = normalise_lemma(w["lemma"])
                if lem:
                    out.append(lem)
    return out


def unit_forms(book: Dict[Tuple[int, int], List[Dict[str, str]]], ref) -> List[str]:
    """Normalised word forms of the same words (for the token-alignment check)."""
    out: List[str] = []
    for ch, a, b in ranges(ref):
        for v in range(a, b + 1):
            for w in book.get((ch, v), []):
                toks = word_tokens(w["word"])
                out.extend(toks)
    return out
