"""
Tests for the lemma baseline: the MorphGNT words must align one to one with
the SBLGNT word tokens used by the word-form Jaccard, and the word-form
Jaccard must reproduce its published values.  Run with ``python -m pytest tests``
(requires data/raw/{matthew,mark,luke}.txt and data/raw/morphgnt/, see
download_sblgnt.py and download_morphgnt.py).
"""
import pathlib
import sys

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from stcm.data_loader import SBLGNTLoader, TRIPLE_TRADITION, DOUBLE_TRADITION  # noqa: E402
from stcm.utils import word_tokens  # noqa: E402

RAW = _REPO / "data" / "raw"
pytestmark = pytest.mark.skipif(
    not ((RAW / "matthew.txt").exists() and (RAW / "morphgnt" / "61-Mt-morphgnt.txt").exists()),
    reason="SBLGNT or MorphGNT files not downloaded")

ELISION = "ʼ"   # SBLGNT keeps the modifier apostrophe; MorphGNT's word column drops it


@pytest.fixture(scope="module")
def corpus():
    return SBLGNTLoader().load()


@pytest.fixture(scope="module")
def books():
    from stcm.morphgnt import load_book
    return {g: load_book(g) for g in ("matthew", "mark", "luke")}


def _sblgnt_tokens(text):
    return [t.replace(ELISION, "") for t in word_tokens(text)]


@pytest.mark.parametrize("gospel,col", [("matthew", 0), ("luke", 1)])
def test_double_tradition_tokens_align(corpus, books, gospel, col):
    from stcm.morphgnt import unit_forms
    refs = {l: (m, k) for l, m, k in DOUBLE_TRADITION}
    for p in corpus.double_tradition:
        assert unit_forms(books[gospel], refs[p.label][col]) == _sblgnt_tokens(getattr(p, gospel)), p.label


@pytest.mark.parametrize("gospel,col", [("matthew", 0), ("mark", 1), ("luke", 2)])
def test_triple_tradition_tokens_align(corpus, books, gospel, col):
    from stcm.morphgnt import unit_forms
    refs = {l: (m, k, lk) for l, m, k, lk in TRIPLE_TRADITION}
    for p in corpus.triple_tradition:
        assert unit_forms(books[gospel], refs[p.label][col]) == _sblgnt_tokens(getattr(p, gospel)), p.label


def test_lemmas_one_per_word_and_normalised(corpus, books):
    from stcm.morphgnt import unit_lemmas, unit_forms
    for l, m, k in DOUBLE_TRADITION:
        lem = unit_lemmas(books["matthew"], m)
        assert len(lem) == len(unit_forms(books["matthew"], m))
        assert all(x and x == x.lower() and x == word_tokens(x)[0] for x in lem)


def test_word_form_jaccard_unchanged(corpus):
    """The word-form Jaccard excess reproduces Table 5 (0.313 DT, 0.167 TT)."""
    from baselines import jaccard_matrix
    from stcm.analysis_core import floors
    d, t = corpus.double_tradition, corpus.triple_tradition
    fd = floors(jaccard_matrix([p.matthew for p in d], [p.luke for p in d]))
    ft = floors(jaccard_matrix([p.matthew for p in t], [p.luke for p in t]))
    assert round(fd["excess"], 3) == 0.313
    assert round(ft["excess"], 3) == 0.167
