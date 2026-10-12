"""
Integrity tests for the pericope tables, the metadata and the published
numbers.  Run with ``python -m pytest tests`` (the SBLGNT files must be
present in data/raw; no model is needed).
"""
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from stcm.data_loader import SBLGNTLoader, TRIPLE_TRADITION, DOUBLE_TRADITION, ranges  # noqa: E402
from stcm.metadata import (THEMATIC_TAGS, FORM, STRATUM, GOULDER_REDACTION_LABELS,  # noqa: E402
                           MARK_Q_OVERLAP, TEXT_CRITICAL_NOTES)

RAW = _REPO / "data" / "raw"
pytestmark = pytest.mark.skipif(not (RAW / "matthew.txt").exists(), reason="SBLGNT files not downloaded")


@pytest.fixture(scope="module")
def corpus():
    return SBLGNTLoader().load()


def test_sizes(corpus):
    assert len(corpus.triple_tradition) == 45
    assert len(corpus.double_tradition) == 51


def test_no_verse_in_two_units_or_two_sets():
    # The loader raises on violation; here we assert the tables are disjoint directly.
    def verses(ref):
        return {(ch, v) for ch, a, b in ranges(ref) for v in range(a, b + 1)}
    matt_t = [verses(m) for _, m, _, _ in TRIPLE_TRADITION]
    matt_d = [verses(m) for _, m, _ in DOUBLE_TRADITION]
    luke_t = [verses(l) for _, _, _, l in TRIPLE_TRADITION]
    luke_d = [verses(l) for _, _, l in DOUBLE_TRADITION]
    for group in (matt_t + matt_d, luke_t + luke_d):
        seen = set()
        for s in group:
            assert not (seen & s)
            seen |= s


def test_every_double_tradition_unit_is_annotated(corpus):
    labels = {p.label for p in corpus.double_tradition}
    assert labels == set(THEMATIC_TAGS)
    assert labels == set(FORM)
    assert set(STRATUM) <= labels
    assert set(GOULDER_REDACTION_LABELS) <= labels
    assert set(MARK_Q_OVERLAP) <= labels
    assert set(TEXT_CRITICAL_NOTES) <= labels


def test_composite_units_have_parts(corpus):
    for p in corpus.all_pericopes:
        for g in ("matthew", "mark", "luke"):
            parts = p.parts(g)
            text = getattr(p, g)
            if text is None:
                assert parts == []
            else:
                assert " ".join(parts) == text


def test_key_stats_match_corpus(corpus):
    ks = _REPO / "outputs" / "reports" / "key_stats.json"
    if not ks.exists():
        pytest.skip("key_stats.json not generated")
    key = json.loads(ks.read_text(encoding="utf-8"))
    assert key["n_double"] == len(corpus.double_tradition)
    assert key["n_triple"] == len(corpus.triple_tradition)
    assert key["mark_q_overlap"]["n_overlap"] == len(MARK_Q_OVERLAP)
