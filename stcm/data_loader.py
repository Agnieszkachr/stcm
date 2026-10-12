"""
stcm/data_loader.py
===================
Data loader for SBLGNT synoptic gospels.

Loads the three synoptic gospels (Matthew, Mark, Luke) and aligns them into
triple-tradition and double-tradition pericope sets using a hard-coded
pericope table derived from the Aland Synopsis numbering and, for the
double tradition, the inventory of the International Q Project.

Triple tradition = passages found in Matt, Mark, and Luke with a plain
                   Markan parallel (Mark-Q overlaps are excluded, because
                   there the Matthew-Luke agreement is not evidence of
                   Markan copying).
Double tradition = passages found in Matt and Luke but NOT Mark (= Q).

A reference is either a single contiguous range (chapter, v_start, v_end)
or a list of such ranges, concatenated in the order given.  Lists are used
where the parallel verses are non-contiguous in one gospel (e.g. Luke
3:7-9 + 16-17 for John's preaching), so that each pericope pair contains
only the material that is actually parallel.  None means that gospel lacks
the pericope.

Alignment principles (version 3, October 2026)
----------------------------------------------
* No pericope appears in both the calibration and the test set.
* No verse appears in two different units of the same set.
* Each Matthew-Luke pair is trimmed to the verses that are parallel; purely
  Matthean or Lukan framing verses are left out.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from stcm.config import default_config
from stcm.utils import load_sblgnt, verses_to_pericope

log = logging.getLogger(__name__)

Range = Tuple[int, int, int]
Ref = Union[Range, List[Range]]

# ---------------------------------------------------------------------------
# Pericope alignment tables
# ---------------------------------------------------------------------------

TripleEntry = Tuple[str, Optional[Ref], Optional[Ref], Optional[Ref]]

# (label, matthew, mark, luke) — only pericopes attested in all three gospels
# with an uncontroversial Markan parallel.  Excluded on principle:
#   Temptation (Mark 1:12-13 is a two-verse notice; Matt/Luke agreement is
#   Q material, and the pericope is a test item), Beelzebul controversy and
#   Mission of the Twelve (Mark-Q overlaps; the Matthean mission text is
#   also part of the test set), "Lamp under the bushel" (Matt 5:14-16 has
#   no genuine parallel to Mark 4:21-25), the Capernaum exorcism and the
#   strange exorcist (no Matthean parallel), temple tax and widow's
#   offering (not triple), and the condemnation of the scribes (Mark
#   12:38-40 has only a two-verse Matthean fragment inside Q/M material).
#   Q verses embedded in Markan contexts (Matt 13:16-17; 17:20; 24:26-28;
#   Luke 22:28-30) are cut out of the calibration units so that no verse
#   belongs to both sets.
#   Version 3 (October 2026): "Call of the first disciples" removed, because
#   Luke 5:1-11 is a different episode (the miraculous catch) sharing only
#   5:10b-11 with Mark 1:16-20; "Rejection at Nazareth" restricted on the
#   Lukan side to the verses with a Markan parallel (Luke 4:16, 22, 24),
#   since 4:17-21, 23, 25-30 are Lukan composition.  Rule: a unit enters the
#   calibration set only if each gospel's text is the Markan episode itself,
#   trimmed to the verses that have a parallel in Aland's synopsis.
TRIPLE_TRADITION: List[TripleEntry] = [
    ("Baptism of Jesus",           (3,13,17),  (1,9,11),   (3,21,22)),
    ("Healing of Peter's mother",  (8,14,15),  (1,29,31),  (4,38,39)),
    ("Healing of a leper",         (8,1,4),    (1,40,45),  (5,12,16)),
    ("Healing of the paralytic",   (9,1,8),    (2,1,12),   (5,17,26)),
    ("Call of Levi",               (9,9,13),   (2,13,17),  (5,27,32)),
    ("Question about fasting",     (9,14,17),  (2,18,22),  (5,33,39)),
    ("Plucking grain Sabbath",     (12,1,8),   (2,23,28),  (6,1,5)),
    ("Man with withered hand",     (12,9,14),  (3,1,6),    (6,6,11)),
    ("Jesus' true family",         (12,46,50), (3,31,35),  (8,19,21)),
    ("Parable of the sower",       (13,1,9),   (4,1,9),    (8,4,8)),
    ("Purpose of parables",        (13,10,15), (4,10,12),  (8,9,10)),
    ("Explanation of sower",       (13,18,23), (4,13,20),  (8,11,15)),
    ("Stilling of storm",          (8,23,27),  (4,35,41),  (8,22,25)),
    ("Gerasene demoniac",          (8,28,34),  (5,1,20),   (8,26,39)),
    ("Jairus daughter",            (9,18,26),  (5,21,43),  (8,40,56)),
    ("Rejection at Nazareth",      (13,54,58), (6,1,6),    [(4,16,16), (4,22,22), (4,24,24)]),
    ("Herod's opinion of Jesus",   (14,1,2),   (6,14,16),  (9,7,9)),
    ("Feeding five thousand",      (14,13,21), (6,30,44),  (9,10,17)),
    ("Confession of Peter",        (16,13,20), (8,27,30),  (9,18,21)),
    ("First passion prediction",   (16,21,28), (8,31,38),  (9,22,27)),
    ("Transfiguration",            (17,1,13),  (9,2,13),   (9,28,36)),
    ("Healing epileptic boy",      (17,14,19), (9,14,29),  (9,37,42)),
    ("Second passion prediction",  (17,22,23), (9,30,32),  (9,43,45)),
    ("True greatness",             (18,1,5),   (9,33,37),  (9,46,48)),
    ("Children blessed",           (19,13,15), (10,13,16), (18,15,17)),
    ("Rich young man",             (19,16,22), (10,17,22), (18,18,23)),
    ("Third passion prediction",   (20,17,19), (10,32,34), (18,31,34)),
    ("Blind Bartimaeus",           (20,29,34), (10,46,52), (18,35,43)),
    ("Entry into Jerusalem",       (21,1,11),  (11,1,11),  (19,28,40)),
    ("Cleansing of temple",        (21,12,17), (11,15,19), (19,45,48)),
    ("Question of authority",      (21,23,27), (11,27,33), (20,1,8)),
    ("Parable of tenants",         (21,33,46), (12,1,12),  (20,9,19)),
    ("Tribute to Caesar",          (22,15,22), (12,13,17), (20,20,26)),
    ("Sadducees resurrection",     (22,23,33), (12,18,27), (20,27,40)),
    ("David's son question",       (22,41,46), (12,35,37), (20,41,44)),
    ("Eschatological discourse",   [(24,1,25), (24,29,36)], (13,1,37), (21,5,36)),
    ("Last Supper",                (26,17,30), (14,12,26), (22,7,23)),
    ("Gethsemane",                 (26,36,46), (14,32,42), (22,39,46)),
    ("Arrest of Jesus",            (26,47,56), (14,43,52), (22,47,53)),
    ("Trial before Sanhedrin",     (26,57,68), (14,53,65), (22,63,71)),
    ("Peter's denial",             (26,69,75), (14,66,72), (22,54,62)),
    ("Trial before Pilate",        (27,1,26),  (15,1,15),  (23,1,25)),
    ("Crucifixion",                (27,27,56), (15,16,41), (23,26,49)),
    ("Burial",                     (27,57,61), (15,42,47), (23,50,56)),
    ("Empty tomb",                 (28,1,10),  (16,1,8),   (24,1,12)),
]

DoubleEntry = Tuple[str, Ref, Ref]

# (label, matthew_ref, luke_ref) — no Markan parallel.  Units follow the
# IQP inventory (Robinson, Hoffmann and Kloppenborg 2000); where one
# evangelist scatters the material, the reference is a list of ranges.
DOUBLE_TRADITION: List[DoubleEntry] = [
    ("John's preaching",              (3,7,12),   [(3,7,9), (3,16,17)]),
    ("Temptation narrative (full)",   (4,1,11),   (4,1,13)),
    ("Beatitudes",                    (5,3,12),   (6,20,23)),
    ("Love of enemies",               [(5,39,42), (5,44,48)], [(6,27,30), (6,32,36)]),
    ("Golden rule",                   (7,12,12),  (6,31,31)),
    ("Judging and the speck",         (7,1,5),    [(6,37,38), (6,41,42)]),
    ("Blind guides",                  [(15,14,14), (10,24,25)], (6,39,40)),
    ("Tree and its fruit",            (7,16,20),  (6,43,45)),
    ("House on the rock",             (7,24,27),  (6,47,49)),
    ("Lord's Prayer",                 (6,9,13),   (11,2,4)),
    ("Ask, seek, knock",              (7,7,11),   (11,9,13)),
    ("Treasure in heaven",            (6,19,21),  (12,33,34)),
    ("Anxieties about life",          (6,25,33),  (12,22,31)),
    ("Narrow gate",                   (7,13,14),  (13,23,24)),
    ("Centurion's servant",           [(8,5,10), (8,13,13)], (7,1,10)),
    ("Would-be followers",            (8,19,22),  (9,57,60)),
    ("John's question from prison",   (11,2,6),   (7,18,23)),
    ("Jesus on John",                 [(11,7,11), (11,16,19)], (7,24,35)),
    ("Woes on Galilean cities",       (11,20,24), (10,13,15)),
    ("Hidden from wise revealed",     (11,25,27), (10,21,22)),
    ("Blessed eyes",                  (13,16,17), (10,23,24)),
    ("Mission discourse",             (10,7,16),  (10,3,12)),
    ("Harvest plentiful",             (9,37,38),  (10,2,2)),
    ("Whoever receives you",          (10,40,40), (10,16,16)),
    ("Sign of Jonah",                 (12,38,42), (11,29,32)),
    ("Return of unclean spirit",      (12,43,45), (11,24,26)),
    ("Lamp on a stand",               (5,15,15),  (11,33,33)),
    ("Lamp of the body",              (6,22,23),  (11,34,36)),
    ("Woes against Pharisees",        [(23,4,4), (23,6,7), (23,13,13), (23,23,36)], (11,39,52)),
    ("Fear of God not men",           (10,26,33), (12,2,9)),
    ("Blasphemy Holy Spirit",         (12,32,32), (12,10,10)),
    ("Thief in the night",            (24,43,44), (12,39,40)),
    ("Faithful servant",              (24,45,51), (12,42,48)),
    ("Not peace but sword",           (10,34,36), (12,51,53)),
    ("Reading the signs",             (16,2,3),   (12,54,56)),
    ("Settling with opponent",        (5,25,26),  (12,57,59)),
    ("Mustard seed and leaven",       (13,31,33), (13,18,21)),
    ("Many come from east west",      (8,11,12),  (13,28,29)),
    ("Lament over Jerusalem",         (23,37,39), (13,34,35)),
    ("Parable of Great Banquet",      (22,2,10),  (14,16,24)),
    ("Conditions of discipleship",    (10,37,38), (14,26,27)),
    ("Salt of the earth",             (5,13,13),  (14,34,35)),
    ("Lost sheep",                    (18,12,14), (15,3,7)),
    ("Serving two masters",           (6,24,24),  (16,13,13)),
    ("Law, prophets and divorce",     [(11,12,13), (5,18,18), (5,32,32)], (16,16,18)),
    ("Scandals, forgiveness, faith",  [(18,7,7), (18,15,15), (18,21,22), (17,20,20)], (17,1,6)),
    ("Day of the Son of Man",         (24,26,28), [(17,23,24), (17,37,37)]),
    ("Days of Noah",                  (24,37,39), (17,26,27)),
    ("One taken, one left",           (24,40,41), (17,34,35)),
    ("Talents / Minas",               (25,14,30), (19,12,27)),
    ("Judging twelve tribes",         (19,28,28), (22,28,30)),
]


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def ranges(ref: Optional[Ref]) -> List[Range]:
    """Return a reference as a list of contiguous ranges."""
    if ref is None:
        return []
    if isinstance(ref, tuple):
        return [ref]
    return list(ref)


def format_ref(ref: Optional[Ref]) -> str:
    """Human-readable reference, e.g. '3:7–9, 16–17' or '8:5–10, 13'."""
    parts: List[str] = []
    last_ch = None
    for ch, a, b in ranges(ref):
        v = f"{a}" if a == b else f"{a}–{b}"
        parts.append(v if ch == last_ch else f"{ch}:{v}")
        last_ch = ch
    return ", ".join(parts) if parts else "—"


def n_verses(ref: Optional[Ref]) -> int:
    return sum(b - a + 1 for _, a, b in ranges(ref))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Pericope:
    """
    A single aligned text unit across gospels.

    ``matthew`` / ``mark`` / ``luke`` hold the unit's text with its ranges
    concatenated (None when the gospel lacks the pericope); ``*_parts`` hold
    the same text as a list of one string per contiguous verse range, so
    that composite units (verses gathered from different chapters) can be
    encoded range by range rather than as one artificial sequence.
    """
    label: str
    tradition: str  # "triple" or "double"
    matthew: Optional[str]  # Greek text or None
    mark: Optional[str]     # Greek text or None
    luke: Optional[str]     # Greek text or None
    matthew_parts: List[str] = field(default_factory=list)
    mark_parts: List[str] = field(default_factory=list)
    luke_parts: List[str] = field(default_factory=list)

    def parts(self, gospel: str) -> List[str]:
        return getattr(self, f"{gospel}_parts")

    def is_composite(self, gospel: str) -> bool:
        return len(self.parts(gospel)) > 1

    def available_gospels(self) -> Dict[str, str]:
        """Return dict of {gospel_name: text} for non-None gospels."""
        d: Dict[str, str] = {}
        if self.matthew:
            d["matthew"] = self.matthew
        if self.mark:
            d["mark"] = self.mark
        if self.luke:
            d["luke"] = self.luke
        return d


@dataclass
class SynopticCorpus:
    """Full loaded corpus."""
    triple_tradition: List[Pericope]
    double_tradition: List[Pericope]

    @property
    def all_pericopes(self) -> List[Pericope]:
        return self.triple_tradition + self.double_tradition


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class SBLGNTLoader:
    """
    Load SBLGNT files and extract pericopes per the alignment table.

    Parameters
    ----------
    data_raw : pathlib.Path
        Directory containing matthew.txt, mark.txt, luke.txt
    """

    def __init__(self, data_raw: Optional[pathlib.Path] = None) -> None:
        self._raw = data_raw or default_config.paths.data_raw

    def _load_books(self) -> Dict[str, List]:
        books: Dict[str, List] = {}
        for name in ("matthew", "mark", "luke"):
            p = self._raw / f"{name}.txt"
            if not p.exists():
                raise FileNotFoundError(
                    f"SBLGNT file not found: {p}. "
                    "Run python download_sblgnt.py first."
                )
            books[name] = load_sblgnt(p)
        return books

    def _extract_parts(
        self,
        verses: List[Dict],
        ref: Optional[Ref],
    ) -> List[str]:
        """One string per contiguous verse range (empty list if ref is None)."""
        if ref is None:
            return []
        parts = []
        for ch, vs, ve in ranges(ref):
            text = verses_to_pericope(verses, ch, vs, ve)
            if not text.strip():
                raise ValueError(f"Empty verse range {ch}:{vs}-{ve}")
            parts.append(text.strip())
        return parts

    def _extract(
        self,
        verses: List[Dict],
        ref: Optional[Ref],
    ) -> Optional[str]:
        parts = self._extract_parts(verses, ref)
        text = " ".join(parts)
        return text if text.strip() else None

    @staticmethod
    def _check_no_overlap(entries: Sequence[Tuple[str, Optional[Ref]]], gospel: str) -> None:
        seen: Dict[Tuple[int, int], str] = {}
        for label, ref in entries:
            for ch, a, b in ranges(ref):
                for v in range(a, b + 1):
                    if (ch, v) in seen:
                        raise ValueError(
                            f"{gospel} {ch}:{v} used by both '{seen[(ch, v)]}' and '{label}'"
                        )
                    seen[(ch, v)] = label

    def load(self) -> SynopticCorpus:
        """
        Load SBLGNT files and build a SynopticCorpus.

        Returns
        -------
        SynopticCorpus
        """
        books = self._load_books()
        matt_v = books["matthew"]
        mark_v = books["mark"]
        luke_v = books["luke"]

        # Integrity checks: no verse in two units of the same set, and no
        # Matthew/Luke verse shared between the calibration and test sets.
        self._check_no_overlap([(l, m) for l, m, _, _ in TRIPLE_TRADITION]
                               + [(l, m) for l, m, _ in DOUBLE_TRADITION], "Matthew")
        self._check_no_overlap([(l, k) for l, _, k, _ in TRIPLE_TRADITION], "Mark")
        self._check_no_overlap([(l, k) for l, _, _, k in TRIPLE_TRADITION]
                               + [(l, k) for l, _, k in DOUBLE_TRADITION], "Luke")

        triple: List[Pericope] = []
        for label, m_ref, mk_ref, lk_ref in TRIPLE_TRADITION:
            matt_txt = self._extract(matt_v, m_ref)
            mark_txt = self._extract(mark_v, mk_ref)
            luke_txt = self._extract(luke_v, lk_ref)
            if matt_txt and mark_txt and luke_txt:
                triple.append(
                    Pericope(
                        label=label,
                        tradition="triple",
                        matthew=matt_txt,
                        mark=mark_txt,
                        luke=luke_txt,
                        matthew_parts=self._extract_parts(matt_v, m_ref),
                        mark_parts=self._extract_parts(mark_v, mk_ref),
                        luke_parts=self._extract_parts(luke_v, lk_ref),
                    )
                )
            else:
                raise ValueError(f"Triple-tradition entry '{label}' lacks a gospel text.")

        double: List[Pericope] = []
        for label, m_ref, lk_ref in DOUBLE_TRADITION:
            matt_txt = self._extract(matt_v, m_ref)
            luke_txt = self._extract(luke_v, lk_ref)
            if matt_txt and luke_txt:
                double.append(
                    Pericope(
                        label=label,
                        tradition="double",
                        matthew=matt_txt,
                        mark=None,
                        luke=luke_txt,
                        matthew_parts=self._extract_parts(matt_v, m_ref),
                        luke_parts=self._extract_parts(luke_v, lk_ref),
                    )
                )
            else:
                raise ValueError(f"Double-tradition entry '{label}' lacks a gospel text.")

        log.info(
            "Loaded corpus: %d triple-tradition + %d double-tradition pericopes.",
            len(triple),
            len(double),
        )
        return SynopticCorpus(triple_tradition=triple, double_tradition=double)
