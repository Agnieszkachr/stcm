"""
stcm/data_loader.py
===================
Data loader for SBLGNT synoptic gospels.

Loads the three synoptic gospels (Matthew, Mark, Luke) and aligns them into
triple-tradition and double-tradition pericope sets using a hard-coded
pericope table derived from the Aland Synopsis numbering.

Triple tradition = passages found in Matt, Mark, and Luke.
Double tradition = passages found in Matt and Luke but NOT Mark (= potential Q).

The pericope table encodes (chapter, verse_start, verse_end) for each gospel.
None means that gospel lacks this pericope.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from stcm.config import default_config
from stcm.utils import load_sblgnt, verses_to_pericope

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pericope alignment table
# ---------------------------------------------------------------------------
# Format per row:
#   (label, matt_ref, mark_ref, luke_ref)
# where each ref is (chapter, v_start, v_end) or None
# Based on Aland Synopsis §§ (representative selection covering major blocks)

TripleEntry = Tuple[str, Optional[Tuple], Optional[Tuple], Optional[Tuple]]

TRIPLE_TRADITION: List[TripleEntry] = [
    # (label, matthew, mark, luke)
    ("Baptism of Jesus",         (3,13,17),  (1,9,11),   (3,21,22)),
    ("Temptation of Jesus",      (4,1,11),   (1,12,13),  (4,1,13)),
    ("Call of first disciples",  (4,18,22),  (1,16,20),  (5,1,11)),
    ("Man with unclean spirit",  (8,28,34),  (1,23,28),  (4,33,37)),
    ("Healing of Peter's mother",(8,14,15),  (1,29,31),  (4,38,39)),
    ("Healing of a leper",       (8,1,4),    (1,40,45),  (5,12,16)),
    ("Healing of the paralytic", (9,1,8),    (2,1,12),   (5,17,26)),
    ("Call of Levi",             (9,9,13),   (2,13,17),  (5,27,32)),
    ("Question about fasting",   (9,14,17),  (2,18,22),  (5,33,39)),
    ("Plucking grain Sabbath",   (12,1,8),   (2,23,28),  (6,1,5)),
    ("Man with withered hand",   (12,9,14),  (3,1,6),    (6,6,11)),
    ("Beelzebul controversy",    (12,22,30), (3,22,27),  (11,14,23)),
    ("Parable of the sower",     (13,1,9),   (4,1,9),    (8,4,8)),
    ("Purpose of parables",      (13,10,17), (4,10,12),  (8,9,10)),
    ("Explanation of sower",     (13,18,23), (4,13,20),  (8,11,15)),
    ("Lamp under bushel",        (5,14,16),  (4,21,25),  (8,16,18)),
    ("Stilling of storm",        (8,23,27),  (4,35,41),  (8,22,25)),
    ("Gerasene demoniac",        (8,28,34),  (5,1,20),   (8,26,39)),
    ("Jairus daughter",          (9,18,26),  (5,21,43),  (8,40,56)),
    ("Rejection at Nazareth",    (13,54,58), (6,1,6),    (4,16,30)),
    ("Mission of the Twelve",    (10,1,15),  (6,7,13),   (9,1,6)),
    ("Death of John Baptist",    (14,1,12),  (6,14,29),  (9,7,9)),
    ("Feeding five thousand",    (14,13,21), (6,30,44),  (9,10,17)),
    ("Walking on water",         (14,22,33), (6,45,52),  None),
    ("Clean and unclean",        (15,1,20),  (7,1,23),   None),
    ("Confession of Peter",      (16,13,20), (8,27,30),  (9,18,21)),
    ("First passion prediction", (16,21,28), (8,31,38),  (9,22,27)),
    ("Transfiguration",          (17,1,13),  (9,2,13),   (9,28,36)),
    ("Healing epileptic boy",    (17,14,21), (9,14,29),  (9,37,43)),
    ("Second passion prediction",(17,22,23), (9,30,32),  (9,43,45)),
    ("Temple tax",               (17,24,27), None,        None),
    ("True greatness",           (18,1,5),   (9,33,37),  (9,46,48)),
    ("Strange exorcist",         (18,6,9),   (9,38,42),  (9,49,50)),
    ("Anointing at Bethany",     (26,6,13),  (14,3,9),   None),
    ("Triumphal entry",          (21,1,11),  (11,1,11),  (19,28,44)),
    ("Cleansing of temple",      (21,12,17), (11,15,19), (19,45,48)),
    ("Cursing of fig tree",      (21,18,22), (11,12,25), None),
    ("Question of authority",    (21,23,27), (11,27,33), (20,1,8)),
    ("Parable of tenants",       (21,33,46), (12,1,12),  (20,9,19)),
    ("Tribute to Caesar",        (22,15,22), (12,13,17), (20,20,26)),
    ("Sadducees resurrection",   (22,23,33), (12,18,27), (20,27,40)),
    ("Greatest commandment",     (22,34,40), (12,28,34), None),
    ("David's son question",     (22,41,46), (12,35,37), (20,41,44)),
    ("Scribes condemned",        (23,1,12),  (12,38,40), (20,45,47)),
    ("Widow's offering",         None,        (12,41,44), (21,1,4)),
    ("Eschatological discourse", (24,1,36),  (13,1,37),  (21,5,36)),
    ("Third passion prediction", (20,17,19), (10,32,34), (18,31,34)),
    ("Entry into Jerusalem",     (21,1,11),  (11,1,10),  (19,29,40)),
    ("Last Supper",              (26,17,30), (14,12,26), (22,7,38)),
    ("Gethsemane",               (26,36,46), (14,32,42), (22,39,46)),
    ("Arrest of Jesus",          (26,47,56), (14,43,52), (22,47,53)),
    ("Peter's denial",           (26,69,75), (14,66,72), (22,54,62)),
    ("Trial before Pilate",      (27,1,26),  (15,1,15),  (23,1,25)),
    ("Crucifixion",              (27,27,56), (15,16,41), (23,26,49)),
    ("Burial",                   (27,57,61), (15,42,47), (23,50,56)),
    ("Empty tomb",               (28,1,10),  (16,1,8),   (24,1,12)),
]

DOUBLE_TRADITION: List[Tuple[str, Tuple, Tuple]] = [
    # (label, matthew_ref, luke_ref) — no Markan parallel
    ("John's preaching",              (3,7,12),   (3,7,9)),
    ("Temptation narrative (full)",   (4,1,11),   (4,1,13)),
    ("Beatitudes",                    (5,3,12),   (6,20,26)),
    ("Love of enemies",               (5,43,48),  (6,27,36)),
    ("Lord's Prayer",                 (6,9,13),   (11,2,4)),
    ("Anxieties about life",          (6,25,34),  (12,22,32)),
    ("Narrow gate",                   (7,13,14),  (13,23,24)),
    ("Centurion's servant",           (8,5,13),   (7,1,10)),
    ("John's question from prison",   (11,2,6),   (7,18,23)),
    ("Jesus on John",                 (11,7,19),  (7,24,35)),
    ("Woes on Galilean cities",       (11,20,24), (10,13,15)),
    ("Hidden from wise revealed",     (11,25,27), (10,21,22)),
    ("Come to me all",                (11,28,30), None),  # Matt only in Q block
    ("Mission discourse",             (10,5,16),  (10,1,12)),
    ("Harvest plentiful",             (9,37,38),  (10,2,3)),
    ("Sign of Jonah",                 (12,38,42), (11,29,32)),
    ("Return of unclean spirit",      (12,43,45), (11,24,26)),
    ("Lamp of the body",              (6,22,23),  (11,34,36)),
    ("Leaven of Pharisees",           (16,6,12),  (12,1,3)),
    ("Fear of God not men",           (10,26,33), (12,4,9)),
    ("Blasphemy Holy Spirit",         (12,31,32), (12,10,12)),
    ("Thief in the night",            (24,43,44), (12,39,40)),
    ("Faithful servant",              (24,45,51), (12,42,48)),
    ("Not peace but sword",           (10,34,36), (12,51,53)),
    ("Reading the signs",             (16,2,3),   (12,54,56)),
    ("Settling with opponent",        (5,25,26),  (12,57,59)),
    ("Mustard seed and leaven",       (13,31,33), (13,18,21)),
    ("Many come from east west",      (8,11,12),  (13,28,29)),
    ("Lament over Jerusalem",         (23,37,39), (13,34,35)),
    ("Parable of Great Banquet",      (22,1,14),  (14,15,24)),
    ("Conditions of discipleship",    (10,37,38), (14,26,27)),
    ("Salt of the earth",             (5,13,13),  (14,34,35)),
    ("Lost sheep",                    (18,12,14), (15,3,7)),
    ("Serving two masters",           (6,24,24),  (16,13,13)),
    ("Kingdom of God within",         None,        (17,20,21)),
    ("Day of the Son of Man",         (24,26,28), (17,23,24)),
    ("Talents / Minas",               (25,14,30), (19,12,27)),
    ("Judging twelve tribes",         (19,28,28), (22,28,30)),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Pericope:
    """A single aligned text unit across gospels."""
    label: str
    tradition: str  # "triple" or "double"
    matthew: Optional[str]  # Greek text or None
    mark: Optional[str]     # Greek text or None
    luke: Optional[str]     # Greek text or None

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

    def _extract(
        self,
        verses: List[Dict],
        ref: Optional[Tuple],
    ) -> Optional[str]:
        if ref is None:
            return None
        ch, vs, ve = ref
        text = verses_to_pericope(verses, ch, vs, ve)
        return text if text.strip() else None

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

        triple: List[Pericope] = []
        for label, m_ref, mk_ref, lk_ref in TRIPLE_TRADITION:
            matt_txt = self._extract(matt_v, m_ref)
            mark_txt = self._extract(mark_v, mk_ref)
            luke_txt = self._extract(luke_v, lk_ref)
            # Keep only if at least two gospels have text
            n_present = sum(t is not None for t in (matt_txt, mark_txt, luke_txt))
            if n_present >= 2:
                triple.append(
                    Pericope(
                        label=label,
                        tradition="triple",
                        matthew=matt_txt,
                        mark=mark_txt,
                        luke=luke_txt,
                    )
                )

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
                    )
                )

        log.info(
            "Loaded corpus: %d triple-tradition + %d double-tradition pericopes.",
            len(triple),
            len(double),
        )
        return SynopticCorpus(triple_tradition=triple, double_tradition=double)
