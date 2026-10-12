"""
stcm/metadata.py
================
Per-pericope annotations for the double tradition, kept in one place so
that the evaluation module, the advanced analysis and the figure script
cannot drift apart.

THEMATIC_TAGS  — broad thematic category used by the thematic permutation
                 null (seven categories).
FORM           — form-critical class used in the confound analysis.
STRATUM        — pericope-level approximation of Kloppenborg's (1987)
                 stratigraphy: Q1 formative sapiential layer, Q2 main
                 redaction (prophetic judgement), Q3 temptation narrative,
                 U left unassigned in his analysis.  Pericopes straddling
                 strata are assigned by their dominant component.
GOULDER_REDACTION_LABELS — pericopes that Goulder (1989) analyses as
                 showing Lukan redaction of Matthew.
"""
from __future__ import annotations

from typing import Dict, List

THEMATIC_TAGS: Dict[str, str] = {
    "John's preaching":              "prophetic",
    "Temptation narrative (full)":   "narrative",
    "Beatitudes":                    "wisdom",
    "Love of enemies":               "wisdom",
    "Golden rule":                   "wisdom",
    "Judging and the speck":         "wisdom",
    "Blind guides":                  "wisdom",
    "Tree and its fruit":            "wisdom",
    "House on the rock":             "wisdom",
    "Lord's Prayer":                 "liturgical",
    "Ask, seek, knock":              "wisdom",
    "Treasure in heaven":            "wisdom",
    "Anxieties about life":          "wisdom",
    "Narrow gate":                   "wisdom",
    "Centurion's servant":           "narrative",
    "Would-be followers":            "discipleship",
    "John's question from prison":   "prophetic",
    "Jesus on John":                 "prophetic",
    "Woes on Galilean cities":       "prophetic",
    "Hidden from wise revealed":     "wisdom",
    "Blessed eyes":                  "prophetic",
    "Mission discourse":             "discipleship",
    "Harvest plentiful":             "discipleship",
    "Whoever receives you":          "discipleship",
    "Sign of Jonah":                 "prophetic",
    "Return of unclean spirit":      "prophetic",
    "Lamp on a stand":               "wisdom",
    "Lamp of the body":              "wisdom",
    "Woes against Pharisees":        "prophetic",
    "Fear of God not men":           "discipleship",
    "Blasphemy Holy Spirit":         "prophetic",
    "Thief in the night":            "apocalyptic",
    "Faithful servant":              "apocalyptic",
    "Not peace but sword":           "discipleship",
    "Reading the signs":             "apocalyptic",
    "Settling with opponent":        "wisdom",
    "Mustard seed and leaven":       "parable",
    "Many come from east west":      "apocalyptic",
    "Lament over Jerusalem":         "prophetic",
    "Parable of Great Banquet":      "parable",
    "Conditions of discipleship":    "discipleship",
    "Salt of the earth":             "wisdom",
    "Lost sheep":                    "parable",
    "Serving two masters":           "wisdom",
    "Law, prophets and divorce":     "wisdom",
    "Scandals, forgiveness, faith":  "discipleship",
    "Day of the Son of Man":         "apocalyptic",
    "Days of Noah":                  "apocalyptic",
    "One taken, one left":           "apocalyptic",
    "Talents / Minas":               "parable",
    "Judging twelve tribes":         "apocalyptic",
}

FORM: Dict[str, str] = {
    "John's preaching": "discourse", "Temptation narrative (full)": "narrative",
    "Beatitudes": "discourse", "Love of enemies": "discourse",
    "Golden rule": "proverbial", "Judging and the speck": "proverbial",
    "Blind guides": "proverbial", "Tree and its fruit": "proverbial",
    "House on the rock": "parable", "Lord's Prayer": "liturgical",
    "Ask, seek, knock": "discourse", "Treasure in heaven": "proverbial",
    "Anxieties about life": "discourse", "Narrow gate": "proverbial",
    "Centurion's servant": "narrative", "Would-be followers": "narrative",
    "John's question from prison": "narrative", "Jesus on John": "discourse",
    "Woes on Galilean cities": "discourse", "Hidden from wise revealed": "proverbial",
    "Blessed eyes": "proverbial", "Mission discourse": "discourse",
    "Harvest plentiful": "proverbial", "Whoever receives you": "proverbial",
    "Sign of Jonah": "discourse", "Return of unclean spirit": "discourse",
    "Lamp on a stand": "proverbial", "Lamp of the body": "proverbial",
    "Woes against Pharisees": "discourse", "Fear of God not men": "discourse",
    "Blasphemy Holy Spirit": "proverbial", "Thief in the night": "proverbial",
    "Faithful servant": "parable", "Not peace but sword": "proverbial",
    "Reading the signs": "proverbial", "Settling with opponent": "proverbial",
    "Mustard seed and leaven": "parable", "Many come from east west": "proverbial",
    "Lament over Jerusalem": "discourse", "Parable of Great Banquet": "parable",
    "Conditions of discipleship": "proverbial", "Salt of the earth": "proverbial",
    "Lost sheep": "parable", "Serving two masters": "proverbial",
    "Law, prophets and divorce": "proverbial", "Scandals, forgiveness, faith": "proverbial",
    "Day of the Son of Man": "discourse", "Days of Noah": "discourse",
    "One taken, one left": "proverbial", "Talents / Minas": "parable",
    "Judging twelve tribes": "proverbial",
}

# Kloppenborg (1987) strata.  Formative stratum: the sapiential speeches
# Q 6:20b-49; 9:57-62 + 10:2-11, 16; 11:2-4, 9-13; 12:2-7, 11-12; 12:22b-31,
# 33-34; 13:24; 14:26-27, 34-35; 17:33.  Q3 = temptation.  U = sayings he
# left without a stratum (16:13, 16-18; 15:4-7; 17:1-6).  Everything else
# belongs to the main redaction (Q2).
STRATUM: Dict[str, str] = {
    "Beatitudes": "Q1", "Love of enemies": "Q1", "Golden rule": "Q1",
    "Judging and the speck": "Q1", "Blind guides": "Q1",
    "Tree and its fruit": "Q1", "House on the rock": "Q1",
    "Would-be followers": "Q1", "Mission discourse": "Q1",
    "Harvest plentiful": "Q1", "Whoever receives you": "Q1",
    "Lord's Prayer": "Q1", "Ask, seek, knock": "Q1",
    "Fear of God not men": "Q1", "Anxieties about life": "Q1",
    "Treasure in heaven": "Q1", "Narrow gate": "Q1",
    "Conditions of discipleship": "Q1", "Salt of the earth": "Q1",
    "Mustard seed and leaven": "Q1",
    "Temptation narrative (full)": "Q3",
    "Serving two masters": "U", "Lost sheep": "U",
    "Law, prophets and divorce": "U", "Scandals, forgiveness, faith": "U",
}


def stratum_of(label: str) -> str:
    return STRATUM.get(label, "Q2")


GOULDER_REDACTION_LABELS: List[str] = [
    "Beatitudes",
    "Love of enemies",
    "Lord's Prayer",
    "Anxieties about life",
    "Narrow gate",
    "Faithful servant",
    "Not peace but sword",
    "Conditions of discipleship",
    "Parable of Great Banquet",
    "Talents / Minas",
]


# Double-tradition units of which all or part has a Markan parallel (the
# Mark-Q overlaps and the Markan doublets of Q sayings).  In these units
# the Matthew-Luke agreement may be mediated by Mark rather than by the
# shared non-Markan source, so the double-tradition statistics are reported
# with and without them.  Value = the Markan parallel; "full" marks units
# whose whole text has a Markan parallel, "partial" units where only some
# verses do.  Reviewed against Aland (1985) and the IQP inventory, Oct 2026.
MARK_Q_OVERLAP: Dict[str, Dict[str, str]] = {
    "John's preaching":             {"mark": "1:7-8",        "extent": "partial"},
    "Temptation narrative (full)":  {"mark": "1:12-13",      "extent": "full"},
    "Judging and the speck":        {"mark": "4:24",         "extent": "partial"},
    "Lamp on a stand":              {"mark": "4:21",         "extent": "full"},
    "Mustard seed and leaven":      {"mark": "4:30-32",      "extent": "partial"},
    "Sign of Jonah":                {"mark": "8:11-12",      "extent": "partial"},
    "Woes against Pharisees":       {"mark": "12:38-39",     "extent": "partial"},
    "Fear of God not men":          {"mark": "4:22",         "extent": "partial"},
    "Blasphemy Holy Spirit":        {"mark": "3:28-29",      "extent": "full"},
    "Mission discourse":            {"mark": "6:7-13",       "extent": "partial"},
    "Whoever receives you":         {"mark": "9:37",         "extent": "full"},
    "Conditions of discipleship":   {"mark": "8:34",         "extent": "partial"},
    "Salt of the earth":            {"mark": "9:50",         "extent": "full"},
    "Law, prophets and divorce":    {"mark": "10:11-12",     "extent": "partial"},
    "Scandals, forgiveness, faith": {"mark": "9:42; 11:23",  "extent": "partial"},
    "Day of the Son of Man":        {"mark": "13:21",        "extent": "partial"},
}


def is_mark_q_overlap(label: str) -> bool:
    return label in MARK_Q_OVERLAP


# Triple-tradition units whose Markan text consists predominantly of
# sayings or discourse of Jesus (parables, their interpretation, the passion
# predictions with their attached sayings, the eschatological discourse,
# and pronouncements that make up most of the unit).  Used only by
# form_control.py, to test whether the double tradition's larger excess is
# an effect of the kind of material.  Annotation made October 2026.
SAYINGS_TT = {
    "Question about fasting", "Parable of the sower", "Purpose of parables",
    "Explanation of sower", "First passion prediction", "Second passion prediction",
    "True greatness", "Third passion prediction", "Parable of tenants",
    "David's son question", "Eschatological discourse",
}


# Text-critical notes on units whose text is uncertain in the manuscript
# tradition (reported in the article's appendix).
TEXT_CRITICAL_NOTES: Dict[str, str] = {
    "Reading the signs": ("Matt 16:2b-3 is absent from א B Γ f13 and others and is "
                          "bracketed in NA28; SBLGNT prints it. The unit is retained "
                          "and the uncertainty is noted."),
}


def check_coverage(labels: List[str]) -> None:
    """Raise if any double-tradition label lacks an annotation."""
    missing = [l for l in labels if l not in THEMATIC_TAGS or l not in FORM]
    if missing:
        raise KeyError(f"Pericopes without metadata: {missing}")
