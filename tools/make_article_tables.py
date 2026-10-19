"""
make_article_tables.py
======================
Assemble the markdown tables for Appendices A, B, and C (Tables A1, B1,
and C1-C3) directly from the CSV and JSON files in ``outputs/reports/``.

Usage
-----
    python tools/make_article_tables.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from stcm.metadata import SAYINGS_TT  # noqa: E402

REPORTS = _REPO / "outputs" / "reports"

_STRATUM_MAP = {"Q1": "Q¹", "Q2": "Q²", "Q3": "Q³", "U": "U"}

_ENCODERS = (
    ("primary", "Koine-DAPT, last layer (primary)"),
    ("last4", "Koine-DAPT, mean of last four layers"),
    ("agb", "Ancient-Greek-BERT, last layer"),
    ("greekbert", "Greek BERT (Modern Greek; no biblical pre-training), last layer"),
)


def make_table_a1(reports_dir: pathlib.Path = REPORTS) -> str:
    rows = list(csv.DictReader((reports_dir / "centred_cosine_ranking.csv").open(encoding="utf-8")))
    lines = [
        "| Rank | Pericope | Matt ref. | Luke ref. | Stratum | Form | Theme | Raw cos | Centred cos | Jaccard |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        label = r["label"]
        if r.get("mark_parallel"):
            label += " †"
        if r.get("text_critical_note"):
            label += " ‡"
        stratum = _STRATUM_MAP.get(r["stratum"], r["stratum"])
        lines.append(
            f"| {r['rank']} | {label} | {r['matt_ref']} | {r['luke_ref']} | {stratum} "
            f"| {r['form']} | {r['theme']} | {r['matt_luke_cos']} | {r['centred_cos']} | {r['word_overlap']} |"
        )
    return "\n".join(lines)


def make_table_b1(reports_dir: pathlib.Path = REPORTS) -> str:
    rows = list(csv.DictReader((reports_dir / "triple_tradition_table.csv").open(encoding="utf-8")))
    lines = [
        "| Pericope | Matt | Mark | Luke | Matt–Luke | Matt–Mark | Luke–Mark | Residual corr. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        label = r["label"] + (" S" if r["label"] in SAYINGS_TT else "")
        lines.append(
            f"| {label} | {r['matt_ref']} | {r['mark_ref']} | {r['luke_ref']} "
            f"| {r['centred_matt_luke']} | {r['centred_matt_mark']} | {r['centred_luke_mark']} | {r['residual_corr']} |"
        )
    return "\n".join(lines)


def make_table_c1(reports_dir: pathlib.Path = REPORTS) -> str:
    rows = list(csv.DictReader((reports_dir / "composite_sensitivity.csv").open(encoding="utf-8")))
    lines = [
        "| Pericope | Set | Composite side | Range-wise (used) | Single sequence | Difference |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['set']} | {r['composite_side']} "
            f"| {r['centred_rangewise']} | {r['centred_single']} | {r['difference']} |"
        )
    return "\n".join(lines)


def make_table_c2(reports_dir: pathlib.Path = REPORTS) -> str:
    rob = json.loads((reports_dir / "key_stats_robustness.json").read_text(encoding="utf-8"))
    lines = [
        "| Encoder | Space | DT excess | DT − TT [95% CI] | DT − Luke–Mark [95% CI] | Residual after Jaccard: matched − mismatched; DT − TT [95% CI] | ρ |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, enc_label in _ENCODERS:
        d = rob[key]
        rho = f"{d['spearman_vs_primary']:.2f}"
        for sp_key, sp_label in (("cen", "centred"), ("raw", "raw")):
            dt_exc = f"{d[f'{sp_key}_dt']['excess']:.3f}"
            c_tt = d[f"{sp_key}_dt_minus_tt"]
            c_lk = d[f"{sp_key}_dt_minus_lk"]
            dt_tt_str = f"{c_tt['diff']:+.3f} [{c_tt['ci_lo']:.3f}, {c_tt['ci_hi']:.3f}]"
            dt_lk_str = f"{c_lk['diff']:+.3f} [{c_lk['ci_lo']:.3f}, {c_lk['ci_hi']:.3f}]"
            if sp_key == "cen":
                f_exc = d["resid_dt"]["excess"]
                r_tt = d["resid_dt_minus_tt"]
                resid_str = f"{f_exc:+.3f}; {r_tt['diff']:+.3f} [{r_tt['ci_lo']:.3f}, {r_tt['ci_hi']:.3f}]"
            else:
                resid_str = "—"
            lines.append(
                f"| {enc_label} | {sp_label} | {dt_exc} | {dt_tt_str} | {dt_lk_str} | {resid_str} | {rho} |"
            )
    return "\n".join(lines)


def make_table_c3(reports_dir: pathlib.Path = REPORTS) -> str:
    ks = json.loads((reports_dir / "key_stats.json").read_text(encoding="utf-8"))
    lines = [
        "| Penalty α | Δ*R*² | Sign-flip *p* |",
        "|---|---|---|",
    ]
    for g in ks["alpha_grid"]:
        lines.append(f"| {g['alpha']:g} | {g['delta_r2']:+.4f} | {g['p']:.3f} |")
    lines.extend([
        "",
        "| Control (α = 1) | Δ*R*² | Sign-flip *p* |",
        "|---|---|---|",
        f"| Double tradition, all pairs (*n* = {ks['n_double']}) | {ks['delta_r2']:+.4f} | {ks['signflip_p']:.3f} |",
        f"| Double tradition, length-balanced pairs (*n* = {ks['n_length_balanced']}) | {ks['delta_r2_length_balanced']:+.4f} | {ks['signflip_p_length_balanced']:.3f} |",
        f"| Triple tradition, Mark → Matthew (*n* = {ks['n_triple']}) | {ks['delta_r2_mark_matt']:+.4f} | {ks['signflip_p_mark_matt']:.3f} |",
        f"| Triple tradition, Mark → Luke (*n* = {ks['n_triple']}) | {ks['delta_r2_mark_luke']:+.4f} | {ks['signflip_p_mark_luke']:.3f} |",
        f"| Triple tradition, Matthew vs Luke (both copying Mark) | {ks['delta_r2_matt_luke_tt']:+.4f} | {ks['signflip_p_matt_luke_tt']:.3f} |",
    ])
    return "\n".join(lines)


def make_all_tables(reports_dir: pathlib.Path = REPORTS) -> dict[str, str]:
    return {
        "A1": make_table_a1(reports_dir),
        "B1": make_table_b1(reports_dir),
        "C1": make_table_c1(reports_dir),
        "C2": make_table_c2(reports_dir),
        "C3": make_table_c3(reports_dir),
    }


def main() -> None:
    tables = make_all_tables()
    out_parts = []
    for name, content in tables.items():
        out_parts.append(f"### Table {name}\n\n{content}")
    text = "\n\n".join(out_parts) + "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))


if __name__ == "__main__":
    main()
