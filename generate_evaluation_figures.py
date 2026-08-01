"""
generate_evaluation_figures.py
==============================
Publication figure set for the STCM article, computed on the mean-centred
cosine similarity (the primary ranking statistic; see advanced_analysis.py).

Reads:
    outputs/reports/centred_cosine_ranking.csv  (from advanced_analysis.py)
Writes:
    outputs/figures/evaluation/fig1_centred_distribution.png
    outputs/figures/evaluation/fig2_residual_symmetry.png
    outputs/figures/evaluation/fig3_full_ranking.png
    outputs/figures/evaluation/fig4_centred_vs_length.png
    outputs/figures/evaluation/fig5_centred_by_form.png
    outputs/figures/evaluation/fig6_centred_by_stratum.png

Figure 1 overlays the permutation-null distribution of the mean using its
summary statistics from the published run (centred random null: mean
0.0737, SD 0.0278); override with --null-mean / --null-std.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.data_loader import DOUBLE_TRADITION
from advanced_analysis import STRATUM, FORM

SIG_B_MEAN = 0.3693   # triple-tradition residual baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-mean", type=float, default=0.0737)
    ap.add_argument("--null-std", type=float, default=0.0278)
    args = ap.parse_args()

    fig_dir = _REPO / "outputs" / "figures" / "evaluation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(
        _REPO / "outputs" / "reports" / "centred_cosine_ranking.csv",
        encoding="utf-8")))
    rows.sort(key=lambda r: int(r["rank"]))
    labels = [r["label"] for r in rows]
    cen = np.array([float(r["centred_cos"]) for r in rows])
    resid = np.array([float(r["residual_sim"]) for r in rows])

    # Pastel, colourblind-conscious palette. Hue is never the sole carrier of
    # meaning — linestyle/hatching duplicate every distinction (per Cambridge's
    # accessible-figures guidance), and edges/text stay dark for print contrast.
    PASTEL_BLUE = "#A6CEE3"      # primary observed data (bars, scatter)
    PASTEL_CORAL = "#FBB4AE"     # comparison / null / baseline reference
    PASTEL_GREEN = "#B2DF8A"     # mean / central-tendency marker
    PASTEL_ORANGE = "#FDBF6F"    # above-mean highlight (fig. 3)
    DARK = "#333333"             # edges, axes, box outlines
    BLACK = "#000000"

    # Figure 1: distribution vs permutation null of the mean
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cen, bins=14, density=True, color=PASTEL_BLUE, edgecolor=DARK,
            alpha=0.9, label="Observed centred cosine (matched pairs)")
    xs = np.linspace(args.null_mean - 5 * args.null_std, cen.max() + 0.05, 400)
    ax.plot(xs, stats.norm.pdf(xs, args.null_mean, args.null_std),
            color=PASTEL_CORAL, lw=2.5, linestyle="-",
            label=f"Permutation null of the mean "
                  f"(μ = {args.null_mean:.4f}, σ = {args.null_std:.4f})")
    ax.axvline(cen.mean(), color=PASTEL_GREEN, lw=2.5, linestyle="--",
               label=f"Observed mean = {cen.mean():.4f}")
    ax.set_xlabel("Mean-centred cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Centred Matt–Luke similarity vs. random permutation null")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_centred_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 2: residual symmetry of the top five (by centred cosine)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(5), resid[:5], color=PASTEL_BLUE, edgecolor=DARK, alpha=0.9)
    ax.axvline(SIG_B_MEAN, color=PASTEL_CORAL, lw=2.5, linestyle="--",
               label=f"Triple-tradition residual baseline ({SIG_B_MEAN:.3f})")
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels[:5], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Residual vector similarity")
    ax.set_title("Residual symmetry of the top five double-tradition pericopes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_residual_symmetry.png", dpi=150)
    plt.close(fig)

    # Figure 3: full ranking (above-mean bars pastel orange with a hatch,
    # below-mean bars pastel blue — colour and pattern both carry the split)
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [PASTEL_ORANGE if v >= cen.mean() else PASTEL_BLUE for v in cen]
    hatches = ["///" if v >= cen.mean() else None for v in cen]
    bars = ax.barh(range(len(cen)), cen, color=colors, edgecolor=DARK, alpha=0.95)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax.set_yticks(range(len(cen)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(cen.mean(), color=PASTEL_GREEN, lw=2, linestyle="--",
               label=f"Mean = {cen.mean():.4f}")
    ax.set_xlabel("Mean-centred cosine similarity")
    ax.set_title("All 36 double-tradition pericopes, ranked by centred cosine")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_full_ranking.png", dpi=150)
    plt.close(fig)

    # Figure 4: length confound
    lengths_map = {}
    for lab, m_ref, l_ref in DOUBLE_TRADITION:
        if m_ref and l_ref:
            lengths_map[lab] = ((m_ref[2] - m_ref[1] + 1) + (l_ref[2] - l_ref[1] + 1)) / 2.0
    lv = np.array([lengths_map[l] for l in labels])
    r_p, p_p = stats.pearsonr(lv, cen)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lv, cen, facecolor=PASTEL_BLUE, edgecolor=DARK, s=60, zorder=3)
    ax.set_xlabel("Mean pericope length (verses)")
    ax.set_ylabel("Mean-centred cosine")
    ax.set_title(f"Centred similarity vs. passage length "
                 f"(Pearson r = {r_p:.3f}, p = {p_p:.3f})")
    ax.grid(alpha=0.3, color=DARK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_centred_vs_length.png", dpi=150)
    plt.close(fig)

    # Figure 5: form-critical categories
    form_order = ["proverbial", "discourse", "parable", "narrative", "liturgical"]
    groups = {}
    for lab, v in zip(labels, cen):
        groups.setdefault(FORM.get(lab, "proverbial"), []).append(v)
    present = [f for f in form_order if f in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([groups[f] for f in present],
                     boxprops=dict(color=DARK),
                     whiskerprops=dict(color=DARK),
                     capprops=dict(color=DARK),
                     medianprops=dict(color=PASTEL_CORAL, lw=2))
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present)
    for i, f in enumerate(present, 1):
        xs = np.random.default_rng(0).normal(i, 0.05, size=len(groups[f]))
        ax.scatter(xs, groups[f], alpha=0.85, facecolor=PASTEL_BLUE,
                   edgecolor=DARK, zorder=3)
    ax.set_ylabel("Mean-centred cosine")
    ax.set_title("Centred similarity by form-critical category")
    ax.grid(alpha=0.3, axis="y", color=DARK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_centred_by_form.png", dpi=150)
    plt.close(fig)

    # Figure 6: Kloppenborg strata
    strata = [STRATUM.get(l, "Q2") for l in labels]
    data = [cen[np.array([s == k for s in strata])] for k in ["Q1", "Q2", "Q3", "U"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data,
               boxprops=dict(color=DARK),
               whiskerprops=dict(color=DARK),
               capprops=dict(color=DARK),
               medianprops=dict(color=PASTEL_CORAL, lw=2))
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "U"])
    for i, vals in enumerate(data, 1):
        xs = np.random.default_rng(1).normal(i, 0.05, size=len(vals))
        ax.scatter(xs, vals, alpha=0.85, facecolor=PASTEL_BLUE,
                   edgecolor=DARK, zorder=3)
    ax.set_ylabel("Mean-centred cosine")
    ax.set_title("Centred similarity by Kloppenborg stratum")
    ax.grid(alpha=0.3, axis="y", color=DARK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_centred_by_stratum.png", dpi=150)
    plt.close(fig)

    print("Figures written to", fig_dir)


if __name__ == "__main__":
    main()
