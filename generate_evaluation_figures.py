"""
generate_evaluation_figures.py
==============================
Publication figure set for the STCM article, computed on the mean-centred
cosine similarity (the primary ranking statistic; see advanced_analysis.py).

Reads:
    outputs/reports/centred_cosine_ranking.csv   (from advanced_analysis.py)
    outputs/reports/pair_values_centred.csv      (from advanced_analysis.py)
    outputs/reports/key_stats.json               (from advanced_analysis.py)
Writes:
    outputs/figures/evaluation/fig1_centred_distribution.png
    outputs/figures/evaluation/fig2_residual_symmetry.png
    outputs/figures/evaluation/fig3_full_ranking.png
    outputs/figures/evaluation/fig4_centred_vs_length.png
    outputs/figures/evaluation/fig5_centred_by_form.png
    outputs/figures/evaluation/fig6_centred_by_stratum.png

Design rules: no in-image titles (the article supplies numbered captions);
every axis label says what is measured and on which texts; the key
reference values are drawn and labelled in plain words inside the plot;
a plain sans-serif face (Calibri where installed, otherwise its
metric-compatible clone Carlito, then Liberation Sans); hue is never the
sole carrier of meaning.
"""
from __future__ import annotations

import csv
import json
import pathlib
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.metadata import FORM, stratum_of

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Carlito", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

# Pastel, colourblind-conscious palette; hatching/linestyle duplicate every
# distinction so that hue is never the sole carrier of meaning.
BLUE = "#A6CEE3"      # observed / matched pairs
CORAL = "#FBB4AE"     # unrelated / mismatched / null reference
GREEN = "#B2DF8A"     # mean of the observed
ORANGE = "#FDBF6F"    # above-mean highlight
GREY = "#BBBBBB"
DARK = "#333333"


def main():
    rep = _REPO / "outputs" / "reports"
    key = json.loads((rep / "key_stats.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(rep / "centred_cosine_ranking.csv", encoding="utf-8")))
    rows.sort(key=lambda r: int(r["rank"]))
    labels = [r["label"] for r in rows]
    cen = np.array([float(r["centred_cos"]) for r in rows])
    resid = np.array([float(r["residual_sim"]) for r in rows])
    n = len(rows)
    pv = list(csv.DictReader(open(rep / "pair_values_centred.csv", encoding="utf-8")))
    dt_mis = np.array([float(r["value"]) for r in pv if r["set"] == "double" and r["kind"] == "mismatched"])
    tt_mat = np.array([float(r["value"]) for r in pv if r["set"] == "triple" and r["kind"] == "matched"])

    fig_dir = _REPO / "outputs" / "figures" / "evaluation"
    fig_dir.mkdir(parents=True, exist_ok=True)
    XLAB = "Mean-centred cosine similarity between the Matthean and Lukan versions"

    # ------------------------------------------------------------------
    # Figure 1: matched pairs vs. mismatched pairs (double tradition)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.4))
    bins = np.linspace(-0.4, 1.0, 36)
    ax.hist(dt_mis, bins=bins, density=True, color=CORAL, edgecolor=DARK, lw=0.4,
            alpha=0.9, hatch="\\\\",
            label=f"Mismatched pairs: a Matthean passage against a Lukan passage it is "
                  f"NOT parallel to (n = {len(dt_mis):,})")
    ax.hist(cen, bins=bins, density=True, color=BLUE, edgecolor=DARK, lw=0.6,
            alpha=0.85,
            label=f"Matched pairs: the Matthean and Lukan versions of the same "
                  f"double-tradition pericope (n = {n})")
    ymax = ax.get_ylim()[1]
    ax.axvline(dt_mis.mean(), color=DARK, lw=1.4, ls=":")
    ax.axvline(cen.mean(), color=DARK, lw=1.4, ls="--")
    ax.annotate(f"mismatched mean\n{dt_mis.mean():.3f}", xy=(dt_mis.mean(), ymax * 1.12),
                xytext=(dt_mis.mean() - 0.03, ymax * 1.12), ha="right", va="top", fontsize=9)
    ax.annotate(f"matched mean\n{cen.mean():.3f}", xy=(cen.mean(), ymax * 1.12),
                xytext=(cen.mean() - 0.03, ymax * 1.12), ha="right", va="top", fontsize=9)
    ax.annotate(f"permutation null for the mean of {n} random pairings:\n"
                f"{key['centred_random_null_mean']:.3f} ± {key['centred_random_null_sd']:.3f} (SD); "
                f"observed matched mean is {((cen.mean()-key['centred_random_null_mean'])/key['centred_random_null_sd']):.0f} SD above it",
                xy=(0.985, 0.80), xycoords="axes fraction", fontsize=8.5, color=DARK,
                va="top", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY))
    ax.set_xlabel(XLAB)
    ax.set_ylabel("Density")
    ax.set_xlim(-0.4, 1.0)
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_centred_distribution.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: residual correlation — top five vs. genre floor
    # ------------------------------------------------------------------
    gf, gsd = key["residual_genre_floor"], key["residual_genre_floor_sd"]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    # all 51 matched values as a light strip for context
    rng = np.random.default_rng(0)
    ax.scatter(resid, 5.4 + rng.uniform(-0.18, 0.18, size=n), s=14, facecolor=GREY,
               edgecolor=DARK, lw=0.3, zorder=2,
               label=f"all {n} matched pericopes (one dot each)")
    ax.barh(range(5), resid[:5], color=BLUE, edgecolor=DARK, height=0.62, zorder=3,
            label="the five pericopes ranked highest on the centred cosine")
    for i, v in enumerate(resid[:5]):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9)
    ax.axvspan(gf - gsd, gf + gsd, color=CORAL, alpha=0.45, hatch="//", zorder=1,
               label=f"mismatched pairs (genre floor): mean {gf:.3f} ± 1 SD")
    ax.axvline(gf, color=DARK, lw=1.2, ls=":")
    ax.axvline(key["residual_matched_mean"], color=DARK, lw=1.4, ls="--",
               label=f"mean of all matched pairs = {key['residual_matched_mean']:.3f}")
    ax.set_yticks(list(range(5)) + [5.4])
    ax.set_yticklabels([f"{i+1}. {labels[i]}" for i in range(5)] + ["all pericopes"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Correlation of the Matthean and Lukan residual vectors\n"
                  "(what each version adds beyond the Markan narrative direction)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_residual_symmetry.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: full ranking
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = [ORANGE if v >= cen.mean() else BLUE for v in cen]
    bars = ax.barh(range(n), cen, color=colors, edgecolor=DARK, lw=0.5, alpha=0.95)
    for bar, v in zip(bars, cen):
        if v >= cen.mean():
            bar.set_hatch("///")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{i+1}. {l}" for i, l in enumerate(labels)], fontsize=7.5)
    ax.axvline(cen.mean(), color=DARK, lw=1.4, ls="--")
    ax.axvline(key["centred_tt_matched"], color=DARK, lw=1.4, ls="-.")
    ax.axvline(key["centred_dt_floor"], color=DARK, lw=1.2, ls=":")
    for x, txt, ha, dx in ((key["centred_dt_floor"], f"unrelated-pair\nfloor {key['centred_dt_floor']:.3f}", "left", 0.01),
                           (key["centred_tt_matched"], f"Markan-copying baseline\n(triple tradition) {key['centred_tt_matched']:.3f}", "right", -0.01),
                           (cen.mean(), f"double-tradition\nmean {cen.mean():.3f}", "left", 0.01)):
        ax.text(x + dx, -1.0, txt, ha=ha, va="bottom", fontsize=8.2)
    ax.text(cen[0] + 0.012, 0, "near-verbatim", va="center", ha="left", fontsize=8, style="italic")
    ax.text(cen[-1] + 0.012, n - 1, "freely reworked", va="center", ha="left", fontsize=8, style="italic")
    ax.set_ylim(n - 0.4, -3.2)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(XLAB)
    ax.legend(handles=[Patch(facecolor=ORANGE, edgecolor=DARK, hatch="///", label="above the double-tradition mean"),
                       Patch(facecolor=BLUE, edgecolor=DARK, label="below the double-tradition mean")],
              loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_full_ranking.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: length confound
    # ------------------------------------------------------------------
    lv = np.array([(float(r["len_matt_chars"]) + float(r["len_luke_chars"])) / 2.0 for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(lv, cen, facecolor=BLUE, edgecolor=DARK, s=46, zorder=3)
    ax.set_xscale("log")
    ax.axhline(cen.mean(), color=DARK, lw=1.2, ls="--")
    ax.text(ax.get_xlim()[1], cen.mean() + 0.012, f"mean {cen.mean():.3f}", ha="right", fontsize=8.5)
    # label the extremes and the longest/shortest pericopes
    marks = {0, n - 1, int(np.argmax(lv)), int(np.argmin(lv)), 1, n - 2}
    for i in sorted(marks):
        off = (-6, -11) if labels[i] == "Harvest plentiful" else (6, 4)
        ha = "right" if labels[i] == "Harvest plentiful" else "left"
        ax.annotate(labels[i], (lv[i], cen[i]), xytext=off, textcoords="offset points", fontsize=8, ha=ha)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["100", "200", "500", "1,000", "2,000"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.text(0.98, 0.04,
            f"Pearson r = {key['length_pearson_r']:.3f} (p = {key['length_pearson_p']:.2f}): "
            f"length does not predict similarity",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("Mean length of the two versions (characters of Greek text, log scale)")
    ax.set_ylabel("Mean-centred cosine similarity\nbetween the Matthean and Lukan versions")
    ax.grid(alpha=0.3, color=DARK, which="major")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_centred_vs_length.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: form-critical categories
    # ------------------------------------------------------------------
    form_order = ["discourse", "narrative", "proverbial", "parable", "liturgical"]
    nice = {"discourse": "extended discourse", "narrative": "narrative", "proverbial": "proverbial saying",
            "parable": "parable", "liturgical": "liturgical"}
    groups = {}
    for lab, v in zip(labels, cen):
        groups.setdefault(FORM[lab], []).append(v)
    present = [f for f in form_order if f in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    _box(ax, [groups[f] for f in present])
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([f"{nice[f]}\n(n = {len(groups[f])})" for f in present])
    ax.axhline(cen.mean(), color=DARK, lw=1.2, ls="--")
    ax.text(len(present) + 0.45, cen.mean() + 0.012, f"overall mean {cen.mean():.3f}", ha="right", fontsize=8.5)
    ax.text(0.02, 0.04, f"Kruskal–Wallis across the four classes with n ≥ 3: p = {key['form_kw_p']:.3f}",
            transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Form-critical class of the pericope")
    ax.set_ylabel("Mean-centred cosine similarity\nbetween the Matthean and Lukan versions")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y", color=DARK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_centred_by_form.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: Kloppenborg strata
    # ------------------------------------------------------------------
    strata = [stratum_of(l) for l in labels]
    order = ["Q1", "Q2", "Q3", "U"]
    names = {"Q1": "Q¹ formative\n(sapiential)", "Q2": "Q² main redaction\n(prophetic judgement)",
             "Q3": "Q³ temptation\nnarrative", "U": "left unassigned\nby Kloppenborg"}
    data = [cen[np.array([s == k for s in strata])] for k in order]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    _box(ax, data)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{names[k]}\n(n = {len(d)})" for k, d in zip(order, data)], fontsize=9)
    ax.axhline(cen.mean(), color=DARK, lw=1.2, ls="--")
    ax.text(4.45, cen.mean() + 0.012, f"overall mean {cen.mean():.3f}", ha="right", fontsize=8.5)
    ax.text(0.02, 0.04, f"Q¹ vs Q²: Mann–Whitney p = {key['strata_mw_p']:.3f}",
            transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Compositional stratum of Q (Kloppenborg 1987), pericope-level approximation")
    ax.set_ylabel("Mean-centred cosine similarity\nbetween the Matthean and Lukan versions")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y", color=DARK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_centred_by_stratum.png")
    plt.close(fig)

    print("Figures written to", fig_dir)

    # Mirror the figures into the MkDocs source tree (docs_dir='site').
    site_img = _REPO / "site" / "img"
    site_img.mkdir(parents=True, exist_ok=True)
    copied = 0
    for png in sorted(fig_dir.glob("fig*.png")):
        shutil.copyfile(png, site_img / png.name)
        copied += 1
    print(f"Mirrored {copied} figures to", site_img)


def _box(ax, data):
    """Box plot with the individual pericopes overlaid as dots."""
    ax.boxplot(data, widths=0.5, showmeans=False,
               boxprops=dict(color=DARK), whiskerprops=dict(color=DARK),
               capprops=dict(color=DARK), medianprops=dict(color=DARK, lw=1.8),
               flierprops=dict(marker=""))
    rng = np.random.default_rng(1)
    for i, vals in enumerate(data, 1):
        xs = rng.normal(i, 0.055, size=len(vals))
        ax.scatter(xs, vals, s=26, facecolor=BLUE, edgecolor=DARK, lw=0.5, alpha=0.9, zorder=3)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=BLUE, markeredgecolor=DARK,
                              label="one pericope"),
                       Line2D([], [], color=DARK, lw=1.8, label="median; box = middle half of the values")],
              loc="upper right", frameon=False, fontsize=8.5)


if __name__ == "__main__":
    main()
