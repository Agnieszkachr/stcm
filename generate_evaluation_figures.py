"""
generate_evaluation_figures.py
==============================
Publication figure set for the STCM article, computed on the
mean-centred cosine similarity (the primary statistic; see
advanced_analysis.py) and the lexical baseline (baselines.py).

Reads:
    outputs/reports/centred_cosine_ranking.csv   (advanced_analysis.py)
    outputs/reports/pair_values_centred.csv      (advanced_analysis.py)
    outputs/reports/per_pericope_excess.csv      (advanced_analysis.py)
    outputs/reports/key_stats.json               (advanced_analysis.py)
    outputs/reports/baseline_disagreements.csv   (baselines.py)
    outputs/reports/key_stats_baselines.json     (baselines.py)
Writes:
    outputs/figures/evaluation/fig1_centred_distribution.png
    outputs/figures/evaluation/fig2_excess_by_set.png
    outputs/figures/evaluation/fig4_full_ranking.png
    outputs/figures/evaluation/fig3_centred_vs_jaccard.png
    outputs/figures/evaluation/fig5_confounds.png

Design rules: no in-image titles (the article supplies numbered captions);
every axis label says what is measured and on which texts; the key
reference values are drawn and labelled in plain words inside the plot;
a plain sans-serif face; hue is never the sole carrier of meaning.
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

from stcm.metadata import FORM, MARK_Q_OVERLAP, stratum_of

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

BLUE = "#A6CEE3"      # observed / matched pairs
CORAL = "#FBB4AE"     # unrelated / mismatched / null reference
GREEN = "#B2DF8A"     # one-step copying (Luke-Mark, Matt-Mark)
ORANGE = "#FDBF6F"    # above-mean highlight
GREY = "#BBBBBB"
DARK = "#333333"


def _box(ax, data, colors=None, legend=True):
    """Box plot with the individual pericopes overlaid as dots."""
    ax.boxplot(data, widths=0.5, showmeans=False,
               boxprops=dict(color=DARK), whiskerprops=dict(color=DARK),
               capprops=dict(color=DARK), medianprops=dict(color=DARK, lw=1.8),
               flierprops=dict(marker=""))
    rng = np.random.default_rng(1)
    for i, vals in enumerate(data, 1):
        xs = rng.normal(i, 0.055, size=len(vals))
        c = colors[i - 1] if colors else BLUE
        ax.scatter(xs, vals, s=22, facecolor=c, edgecolor=DARK, lw=0.5, alpha=0.9, zorder=3)
    if legend:
        ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=BLUE, markeredgecolor=DARK,
                                  label="one pericope"),
                           Line2D([], [], color=DARK, lw=1.8, label="median; box = middle half of the values")],
                  loc="upper right", frameon=False, fontsize=8.5)


def main():
    rep = _REPO / "outputs" / "reports"
    key = json.loads((rep / "key_stats.json").read_text(encoding="utf-8"))
    kb = json.loads((rep / "key_stats_baselines.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(rep / "centred_cosine_ranking.csv", encoding="utf-8")))
    rows.sort(key=lambda r: int(r["rank"]))
    labels = [r["label"] for r in rows]
    cen = np.array([float(r["centred_cos"]) for r in rows])
    jac = np.array([float(r["word_overlap"]) for r in rows])
    n = len(rows)
    pv = list(csv.DictReader(open(rep / "pair_values_centred.csv", encoding="utf-8")))
    dt_mis = np.array([float(r["value"]) for r in pv if r["set"] == "double" and r["kind"] == "mismatched"])
    pe = list(csv.DictReader(open(rep / "per_pericope_excess.csv", encoding="utf-8")))

    fig_dir = _REPO / "outputs" / "figures" / "evaluation"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in ("fig2_residual_symmetry.png", "fig3_full_ranking.png", "fig4_centred_vs_length.png",
                "fig5_centred_by_form.png", "fig6_centred_by_stratum.png"):
        (fig_dir / old).unlink(missing_ok=True)
        (_REPO / "site" / "img" / old).unlink(missing_ok=True)
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
                f"no permutation reaches the observed mean (p < 0.001)",
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
    # Figure 2: per-pericope excess over floor, by set (the central contrast)
    # ------------------------------------------------------------------
    sets = [("dt", "Matthew–Luke,\ndouble tradition\n(disputed source)", BLUE),
            ("tt", "Matthew–Luke,\ntriple tradition\n(both copy Mark)", CORAL),
            ("lk", "Luke–Mark,\ntriple tradition\n(Luke copies)", GREEN),
            ("mk", "Matthew–Mark,\ntriple tradition\n(Matthew copies)", GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, (col, space, ylab, title) in zip(axes, (
            ("centred_excess_mean", "cen", "Excess over the mismatched-pair floor", "(a) mean-centred cosine"),
            ("raw_excess_mean", "raw", "Excess over the mismatched-pair floor", "(b) uncorrected cosine"))):
        data = [np.array([float(r[col]) for r in pe if r["set"] == s]) for s, _, _ in sets]
        _box(ax, data, colors=[c for _, _, c in sets], legend=False)
        for i, (s, _, _) in enumerate(sets, 1):
            lo, hi = key[f"{space}_{s}_excess_ci"]
            m = key[f"{space}_{s}_excess"]
            ax.errorbar(i + 0.36, m, yerr=[[m - lo], [hi - m]], fmt="D", color=DARK, ms=4, capsize=3, lw=1.2, zorder=4)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([t_ for _, t_, _ in sets], fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_xlabel(title)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.grid(alpha=0.3, axis="y", color=DARK)
        r_tt = key[f"{space}_dt_minus_tt_mean"]
        r_lk = key[f"{space}_dt_minus_lk_mean"]
        ax.text(0.5, -0.24,
                f"double minus triple (Matthew–Luke): {r_tt['diff']:+.3f} [{r_tt['ci_lo']:+.3f}, {r_tt['ci_hi']:+.3f}]\n"
                f"double minus Luke–Mark: {r_lk['diff']:+.3f} [{r_lk['ci_lo']:+.3f}, {r_lk['ci_hi']:+.3f}]",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="center")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=GREY, markeredgecolor=DARK, label="one pericope"),
                        Line2D([], [], color=DARK, lw=1.8, label="median; box = middle half of the values"),
                        Line2D([], [], marker="D", ls="", color=DARK, ms=4, label="mean with 95% bootstrap CI")],
               loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / "fig2_excess_by_set.png")
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
    ax.set_yticklabels([f"{i+1}. {l}" + (" †" if l in MARK_Q_OVERLAP else "") for i, l in enumerate(labels)], fontsize=7.5)
    ax.axvline(cen.mean(), color=DARK, lw=1.4, ls="--")
    ax.axvline(key["cen_dt_floor"], color=DARK, lw=1.2, ls=":")
    for x, txt, ha, dx in ((key["cen_dt_floor"], f"unrelated-pair\nfloor {key['cen_dt_floor']:.3f}", "left", 0.01),
                           (cen.mean(), f"double-tradition\nmean {cen.mean():.3f}", "left", 0.01)):
        ax.text(x + dx, -1.0, txt, ha=ha, va="bottom", fontsize=8.2)
    ax.text(cen[0] + 0.012, 0, "near-verbatim", va="center", ha="left", fontsize=8, style="italic")
    ax.text(cen[-1] + 0.012, n - 1, "freely reworked", va="center", ha="left", fontsize=8, style="italic")
    ax.set_ylim(n - 0.4, -3.2)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(XLAB)
    ax.legend(handles=[Patch(facecolor=ORANGE, edgecolor=DARK, hatch="///", label="above the double-tradition mean"),
                       Patch(facecolor=BLUE, edgecolor=DARK, label="below the double-tradition mean"),
                       Line2D([], [], ls="", label="† all or part of the unit has a Markan parallel")],
              loc="lower right", frameon=True, framealpha=1.0, facecolor="white", edgecolor="white")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_full_ranking.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: centred cosine against word overlap (what the embedding adds)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.scatter(jac, cen, facecolor=BLUE, edgecolor=DARK, s=46, zorder=3)
    b1, b0 = np.polyfit(jac, cen, 1)
    xs = np.linspace(jac.min(), jac.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color=DARK, lw=1.2, ls="--", zorder=2)
    resid = cen - (b0 + b1 * jac)
    marks = set(np.argsort(resid)[:6]) | set(np.argsort(resid)[-6:]) | {0, n - 1}
    # Labels placed in data coordinates with leader lines, so that every label points at its dot.
    place = {"Lord's Prayer": (0.40, 0.93), "Woes against Pharisees": (0.42, 0.98),
             "Golden rule": (0.10, 0.94), "Conditions of discipleship": (0.02, 0.80),
             "One taken, one left": (0.04, 0.75), "Mission discourse": (0.12, 0.88),
             "Jesus on John": (0.62, 0.60), "John's preaching": (0.74, 0.70),
             "Scandals, forgiveness, faith": (0.34, 0.36), "Judging twelve tribes": (0.40, 0.42),
             "Parable of Great Banquet": (0.24, 0.24), "Narrow gate": (0.22, 0.12),
             "Serving two masters": (0.80, 0.90)}
    for i in sorted(marks):
        if labels[i] not in place:
            continue
        ax.annotate(labels[i], (jac[i], cen[i]), xytext=place[labels[i]], textcoords="data", fontsize=7.8,
                    arrowprops=dict(arrowstyle="-", color=DARK, lw=0.6, shrinkA=1, shrinkB=4))
    rho = kb["Jaccard_spearman_vs_centred_dt"]
    ax.text(0.98, 0.04, f"Spearman ρ = {rho:.2f}; Pearson r = {kb['Jaccard_pearson_vs_centred_dt']:.2f}",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("Word overlap of the two versions (multiset Jaccard coefficient on accent-stripped word forms)")
    ax.set_ylabel("Mean-centred cosine similarity\nbetween the Matthean and Lukan versions")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.02)
    ax.grid(alpha=0.3, color=DARK)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_centred_vs_jaccard.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: confounds — length, form, stratum (three panels)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.15, 1.1, 1.0]))
    # (a) length
    ax = axes[0]
    lv = np.array([(float(r["len_matt_chars"]) + float(r["len_luke_chars"])) / 2.0 for r in rows])
    ax.scatter(lv, cen, facecolor=BLUE, edgecolor=DARK, s=36, zorder=3)
    ax.set_xscale("log")
    ax.axhline(cen.mean(), color=DARK, lw=1.2, ls="--")
    for i in sorted({0, n - 1, int(np.argmax(lv)), int(np.argmin(lv))}):
        ax.annotate(labels[i], (lv[i], cen[i]), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["100", "200", "500", "1,000", "2,000"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.text(0.98, 0.04, f"Pearson r = {key['length_pearson_r']:.3f} (p = {key['length_pearson_p']:.2f})",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right")
    ax.set_xlabel("(a) Mean length of the two versions\n(characters of Greek text, log scale)")
    ax.set_ylabel("Mean-centred cosine similarity\nbetween the Matthean and Lukan versions")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, color=DARK, which="major")
    # (b) form
    ax = axes[1]
    form_order = ["discourse", "narrative", "proverbial", "parable", "liturgical"]
    nice = {"discourse": "extended\ndiscourse", "narrative": "narrative", "proverbial": "proverbial\nsaying",
            "parable": "parable", "liturgical": "liturgical"}
    groups = {}
    for lab, v in zip(labels, cen):
        groups.setdefault(FORM[lab], []).append(v)
    present = [f for f in form_order if f in groups]
    _box(ax, [groups[f] for f in present], legend=False)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([f"{nice[f]}\n(n = {len(groups[f])})" for f in present], fontsize=8.5)
    ax.axhline(cen.mean(), color=DARK, lw=1.2, ls="--")
    ax.text(0.02, 0.04, f"Kruskal–Wallis, classes with n ≥ 3:\np = {key['form_kw_p']:.3f}",
            transform=ax.transAxes, fontsize=8.5)
    ax.set_xlabel("(b) Form-critical class of the pericope")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y", color=DARK)
    # (c) stratum
    ax = axes[2]
    strata = [stratum_of(l) for l in labels]
    order = ["Q1", "Q2", "Q3", "U"]
    names = {"Q1": "Q¹\nformative", "Q2": "Q²\nmain redaction", "Q3": "Q³\ntemptation", "U": "unassigned"}
    data = [cen[np.array([s == k for s in strata])] for k in order]
    _box(ax, data, legend=False)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{names[k]}\n(n = {len(d)})" for k, d in zip(order, data)], fontsize=8.5)
    ax.axhline(cen.mean(), color=DARK, lw=1.2, ls="--")
    ax.text(0.02, 0.04, f"Q¹ vs Q²: Mann–Whitney\np = {key['strata_mw_p']:.3f}", transform=ax.transAxes, fontsize=8.5)
    ax.set_xlabel("(c) Compositional stratum of Q\n(Kloppenborg 1987, pericope level)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y", color=DARK)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=BLUE, markeredgecolor=DARK, label="one pericope"),
                        Line2D([], [], color=DARK, lw=1.8, label="median; box = middle half of the values"),
                        Line2D([], [], color=DARK, lw=1.2, ls="--", label=f"overall mean {cen.mean():.3f}")],
               loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "fig5_confounds.png")
    plt.close(fig)

    print("Figures written to", fig_dir)
    site_img = _REPO / "site" / "img"
    site_img.mkdir(parents=True, exist_ok=True)
    copied = 0
    for png in sorted(fig_dir.glob("fig*.png")):
        shutil.copyfile(png, site_img / png.name)
        copied += 1
    print(f"Mirrored {copied} figures to", site_img)


if __name__ == "__main__":
    main()
