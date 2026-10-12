"""
advanced_analysis.py
====================
All embedding-based statistics reported in the article, computed from the
cached pericope vectors (``populate_cache.py`` fills the cache; once it is
full no model weights are needed except for the sentence-level bootstrap,
which re-embeds resampled text).

Analyses
--------
1. Calibration (Table 1): Matt-Luke cosine in the triple tradition
   (two-step derivation from Mark), Matt-Mark and Luke-Mark cosine
   (one-step copying), and the pericope-specific residual correlation.
2. Floors and excess (Table 2): matched mean, mismatched-pair floor,
   nearest-non-parallel floor and the two excesses, for the double
   tradition (Matt-Luke), the triple tradition (Matt-Luke, Matt-Mark,
   Luke-Mark), raw and mean-centred; per-pericope excess values with
   bootstrap confidence intervals, Mann-Whitney and label-permutation
   tests for every between-set contrast; size-matched nearest-neighbour
   excess; internal homogeneity; centred norms; sensitivity to the
   centring vector (Table 4).
3. Permutation tests on the raw and centred cosine against four nulls
   (random, thematic, length-matched, length-and-theme-matched).
4. Pericope-specific residual correlation in the triple tradition (the
   geometric form of the minor agreements): matched mean against the
   mismatched floor, random and length-matched nulls.
5. Mark-Q overlaps: double-tradition statistics with and without the
   units that have a Markan parallel.
6. Composite units: centred cosine under range-wise and single-sequence
   encoding (sensitivity table).
7. Directionality: ridge regression with intercept in both directions,
   exact leave-one-out R^2, penalty chosen by leave-one-out error,
   effective degrees of freedom, jackknife SE, sign-flip permutation test,
   alpha sweep, length-balanced control and the known-direction control.
8. Ranking and per-pericope statistics: confounds (length, form, stratum,
   Goulder set, word overlap) and the sentence-level bootstrap (pericopes
   with fewer than two sentences in either version are excluded).

Outputs
-------
    outputs/reports/advanced_analysis.md
    outputs/reports/centred_cosine_ranking.csv
    outputs/reports/calibration_table.csv
    outputs/reports/floor_table.csv
    outputs/reports/per_pericope_excess.csv
    outputs/reports/composite_sensitivity.csv
    outputs/reports/pair_values_centred.csv
    outputs/reports/key_stats.json
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
from scipy import stats

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.analysis_core import (SEED, N_PERM, N_BOOT, cos, cos_matrix, floors,
                                per_pericope_excess, compare_excess, bootstrap_ci,
                                size_matched_nn_excess, permutation_tests, fmt_null_block,
                                perm_pvalue, loo_r2, loo_sse, delta_r2, signflip_p,
                                effective_df, split_sentences)
from stcm.data_loader import SBLGNTLoader, DOUBLE_TRADITION, TRIPLE_TRADITION, format_ref
from stcm.embeddings import EmbeddingPipeline
from stcm.metadata import (THEMATIC_TAGS, FORM, GOULDER_REDACTION_LABELS, MARK_Q_OVERLAP,
                           TEXT_CRITICAL_NOTES, stratum_of, check_coverage)
from stcm.utils import residual_vector, word_tokens

N_SENT_BOOT = 200
ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
OUT = _REPO / "outputs" / "reports"


def word_overlap(text_a: str, text_b: str) -> float:
    """Multiset (token-level) Jaccard coefficient on cleaned word tokens."""
    from collections import Counter
    ca, cb = Counter(word_tokens(text_a)), Counter(word_tokens(text_b))
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    corpus = SBLGNTLoader().load()
    pipe = EmbeddingPipeline()

    double = [p for p in corpus.double_tradition if p.matthew and p.luke]
    labels = [p.label for p in double]
    check_coverage(labels)
    M = np.array([pipe.embed_parts(p.matthew_parts) for p in double], dtype=np.float64)
    L = np.array([pipe.embed_parts(p.luke_parts) for p in double], dtype=np.float64)
    len_m = np.array([len(p.matthew) for p in double], dtype=float)
    len_l = np.array([len(p.luke) for p in double], dtype=float)
    tags = [THEMATIC_TAGS[l] for l in labels]

    triple = [p for p in corpus.triple_tradition if p.matthew and p.mark and p.luke]
    tlabels = [p.label for p in triple]
    TM = np.array([pipe.embed_parts(p.matthew_parts) for p in triple], dtype=np.float64)
    TK = np.array([pipe.embed_parts(p.mark_parts) for p in triple], dtype=np.float64)
    TL = np.array([pipe.embed_parts(p.luke_parts) for p in triple], dtype=np.float64)
    tlen_m = np.array([len(p.matthew) for p in triple], dtype=float)
    tlen_k = np.array([len(p.mark) for p in triple], dtype=float)
    tlen_l = np.array([len(p.luke) for p in triple], dtype=float)

    n, nt = len(M), len(TM)
    centroid = TK.mean(axis=0)
    all_vecs = np.vstack([M, L, TM, TK, TL])
    mu = all_vecs.mean(axis=0)

    key = {"n_double": n, "n_triple": nt, "embedder": pipe.embedder_id}
    out = ["# STCM Advanced Analysis (version 3)", "",
           f"(n_double = {n}, n_triple = {nt}, corpus vectors = {len(all_vecs)}, "
           f"permutations = {N_PERM}, bootstrap = {N_BOOT}, seed = {SEED}; "
           f"encoder = {pipe.embedder_id})", "",
           "Lengths are in characters of the normalised Greek text. "
           f"Double tradition: Matthew median {np.median(len_m):.0f}, Luke median {np.median(len_l):.0f}; "
           f"triple tradition: Matthew median {np.median(tlen_m):.0f}, Mark median {np.median(tlen_k):.0f}, "
           f"Luke median {np.median(tlen_l):.0f}.", ""]

    # ------------------------------------------------------------------
    # Pair matrices
    # ------------------------------------------------------------------
    S = {}
    for tag, c in (("raw", None), ("cen", mu)):
        S[(tag, "dt")] = cos_matrix(M, L, c)          # Matt-Luke, double tradition
        S[(tag, "tt")] = cos_matrix(TM, TL, c)        # Matt-Luke, triple tradition
        S[(tag, "mk")] = cos_matrix(TM, TK, c)        # Matt-Mark
        S[(tag, "lk")] = cos_matrix(TL, TK, c)        # Luke-Mark
    SETS = [("dt", "Double tradition, Matt-Luke"), ("tt", "Triple tradition, Matt-Luke"),
            ("mk", "Triple tradition, Matt-Mark"), ("lk", "Triple tradition, Luke-Mark")]

    # ------------------------------------------------------------------
    # 1. Calibration signatures (Table 1)
    # ------------------------------------------------------------------
    sigb_v = np.array([cos(residual_vector(TM[i], TK[i]), residual_vector(TL[i], TK[i]))
                       for i in range(nt)])
    out.append("## 1. Calibration signatures (triple tradition)\n")
    out.append("| Signature | Mean | SD | 95% bootstrap CI |")
    out.append("|---|---|---|---|")
    rows = [("A: Matt-Luke cosine (raw)", np.diag(S[("raw", "tt")]), 1),
            ("A: Matt-Luke cosine (centred)", np.diag(S[("cen", "tt")]), 2),
            ("Matt-Mark cosine (raw)", np.diag(S[("raw", "mk")]), 4),
            ("Matt-Mark cosine (centred)", np.diag(S[("cen", "mk")]), 6),
            ("Luke-Mark cosine (raw)", np.diag(S[("raw", "lk")]), 5),
            ("Luke-Mark cosine (centred)", np.diag(S[("cen", "lk")]), 7),
            ("B: residual correlation vs pericope-specific Mark", sigb_v, 3)]
    cal_rows = []
    for name, v, k in rows:
        lo, hi = bootstrap_ci(v, SEED + k)
        out.append(f"| {name} | {v.mean():.4f} | {v.std(ddof=1):.4f} | [{lo:.4f}, {hi:.4f}] |")
        cal_rows.append([name, f"{v.mean():.4f}", f"{v.std(ddof=1):.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    write_csv(OUT / "calibration_table.csv", ["signature", "mean", "sd", "ci_lower", "ci_upper"], cal_rows)
    out.append("")
    out.append("Triple-tradition Matt-Luke centred cosine, lowest five (alignment check):")
    siga_cen = np.diag(S[("cen", "tt")])
    for i in np.argsort(siga_cen)[:5]:
        out.append(f"  {tlabels[i]:<28s} {siga_cen[i]:.4f}")
    out.append("")
    key["sig_b_mean"] = float(sigb_v.mean())

    # ------------------------------------------------------------------
    # 2. Floors, excess and uncertainty (Table 2)
    # ------------------------------------------------------------------
    out.append("## 2. Floors and excess over floor (Table 2)\n")
    out.append("| Set | Space | Matched | Mean floor | NN floor | Excess (mean floor) [95% CI] | Excess (NN floor) [95% CI] |")
    out.append("|---|---|---|---|---|---|---|")
    E = {}        # per-pericope excess values
    floor_rows = []
    for tag in ("raw", "cen"):
        for s, name in SETS:
            f = floors(S[(tag, s)])
            e, e_nn = per_pericope_excess(S[(tag, s)])
            E[(tag, s)] = (e, e_nn)
            lo, hi = bootstrap_ci(e, SEED + 11)
            lo2, hi2 = bootstrap_ci(e_nn, SEED + 12)
            out.append(f"| {name} | {tag} | {f['matched']:.4f} | {f['floor']:.4f} | {f['nn']:.4f} | "
                       f"{f['excess']:.4f} [{lo:.4f}, {hi:.4f}] | {f['excess_nn']:.4f} [{lo2:.4f}, {hi2:.4f}] |")
            floor_rows.append([name, tag, f"{f['matched']:.4f}", f"{f['floor']:.4f}", f"{f['nn']:.4f}",
                               f"{f['excess']:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                               f"{f['excess_nn']:.4f}", f"{lo2:.4f}", f"{hi2:.4f}"])
            for k_, v_ in f.items():
                key[f"{tag}_{s}_{k_}"] = v_
            key[f"{tag}_{s}_excess_ci"] = [lo, hi]
            key[f"{tag}_{s}_excess_nn_ci"] = [lo2, hi2]
    write_csv(OUT / "floor_table.csv",
              ["set", "space", "matched", "mean_floor", "nn_floor", "excess", "excess_ci_lo", "excess_ci_hi",
               "excess_nn", "excess_nn_ci_lo", "excess_nn_ci_hi"], floor_rows)
    out.append("")
    out.append("Contrasts between sets (difference of mean per-pericope excess, a minus b; "
               "bootstrap CI over pericopes, Mann-Whitney U, label-permutation p):\n")
    out.append("| Contrast | Space | Floor | Diff | 95% CI | MWU p | perm p |")
    out.append("|---|---|---|---|---|---|---|")
    contrasts = [("dt", "tt"), ("dt", "lk"), ("dt", "mk"), ("lk", "tt"), ("mk", "tt")]
    for tag in ("raw", "cen"):
        for a, b in contrasts:
            for which, idx in (("mean", 0), ("NN", 1)):
                r = compare_excess(E[(tag, a)][idx], E[(tag, b)][idx], SEED + 21)
                out.append(f"| {a.upper()} - {b.upper()} | {tag} | {which} | {r['diff']:+.4f} | "
                           f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_mwu']:.3g} | {r['p_perm']:.3g} |")
                key[f"{tag}_{a}_minus_{b}_{which}"] = r
    out.append("")
    # Size-matched nearest-neighbour excess (the NN floor is an order statistic).
    big_is_triple = nt > n
    k_small = min(n, nt)
    for tag in ("raw", "cen"):
        Sm = S[(tag, "tt")] if big_is_triple else S[(tag, "dt")]
        exc = size_matched_nn_excess(Sm, k_small, SEED)
        lo, hi = np.percentile(exc, [2.5, 97.5])
        which = "Triple" if big_is_triple else "Double"
        other = floors(S[(tag, "dt" if big_is_triple else "tt")])["excess_nn"]
        out.append(f"- {which} tradition subsampled to n={k_small}, excess over nearest floor ({tag}): "
                   f"mean {exc.mean():+.4f}, SD {exc.std():.4f}, 95% of draws [{lo:+.4f}, {hi:+.4f}]; "
                   f"the other tradition's value is {other:+.4f}")
        key[f"{tag}_size_matched_nn_excess"] = dict(mean=float(exc.mean()), sd=float(exc.std()),
                                                   lo=float(lo), hi=float(hi), subsampled=which)
    out.append("")

    def within(V, centre=None):
        Sm = cos_matrix(V, V, centre)
        o = ~np.eye(len(V), dtype=bool)
        return float(Sm[o].mean())

    out.append("Internal homogeneity (mean raw cosine among one evangelist's own pericopes within a tradition):")
    out.append(f"- Triple tradition: Matthew {within(TM):.4f} | Mark {within(TK):.4f} | Luke {within(TL):.4f}")
    out.append(f"- Double tradition: Matthew {within(M):.4f} | Luke {within(L):.4f}")
    norm_dt = float(np.mean(np.linalg.norm(np.vstack([M, L]) - mu, axis=1)))
    norm_tt = float(np.mean(np.linalg.norm(np.vstack([TM, TL]) - mu, axis=1)))
    norm_mk = float(np.mean(np.linalg.norm(TK - mu, axis=1)))
    out.append(f"- Mean centred norm: double-tradition vectors {norm_dt:.4f}; triple-tradition Matt+Luke "
               f"{norm_tt:.4f}; Mark {norm_mk:.4f}")
    out.append("")
    key.update(norm_dt=norm_dt, norm_tt=norm_tt, norm_mk=norm_mk)

    # Sensitivity to the centring vector (Table 4)
    mu_dt = np.vstack([M, L]).mean(axis=0)
    mu_tt = np.vstack([TM, TK, TL]).mean(axis=0)
    schemes = [(f"All {len(all_vecs)} pericope vectors (used throughout)", mu),
               (f"Double-tradition vectors only ({2*n})", mu_dt),
               (f"Triple-tradition vectors only ({3*nt})", mu_tt),
               (f"Mark vectors only ({nt})", centroid),
               ("Mean of the two tradition means (balanced)", (mu_dt + mu_tt) / 2.0),
               (f"Matthew and Luke vectors only, both traditions ({2*n + 2*nt})",
                np.vstack([M, L, TM, TL]).mean(axis=0))]
    out.append("### 2c. Sensitivity to the centring vector (Table 4)\n")
    out.append("| Centring vector | DT matched | DT excess | TT matched | TT excess | Luke-Mark excess |")
    out.append("|---|---|---|---|---|---|")
    key["centring_schemes"] = []
    for name, centre in schemes:
        fd = floors(cos_matrix(M, L, centre))
        ft = floors(cos_matrix(TM, TL, centre))
        fl = floors(cos_matrix(TL, TK, centre))
        out.append(f"| {name} | {fd['matched']:.4f} | {fd['excess']:.4f} | {ft['matched']:.4f} | "
                   f"{ft['excess']:.4f} | {fl['excess']:.4f} |")
        key["centring_schemes"].append(dict(name=name, dt_matched=fd["matched"], dt_excess=fd["excess"],
                                            tt_matched=ft["matched"], tt_excess=ft["excess"],
                                            lk_excess=fl["excess"]))
    out.append("")

    # ------------------------------------------------------------------
    # 3. Permutation tests (double tradition, Matt-Luke)
    # ------------------------------------------------------------------
    raw = permutation_tests(S[("raw", "dt")], len_m, len_l, tags)
    cen = permutation_tests(S[("cen", "dt")], len_m, len_l, tags)
    out.append(fmt_null_block("3a. Raw-cosine permutation tests (double tradition)", raw))
    out.append(fmt_null_block("3b. Mean-centred cosine permutation tests (primary statistic)", cen))
    off = ~np.eye(n, dtype=bool)
    ratio = np.abs(np.log(len_m[:, None] / len_l[None, :]))
    rho_off, p_off = stats.spearmanr(S[("cen", "dt")][off], ratio[off])
    out.append(f"- Length mismatch |log(len_Matt/len_Luke)|: matched pairs mean {np.diag(ratio).mean():.3f}, "
               f"mismatched pairs mean {ratio[off].mean():.3f}; Spearman rho between centred cosine and "
               f"length mismatch over mismatched pairs = {rho_off:.3f} (p = {p_off:.2g}).")
    out.append("")
    for name in ("random", "thematic", "length", "length_theme"):
        key[f"centred_{name}_null_mean"] = float(cen[name].mean())
        key[f"centred_{name}_null_sd"] = float(cen[name].std())
        key[f"centred_{name}_p"] = perm_pvalue(cen["observed"], cen[name])
        key[f"raw_{name}_null_mean"] = float(raw[name].mean())
        key[f"raw_{name}_null_sd"] = float(raw[name].std())
        key[f"raw_{name}_p"] = perm_pvalue(raw["observed"], raw[name])
    key["centred_matched_mean"] = float(cen["observed"])
    key["raw_matched_mean"] = float(raw["observed"])
    key["length_mismatch_rho"] = float(rho_off)
    # The same nulls for the triple tradition (no thematic tags there).
    cen_tt = permutation_tests(S[("cen", "tt")], tlen_m, tlen_l, None)
    out.append(fmt_null_block("3c. Mean-centred cosine permutation tests (triple tradition, Matt-Luke)", cen_tt))

    # ------------------------------------------------------------------
    # 4. Pericope-specific residual correlation (minor agreements)
    # ------------------------------------------------------------------
    RMt = np.array([residual_vector(TM[i], TK[i]) for i in range(nt)])
    RLt = np.array([residual_vector(TL[i], TK[i]) for i in range(nt)])
    S_resid_tt = cos_matrix(RMt, RLt)
    f_res = floors(S_resid_tt)
    e_res, e_res_nn = per_pericope_excess(S_resid_tt)
    res_null = permutation_tests(S_resid_tt, tlen_m, tlen_l, None, seed=SEED + 3)
    lo, hi = bootstrap_ci(e_res, SEED + 31)
    out.append("## 4. Pericope-specific residual correlation in the triple tradition\n")
    out.append("What Matthew adds to Mark correlated with what Luke adds to Mark, pericope by pericope "
               "(the geometric form of the minor agreements).\n")
    out.append(f"- Matched mean (Signature B)          : {f_res['matched']:.4f}")
    out.append(f"- Mismatched-pair floor               : {f_res['floor']:.4f}")
    out.append(f"- Nearest non-parallel floor          : {f_res['nn']:.4f}")
    out.append(f"- Excess over mean floor [95% CI]     : {f_res['excess']:.4f} [{lo:.4f}, {hi:.4f}]")
    out.append(f"- Excess over NN floor                : {f_res['excess_nn']:.4f}")
    out.append("")
    out.append(fmt_null_block("4b. Permutation tests on the residual correlation (triple tradition)", res_null))
    key["resid_tt"] = dict(**f_res, excess_ci=[lo, hi],
                           random_null_mean=float(res_null["random"].mean()),
                           random_null_sd=float(res_null["random"].std()),
                           random_p=perm_pvalue(res_null["observed"], res_null["random"]),
                           length_null_mean=float(res_null["length"].mean()),
                           length_p=perm_pvalue(res_null["observed"], res_null["length"]))
    # Equivalence of the centroid residual and Mark-centred cosine (double tradition).
    RM = np.array([residual_vector(m, centroid) for m in M])
    RL = np.array([residual_vector(l, centroid) for l in L])
    S_res_dt = cos_matrix(RM, RL)
    f_rdt = floors(S_res_dt)
    f_mkc = floors(cos_matrix(M, L, centroid))
    rho_eq, _ = stats.spearmanr(np.diag(S_res_dt), np.diag(S[("cen", "dt")]))
    out.append("Centroid residual (double tradition) versus Mark-centred cosine:")
    out.append(f"- residual-correlation matched mean {f_rdt['matched']:.4f}, floor {f_rdt['floor']:.4f}; "
               f"Mark-centred cosine matched mean {f_mkc['matched']:.4f}, floor {f_mkc['floor']:.4f}; "
               f"Spearman between residual correlation and the primary centred cosine over the "
               f"{n} pericopes = {rho_eq:.3f}")
    out.append("")
    key["resid_dt"] = dict(matched=f_rdt["matched"], floor=f_rdt["floor"],
                           mark_centred_matched=f_mkc["matched"], mark_centred_floor=f_mkc["floor"],
                           spearman_vs_centred=float(rho_eq))

    # ------------------------------------------------------------------
    # 5. Mark-Q overlaps
    # ------------------------------------------------------------------
    ov = np.array([l in MARK_Q_OVERLAP for l in labels])
    out.append("## 5. Double-tradition statistics with and without the Mark-Q overlap units\n")
    out.append(f"Units with a Markan parallel to all or part of their text (n = {int(ov.sum())}): "
               + "; ".join(f"{l} (Mark {MARK_Q_OVERLAP[l]['mark']}, {MARK_Q_OVERLAP[l]['extent']})"
                           for l in labels if l in MARK_Q_OVERLAP) + "\n")
    out.append("| Subset | n | Space | Matched | Mean floor | Excess | NN excess |")
    out.append("|---|---|---|---|---|---|---|")
    key["mark_q_overlap"] = {"n_overlap": int(ov.sum()), "labels": [l for l in labels if l in MARK_Q_OVERLAP]}
    for tag in ("raw", "cen"):
        for sub_name, mask in (("all", np.ones(n, bool)), ("without overlaps", ~ov), ("overlaps only", ov)):
            idx = np.where(mask)[0]
            f = floors(S[(tag, "dt")][np.ix_(idx, idx)])
            out.append(f"| {sub_name} | {len(idx)} | {tag} | {f['matched']:.4f} | {f['floor']:.4f} | "
                       f"{f['excess']:.4f} | {f['excess_nn']:.4f} |")
            key["mark_q_overlap"][f"{tag}_{sub_name.replace(' ', '_')}"] = f
        # Contrast of the non-overlap subset against the triple tradition.
        idx = np.where(~ov)[0]
        e_sub, _ = per_pericope_excess(S[(tag, "dt")][np.ix_(idx, idx)])
        r = compare_excess(e_sub, E[(tag, "tt")][0], SEED + 41)
        out.append(f"- Non-overlap DT minus TT excess ({tag}): {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                   f"MWU p = {r['p_mwu']:.3g}")
        key["mark_q_overlap"][f"{tag}_nonoverlap_minus_tt"] = r
    out.append("")

    # ------------------------------------------------------------------
    # 6. Composite units: range-wise vs single-sequence encoding
    # ------------------------------------------------------------------
    out.append("## 6. Composite units: range-wise versus single-sequence encoding\n")
    out.append("| Pericope | Set | Composite side | Centred cos (range-wise, used) | Centred cos (single sequence) | Difference |")
    out.append("|---|---|---|---|---|---|")
    comp_rows = []
    for p_, A_, B_, s_ in ([(p, M[i], L[i], "double") for i, p in enumerate(double)]
                           + [(p, TM[i], TL[i], "triple") for i, p in enumerate(triple)]):
        comp_sides = [g for g in ("matthew", "luke") if p_.is_composite(g)]
        if not comp_sides:
            continue
        a_single = pipe.embed_text(" ".join(p_.matthew_parts)) if "matthew" in comp_sides else A_
        b_single = pipe.embed_text(" ".join(p_.luke_parts)) if "luke" in comp_sides else B_
        c_used = cos(A_ - mu, B_ - mu)
        c_single = cos(np.asarray(a_single, float) - mu, np.asarray(b_single, float) - mu)
        out.append(f"| {p_.label} | {s_} | {', '.join(comp_sides)} | {c_used:.4f} | {c_single:.4f} | {c_single - c_used:+.4f} |")
        comp_rows.append([p_.label, s_, ", ".join(comp_sides), f"{c_used:.4f}", f"{c_single:.4f}", f"{c_single - c_used:+.4f}"])
    write_csv(OUT / "composite_sensitivity.csv",
              ["label", "set", "composite_side", "centred_rangewise", "centred_single", "difference"], comp_rows)
    key["composite_max_abs_diff"] = float(max(abs(float(r[5])) for r in comp_rows)) if comp_rows else 0.0
    out.append("")

    # ------------------------------------------------------------------
    # 7. Directionality
    # ------------------------------------------------------------------
    def cnorm(V):
        A = V - mu
        return A / np.linalg.norm(A, axis=1, keepdims=True)
    Mc, Lc = cnorm(M), cnorm(L)
    sse = {a: loo_sse(Mc, Lc, a) + loo_sse(Lc, Mc, a) for a in ALPHA_GRID}
    alpha = min(sse, key=sse.get)
    r2_lm, r2_ml = loo_r2(Mc, Lc, alpha), loo_r2(Lc, Mc, alpha)
    d_obs = r2_lm - r2_ml
    jk = np.array([delta_r2(np.delete(Mc, i, 0), np.delete(Lc, i, 0), alpha) for i in range(n)])
    jk_se = float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2)))
    ci_lo, ci_hi = d_obs - 1.96 * jk_se, d_obs + 1.96 * jk_se
    _, p_two = signflip_p(Mc, Lc, alpha, SEED + 13)
    grid_rows = [(a,) + signflip_p(Mc, Lc, a, SEED + 17) for a in ALPHA_GRID]
    bal = np.abs(np.log(len_m / len_l)) < 0.25
    d_bal, p_bal = signflip_p(Mc[bal], Lc[bal], alpha, SEED + 19)
    Tm, Tk, Tl = cnorm(TM), cnorm(TK), cnorm(TL)
    d_km, p_km = signflip_p(Tk, Tm, alpha, SEED + 23)
    d_kl, p_kl = signflip_p(Tk, Tl, alpha, SEED + 29)
    d_ml, p_ml = signflip_p(Tm, Tl, alpha, SEED + 31)
    edf = effective_df(Mc, alpha)
    norm_m = np.linalg.norm(RM, axis=1)
    norm_l = np.linalg.norm(RL, axis=1)
    rho_len, p_len = stats.spearmanr(np.r_[norm_m, norm_l], np.r_[len_m, len_l])
    w_stat, w_p = stats.wilcoxon(norm_m, norm_l)
    out.append("## 7. Directionality analysis\n")
    out.append(f"- Ridge penalty alpha (min total LOO error) : {alpha}; effective df of the fit tr(H) = {edf:.1f} of n = {n}")
    out.append("- LOO error by alpha: " + "; ".join(f"{a}: {sse[a]:.3f}" for a in ALPHA_GRID))
    out.append(f"- Exact LOO ridge R^2, Luke | Matthew      : {r2_lm:.4f}")
    out.append(f"- Exact LOO ridge R^2, Matthew | Luke      : {r2_ml:.4f}")
    out.append(f"- delta-R2 (L|M minus M|L)                 : {d_obs:+.4f}")
    out.append(f"- Jackknife SE / 95% CI                    : {jk_se:.4f} / [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    out.append(f"- Sign-flip permutation p (two-sided)      : {p_two:.4g}")
    out.append("- delta-R2 and sign-flip p across the alpha grid: " +
               "; ".join(f"{a}: {d0:+.4f} (p = {pp:.3f})" for a, d0, pp in grid_rows))
    out.append("")
    out.append("Controls:")
    out.append(f"- Length-balanced double-tradition pairs (|log length ratio| < 0.25, n = {int(bal.sum())}): "
               f"delta-R2 = {d_bal:+.4f}, sign-flip p = {p_bal:.3f}")
    out.append(f"- Known direction, triple tradition (n = {nt}): Mark -> Matthew delta-R2 = {d_km:+.4f} "
               f"(p = {p_km:.3f}); Mark -> Luke delta-R2 = {d_kl:+.4f} (p = {p_kl:.3f}); "
               f"Matthew vs Luke (both copying Mark) delta-R2 = {d_ml:+.4f} (p = {p_ml:.3f})")
    out.append("")
    out.append("Residual-norm diagnostic (NOT used as evidence of directionality):")
    out.append(f"- Residual norm, Matthew mean {norm_m.mean():.4f}, Luke mean {norm_l.mean():.4f}; "
               f"Wilcoxon W = {w_stat:.1f}, p = {w_p:.3g}; residual norm vs length (pooled): "
               f"Spearman rho = {rho_len:.3f} (p = {p_len:.2g})")
    out.append("")
    key.update(alpha=alpha, effective_df=edf, r2_luke_given_matt=r2_lm, r2_matt_given_luke=r2_ml,
               delta_r2=d_obs, delta_r2_ci=[ci_lo, ci_hi], signflip_p=p_two,
               alpha_grid=[dict(alpha=a, delta_r2=d0, p=pp) for a, d0, pp in grid_rows],
               n_length_balanced=int(bal.sum()),
               delta_r2_length_balanced=d_bal, signflip_p_length_balanced=p_bal,
               delta_r2_mark_matt=d_km, signflip_p_mark_matt=p_km,
               delta_r2_mark_luke=d_kl, signflip_p_mark_luke=p_kl,
               delta_r2_matt_luke_tt=d_ml, signflip_p_matt_luke_tt=p_ml,
               resid_norm_length_rho=float(rho_len))

    # ------------------------------------------------------------------
    # 8. Ranking and per-pericope statistics
    # ------------------------------------------------------------------
    ccos_vals = np.diag(S[("cen", "dt")]).copy()
    raw_vals = np.diag(S[("raw", "dt")]).copy()
    resid_vals = np.diag(S_res_dt).copy()
    wo = np.array([word_overlap(p.matthew, p.luke) for p in double])
    order = np.argsort(ccos_vals)[::-1]
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    refs = {lab: (format_ref(m), format_ref(l)) for lab, m, l in DOUBLE_TRADITION}
    e_cen, e_cen_nn = E[("cen", "dt")]
    e_raw, e_raw_nn = E[("raw", "dt")]
    rows = []
    for i in order:
        rows.append([rank[i], labels[i], refs[labels[i]][0], refs[labels[i]][1],
                     stratum_of(labels[i]), FORM[labels[i]], THEMATIC_TAGS[labels[i]],
                     f"{raw_vals[i]:.4f}", f"{ccos_vals[i]:.4f}", f"{resid_vals[i]:.4f}",
                     f"{e_cen[i]:.4f}", f"{e_cen_nn[i]:.4f}",
                     int(len_m[i]), int(len_l[i]), f"{wo[i]:.4f}",
                     MARK_Q_OVERLAP[labels[i]]["mark"] if labels[i] in MARK_Q_OVERLAP else "",
                     TEXT_CRITICAL_NOTES.get(labels[i], "")])
    write_csv(OUT / "centred_cosine_ranking.csv",
              ["rank", "label", "matt_ref", "luke_ref", "stratum", "form", "theme",
               "matt_luke_cos", "centred_cos", "residual_sim", "excess_mean_floor", "excess_nn_floor",
               "len_matt_chars", "len_luke_chars", "word_overlap", "mark_parallel", "text_critical_note"], rows)
    # Per-pericope excess for every set (for the figure script and the appendix).
    pe_rows = []
    for s, name in SETS:
        labs = labels if s == "dt" else tlabels
        for i, lab in enumerate(labs):
            pe_rows.append([s, lab, f"{E[('raw', s)][0][i]:.4f}", f"{E[('raw', s)][1][i]:.4f}",
                            f"{E[('cen', s)][0][i]:.4f}", f"{E[('cen', s)][1][i]:.4f}"])
    write_csv(OUT / "per_pericope_excess.csv",
              ["set", "label", "raw_excess_mean", "raw_excess_nn", "centred_excess_mean", "centred_excess_nn"], pe_rows)
    # Triple-tradition table for the appendix.
    tt_rows = []
    for i, lab in enumerate(tlabels):
        m_ref, k_ref, l_ref = [format_ref(r) for r in next(t for t in TRIPLE_TRADITION if t[0] == lab)[1:]]
        tt_rows.append([lab, m_ref, k_ref, l_ref,
                        f"{S[('cen', 'tt')][i, i]:.4f}", f"{S[('cen', 'mk')][i, i]:.4f}", f"{S[('cen', 'lk')][i, i]:.4f}",
                        f"{S[('raw', 'tt')][i, i]:.4f}", f"{sigb_v[i]:.4f}", f"{E[('cen', 'tt')][0][i]:.4f}"])
    write_csv(OUT / "triple_tradition_table.csv",
              ["label", "matt_ref", "mark_ref", "luke_ref", "centred_matt_luke", "centred_matt_mark",
               "centred_luke_mark", "raw_matt_luke", "residual_corr", "centred_excess_matt_luke"], tt_rows)

    out.append("## 8. Centred-cosine ranking (primary ranking statistic)\n")
    out.append(f"- Mean (SD, sample)    : {ccos_vals.mean():.4f} ({ccos_vals.std(ddof=1):.4f})")
    out.append(f"- Range                : [{ccos_vals.min():.4f}, {ccos_vals.max():.4f}]")
    out.append("Top five:")
    for i in order[:5]:
        out.append(f"  {rank[i]}. {labels[i]:<30s} centred={ccos_vals[i]:.4f} raw={raw_vals[i]:.4f} jaccard={wo[i]:.4f}")
    out.append("Bottom five:")
    for i in order[-5:]:
        out.append(f"  {rank[i]}. {labels[i]:<30s} centred={ccos_vals[i]:.4f} raw={raw_vals[i]:.4f} jaccard={wo[i]:.4f}")
    out.append("")
    key["top5"] = [dict(rank=int(rank[i]), label=labels[i], centred=float(ccos_vals[i]), raw=float(raw_vals[i]))
                   for i in order[:5]]
    key["bottom5"] = [dict(rank=int(rank[i]), label=labels[i], centred=float(ccos_vals[i]), raw=float(raw_vals[i]))
                      for i in order[-5:]]
    key["centred_sd"] = float(ccos_vals.std(ddof=1))
    key["centred_min"], key["centred_max"] = float(ccos_vals.min()), float(ccos_vals.max())
    key["ranks"] = {labels[i]: int(rank[i]) for i in range(n)}

    # Confounds
    lv = (len_m + len_l) / 2.0
    r_p, p_p = stats.pearsonr(lv, ccos_vals)
    r_s, p_s = stats.spearmanr(lv, ccos_vals)
    forms = [FORM[l] for l in labels]
    fgroups = {}
    for f_, v in zip(forms, ccos_vals):
        fgroups.setdefault(f_, []).append(v)
    big = {k_: np.array(v) for k_, v in fgroups.items() if len(v) >= 3}
    h_f, p_f = stats.kruskal(*big.values())
    # Pairwise Mann-Whitney with Holm correction (descriptive)
    pairs = [(a, b) for i_, a in enumerate(big) for b in list(big)[i_ + 1:]]
    pvals = [stats.mannwhitneyu(big[a], big[b], alternative="two-sided")[1] for a, b in pairs]
    order_p = np.argsort(pvals)
    holm = {}
    m_ = len(pvals)
    running = 0.0
    for rank_i, j in enumerate(order_p):
        adj = min(1.0, (m_ - rank_i) * pvals[j])
        running = max(running, adj)
        holm[pairs[j]] = running
    strata = [stratum_of(l) for l in labels]
    q1 = ccos_vals[np.array([s == "Q1" for s in strata])]
    q2 = ccos_vals[np.array([s == "Q2" for s in strata])]
    u_st, p_u = stats.mannwhitneyu(q1, q2, alternative="two-sided")
    gmask = np.array([l in GOULDER_REDACTION_LABELS for l in labels])
    g, ng = ccos_vals[gmask], ccos_vals[~gmask]
    t_g, p_g = stats.ttest_ind(g, ng, equal_var=False)
    pooled = np.sqrt(((len(g) - 1) * g.var(ddof=1) + (len(ng) - 1) * ng.var(ddof=1)) / (n - 2))
    d_g = float((g.mean() - ng.mean()) / (pooled + 1e-12))
    r_w, p_w = stats.pearsonr(ccos_vals, wo)
    rho_w, _ = stats.spearmanr(ccos_vals, wo)
    out.append("### 8b. Confound analyses on the centred cosine\n")
    out.append(f"- Length (characters): Pearson r = {r_p:.3f} (p = {p_p:.3f}); Spearman rho = {r_s:.3f} (p = {p_s:.3f})")
    for f_ in ["proverbial", "discourse", "parable", "narrative", "liturgical"]:
        if f_ in fgroups:
            v = np.array(fgroups[f_])
            out.append(f"- Form {f_:<11s}: mean = {v.mean():.4f}, median = {np.median(v):.4f}, n = {len(v)}")
    out.append(f"- Form Kruskal-Wallis (n >= 3 groups): H = {h_f:.3f}, p = {p_f:.3f}")
    out.append("- Pairwise Mann-Whitney, Holm-adjusted p: " +
               "; ".join(f"{a} vs {b}: {holm[(a, b)]:.3f}" for a, b in pairs))
    counts = {k_: strata.count(k_) for k_ in ("Q1", "Q2", "Q3", "U")}
    out.append(f"- Strata counts: {counts}; Q1 mean = {q1.mean():.4f} (n = {len(q1)}), Q2 mean = {q2.mean():.4f} "
               f"(n = {len(q2)}); Mann-Whitney U = {u_st:.1f}, p = {p_u:.3f}")
    out.append(f"- Goulder set: mean = {g.mean():.4f} (n = {len(g)}) vs {ng.mean():.4f} (n = {len(ng)}); "
               f"Welch t = {t_g:.3f}, p = {p_g:.4f}, Cohen's d = {d_g:.3f}")
    out.append(f"- Word overlap (multiset Jaccard): Pearson r = {r_w:.3f} (p = {p_w:.3g}), Spearman rho = {rho_w:.3f}; "
               f"unexplained variance ~ {100 * (1 - r_w**2):.0f}%")
    out.append("")
    key.update(length_pearson_r=float(r_p), length_pearson_p=float(p_p), length_spearman_rho=float(r_s),
               form_means={k_: float(np.mean(v)) for k_, v in fgroups.items()},
               form_n={k_: len(v) for k_, v in fgroups.items()},
               form_kw_h=float(h_f), form_kw_p=float(p_f),
               form_pairwise_holm={f"{a} vs {b}": float(holm[(a, b)]) for a, b in pairs},
               strata_counts=counts, strata_q1_mean=float(q1.mean()), strata_q2_mean=float(q2.mean()),
               strata_mw_u=float(u_st), strata_mw_p=float(p_u),
               goulder_mean=float(g.mean()), goulder_other_mean=float(ng.mean()), goulder_n=int(gmask.sum()),
               goulder_t=float(t_g), goulder_p=float(p_g), goulder_d=d_g,
               word_overlap_r=float(r_w), word_overlap_rho=float(rho_w))

    # Sentence-level bootstrap on the centred cosine (needs the model)
    rng = np.random.default_rng(SEED + 99)
    stds, skipped = [], []
    for p in double:
        ms, ls = split_sentences(p.matthew), split_sentences(p.luke)
        if len(ms) < 2 or len(ls) < 2:
            skipped.append(p.label)
            continue
        vals = []
        for _ in range(N_SENT_BOOT):
            mi = rng.choice(len(ms), size=len(ms), replace=True)
            li = rng.choice(len(ls), size=len(ls), replace=True)
            em = pipe.embed_text(" ".join(ms[k_] for k_ in mi))
            el = pipe.embed_text(" ".join(ls[k_] for k_ in li))
            vals.append(cos(np.asarray(em, float) - mu, np.asarray(el, float) - mu))
        stds.append(float(np.std(vals)))
    out.append("### 8c. Sentence-level bootstrap on the centred cosine\n")
    out.append(f"- Pericopes with >= 2 sentences in both versions: {len(stds)} (excluded: {', '.join(skipped)})")
    out.append(f"- Per-pericope bootstrap SD ({N_SENT_BOOT} resamples): mean {np.mean(stds):.4f}, median {np.median(stds):.4f}")
    out.append("")
    key.update(sentence_bootstrap_sd=float(np.mean(stds)), sentence_bootstrap_sd_median=float(np.median(stds)),
               sentence_bootstrap_n=len(stds), sentence_bootstrap_excluded=skipped)

    # Pair values for Figure 1.
    offt = ~np.eye(nt, dtype=bool)
    pv = [["double", "mismatched", f"{v:.4f}"] for v in S[("cen", "dt")][off]]
    pv += [["double", "matched", f"{v:.4f}"] for v in np.diag(S[("cen", "dt")])]
    pv += [["triple", "matched", f"{v:.4f}"] for v in np.diag(S[("cen", "tt")])]
    pv += [["triple", "mismatched", f"{v:.4f}"] for v in S[("cen", "tt")][offt]]
    write_csv(OUT / "pair_values_centred.csv", ["set", "kind", "value"], pv)

    def _clean(o):
        if isinstance(o, dict):
            return {k_: _clean(v) for k_, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "key_stats.json").write_text(json.dumps(_clean(key), indent=2), encoding="utf-8")
    rep = OUT / "advanced_analysis.md"
    rep.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nReport: {rep}")


if __name__ == "__main__":
    main()
