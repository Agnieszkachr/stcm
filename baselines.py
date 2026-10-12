"""
baselines.py
============
Lexical baselines for the embedding analysis: what does the transformer
representation add over plain word overlap?

Two lexical similarity measures are run through the same pipeline as the
centred cosine (floors, excess over floor with uncertainty, permutation
nulls, between-set contrasts, ranking):

* Jaccard   — multiset (token-level) Jaccard coefficient on accent-stripped,
              lower-cased word forms (the same ``word_tokens`` used for the
              word-overlap comparison in the article);
* TF-IDF    — cosine similarity of character 3-5-gram TF-IDF vectors fitted
              on all unit texts of both traditions (a sub-word lexical
              measure that is insensitive to inflectional endings).

Then the embedding-specific increment is isolated: the centred-cosine pair
matrix is regressed on the Jaccard pair matrix over all Matthew-Luke pairs,
and the residual matrix is tested for matched-versus-mismatched separation;
the pericopes on which the two measures disagree most are tabulated.

Outputs
-------
    outputs/reports/baselines.md
    outputs/reports/baseline_floor_table.csv
    outputs/reports/baseline_disagreements.csv
    outputs/reports/key_stats_baselines.json
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.analysis_core import (SEED, N_PERM, cos_matrix, floors, per_pericope_excess, compare_excess,
                                bootstrap_ci, permutation_tests, fmt_null_block, perm_pvalue)
from stcm.data_loader import SBLGNTLoader
from stcm.embeddings import EmbeddingPipeline
from stcm.metadata import THEMATIC_TAGS, FORM
from stcm.utils import word_tokens, normalise_greek, strip_accents

OUT = _REPO / "outputs" / "reports"


def jaccard_matrix(A_texts, B_texts):
    from collections import Counter
    CA = [Counter(word_tokens(t)) for t in A_texts]
    CB = [Counter(word_tokens(t)) for t in B_texts]
    S = np.empty((len(CA), len(CB)))
    for i, ca in enumerate(CA):
        for j, cb in enumerate(CB):
            inter = sum((ca & cb).values())
            union = sum((ca | cb).values())
            S[i, j] = inter / union if union else 0.0
    return S


def clean(t):
    return strip_accents(normalise_greek(t)).lower()


def main():
    corpus = SBLGNTLoader().load()
    pipe = EmbeddingPipeline()
    double = corpus.double_tradition
    triple = corpus.triple_tradition
    labels = [p.label for p in double]
    tlabels = [p.label for p in triple]
    tags = [THEMATIC_TAGS[l] for l in labels]
    n, nt = len(double), len(triple)
    len_m = np.array([len(p.matthew) for p in double], float)
    len_l = np.array([len(p.luke) for p in double], float)
    tlen_m = np.array([len(p.matthew) for p in triple], float)
    tlen_l = np.array([len(p.luke) for p in triple], float)

    # Embedding matrices (primary configuration) for the comparison.
    M = np.array([pipe.embed_parts(p.matthew_parts) for p in double], float)
    L = np.array([pipe.embed_parts(p.luke_parts) for p in double], float)
    TM = np.array([pipe.embed_parts(p.matthew_parts) for p in triple], float)
    TK = np.array([pipe.embed_parts(p.mark_parts) for p in triple], float)
    TL = np.array([pipe.embed_parts(p.luke_parts) for p in triple], float)
    mu = np.vstack([M, L, TM, TK, TL]).mean(axis=0)
    C_dt = cos_matrix(M, L, mu)
    C_tt = cos_matrix(TM, TL, mu)

    # Lexical matrices
    texts = {"dt": ([p.matthew for p in double], [p.luke for p in double]),
             "tt": ([p.matthew for p in triple], [p.luke for p in triple]),
             "mk": ([p.matthew for p in triple], [p.mark for p in triple]),
             "lk": ([p.luke for p in triple], [p.mark for p in triple])}
    all_texts = [clean(t) for p in corpus.all_pericopes for t in (p.matthew, p.mark, p.luke) if t]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(all_texts)

    def tfidf_matrix(A_texts, B_texts):
        A = tfidf.transform([clean(t) for t in A_texts])
        B = tfidf.transform([clean(t) for t in B_texts])
        return (A @ B.T).toarray()

    measures = {"Jaccard": jaccard_matrix, "TF-IDF char 3-5": tfidf_matrix}
    SETS = [("dt", "Double tradition, Matt-Luke"), ("tt", "Triple tradition, Matt-Luke"),
            ("mk", "Triple tradition, Matt-Mark"), ("lk", "Triple tradition, Luke-Mark")]
    out = ["# Lexical baselines", "", f"(n_double = {n}, n_triple = {nt}, permutations = {N_PERM}, seed = {SEED})", ""]
    key = {"n_double": n, "n_triple": nt}
    floor_rows = []
    Smat = {}
    for mname, fn in measures.items():
        out.append(f"## {mname}\n")
        out.append("| Set | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess [95% CI] |")
        out.append("|---|---|---|---|---|---|")
        E = {}
        for s, name in SETS:
            Sm = fn(*texts[s])
            Smat[(mname, s)] = Sm
            f = floors(Sm)
            e, e_nn = per_pericope_excess(Sm)
            E[s] = (e, e_nn)
            lo, hi = bootstrap_ci(e, SEED + 11)
            lo2, hi2 = bootstrap_ci(e_nn, SEED + 12)
            out.append(f"| {name} | {f['matched']:.4f} | {f['floor']:.4f} | {f['nn']:.4f} | "
                       f"{f['excess']:.4f} [{lo:.4f}, {hi:.4f}] | {f['excess_nn']:.4f} [{lo2:.4f}, {hi2:.4f}] |")
            floor_rows.append([mname, name, f"{f['matched']:.4f}", f"{f['floor']:.4f}", f"{f['nn']:.4f}",
                               f"{f['excess']:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{f['excess_nn']:.4f}", f"{lo2:.4f}", f"{hi2:.4f}"])
            key[f"{mname}_{s}"] = dict(**f, excess_ci=[lo, hi], excess_nn_ci=[lo2, hi2])
        out.append("")
        for a, b in (("dt", "tt"), ("dt", "lk"), ("dt", "mk")):
            r = compare_excess(E[a][0], E[b][0], SEED + 21)
            r_nn = compare_excess(E[a][1], E[b][1], SEED + 22)
            out.append(f"- {a.upper()} minus {b.upper()} excess: {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                       f"MWU p = {r['p_mwu']:.3g}; NN excess {r_nn['diff']:+.4f} [{r_nn['ci_lo']:+.4f}, {r_nn['ci_hi']:+.4f}]")
            key[f"{mname}_{a}_minus_{b}"] = r
            key[f"{mname}_{a}_minus_{b}_nn"] = r_nn
        nulls = permutation_tests(Smat[(mname, "dt")], len_m, len_l, tags)
        out.append("")
        out.append(fmt_null_block(f"{mname}: permutation tests (double tradition)", nulls))
        for nm in ("random", "thematic", "length", "length_theme"):
            key[f"{mname}_dt_{nm}_null_mean"] = float(nulls[nm].mean())
            key[f"{mname}_dt_{nm}_p"] = perm_pvalue(nulls["observed"], nulls[nm])
        # Rank agreement with the centred cosine
        rho, _ = stats.spearmanr(np.diag(Smat[(mname, "dt")]), np.diag(C_dt))
        r_p, _ = stats.pearsonr(np.diag(Smat[(mname, "dt")]), np.diag(C_dt))
        rho_t, _ = stats.spearmanr(np.diag(Smat[(mname, "tt")]), np.diag(C_tt))
        out.append(f"- Rank agreement with the centred cosine: Spearman rho = {rho:.3f} (double tradition), "
                   f"{rho_t:.3f} (triple tradition); Pearson r = {r_p:.3f} (double tradition)")
        out.append("")
        key[f"{mname}_spearman_vs_centred_dt"] = float(rho)
        key[f"{mname}_pearson_vs_centred_dt"] = float(r_p)
        key[f"{mname}_spearman_vs_centred_tt"] = float(rho_t)

    with open(OUT / "baseline_floor_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measure", "set", "matched", "mean_floor", "nn_floor", "excess", "excess_ci_lo", "excess_ci_hi",
                    "excess_nn", "excess_nn_ci_lo", "excess_nn_ci_hi"])
        w.writerows(floor_rows)

    # ------------------------------------------------------------------
    # The embedding-specific increment
    # ------------------------------------------------------------------
    out.append("## Embedding-specific increment\n")
    J = Smat[("Jaccard", "dt")]
    off = ~np.eye(n, dtype=bool)
    # (a) Regress the centred cosine pair matrix on the Jaccard pair matrix over ALL pairs
    x, y = J.ravel(), C_dt.ravel()
    slope, intercept, r_all, _, _ = stats.linregress(x, y)
    R = C_dt - (intercept + slope * J)
    f_R = floors(R)
    nulls_R = permutation_tests(R, len_m, len_l, tags, seed=SEED + 5)
    out.append(f"- Centred cosine ~ Jaccard over all {n*n} Matthew-Luke pairs: slope {slope:.3f}, intercept {intercept:.3f}, "
               f"r = {r_all:.3f} (r^2 = {r_all**2:.3f})")
    out.append(f"- Residual matrix: matched mean {f_R['matched']:+.4f}, mismatched mean {f_R['floor']:+.4f}, "
               f"excess {f_R['excess']:+.4f}, NN excess {f_R['excess_nn']:+.4f}")
    out.append(fmt_null_block("Permutation tests on the residualised centred cosine", nulls_R))
    key["residualised"] = dict(slope=float(slope), intercept=float(intercept), r_all_pairs=float(r_all),
                               **f_R, random_null_mean=float(nulls_R["random"].mean()),
                               random_null_sd=float(nulls_R["random"].std()),
                               random_p=perm_pvalue(nulls_R["observed"], nulls_R["random"]),
                               length_theme_null_mean=float(nulls_R["length_theme"].mean()),
                               length_theme_p=perm_pvalue(nulls_R["observed"], nulls_R["length_theme"]))
    # Same with TF-IDF as the lexical control
    T = Smat[("TF-IDF char 3-5", "dt")]
    slope2, intercept2, r_all2, _, _ = stats.linregress(T.ravel(), C_dt.ravel())
    R2 = C_dt - (intercept2 + slope2 * T)
    f_R2 = floors(R2)
    nulls_R2 = permutation_tests(R2, len_m, len_l, tags, seed=SEED + 6)
    out.append(f"- Centred cosine ~ TF-IDF over all pairs: r = {r_all2:.3f}; residual matched mean {f_R2['matched']:+.4f}, "
               f"mismatched {f_R2['floor']:+.4f}, excess {f_R2['excess']:+.4f}; random-null p = "
               f"{perm_pvalue(nulls_R2['observed'], nulls_R2['random']):.3g}, length+theme p = "
               f"{perm_pvalue(nulls_R2['observed'], nulls_R2['length_theme']):.3g}")
    out.append("")
    key["residualised_tfidf"] = dict(r_all_pairs=float(r_all2), **f_R2,
                                     random_p=perm_pvalue(nulls_R2["observed"], nulls_R2["random"]),
                                     length_theme_p=perm_pvalue(nulls_R2["observed"], nulls_R2["length_theme"]))
    # Does the residual signal survive within the triple tradition too?
    Jt = Smat[("Jaccard", "tt")]
    s_t, i_t, r_t, _, _ = stats.linregress(Jt.ravel(), C_tt.ravel())
    Rt = C_tt - (i_t + s_t * Jt)
    f_Rt = floors(Rt)
    nulls_Rt = permutation_tests(Rt, tlen_m, tlen_l, None, seed=SEED + 7)
    out.append(f"- Triple tradition: centred cosine ~ Jaccard r = {r_t:.3f}; residual matched {f_Rt['matched']:+.4f}, "
               f"mismatched {f_Rt['floor']:+.4f}, random-null p = {perm_pvalue(nulls_Rt['observed'], nulls_Rt['random']):.3g}")
    out.append("")
    key["residualised_tt"] = dict(r_all_pairs=float(r_t), **f_Rt,
                                  random_p=perm_pvalue(nulls_Rt["observed"], nulls_Rt["random"]))

    # (b) Does the DT > TT contrast hold on the residualised measure?
    e_R, _ = per_pericope_excess(R)
    e_Rt, _ = per_pericope_excess(Rt)
    rr = compare_excess(e_R, e_Rt, SEED + 23)
    out.append(f"- Residualised DT minus TT excess: {rr['diff']:+.4f} [{rr['ci_lo']:+.4f}, {rr['ci_hi']:+.4f}], MWU p = {rr['p_mwu']:.3g}")
    out.append("")
    key["residualised_dt_minus_tt"] = rr

    # (c) Disagreement table: matched-pair centred cosine vs Jaccard (diagonal regression)
    jd, cd = np.diag(J), np.diag(C_dt)
    b1, b0, r_d, _, _ = stats.linregress(jd, cd)
    resid = cd - (b0 + b1 * jd)
    order = np.argsort(resid)
    rank_c = stats.rankdata(-cd).astype(int)
    rank_j = stats.rankdata(-jd).astype(int)
    rows = []
    for i in range(n):
        rows.append([labels[i], FORM[labels[i]], f"{cd[i]:.4f}", int(rank_c[i]), f"{jd[i]:.4f}", int(rank_j[i]),
                     int(rank_j[i] - rank_c[i]), f"{resid[i]:+.4f}"])
    rows.sort(key=lambda r: float(r[-1]))
    with open(OUT / "baseline_disagreements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "form", "centred_cos", "rank_centred", "jaccard", "rank_jaccard",
                    "rank_shift_jaccard_minus_centred", "residual_centred_given_jaccard"])
        w.writerows(rows)
    out.append("Matched pairs: centred cosine regressed on Jaccard (r = %.3f). Pericopes the embedding rates "
               "lower than their word overlap predicts:" % r_d)
    for i in order[:6]:
        out.append(f"  {labels[i]:<30s} centred {cd[i]:.3f} (rank {rank_c[i]}), Jaccard {jd[i]:.3f} (rank {rank_j[i]}), residual {resid[i]:+.3f}")
    out.append("Pericopes the embedding rates higher than their word overlap predicts:")
    for i in order[-6:][::-1]:
        out.append(f"  {labels[i]:<30s} centred {cd[i]:.3f} (rank {rank_c[i]}), Jaccard {jd[i]:.3f} (rank {rank_j[i]}), residual {resid[i]:+.3f}")
    out.append("")
    key["disagreement_lower"] = [dict(label=labels[i], centred=float(cd[i]), rank_centred=int(rank_c[i]),
                                      jaccard=float(jd[i]), rank_jaccard=int(rank_j[i]), residual=float(resid[i]))
                                 for i in order[:6]]
    key["disagreement_higher"] = [dict(label=labels[i], centred=float(cd[i]), rank_centred=int(rank_c[i]),
                                       jaccard=float(jd[i]), rank_jaccard=int(rank_j[i]), residual=float(resid[i]))
                                  for i in order[-6:][::-1]]
    key["diag_regression_r"] = float(r_d)
    big_shift = sorted(range(n), key=lambda i: -abs(rank_j[i] - rank_c[i]))[:8]
    out.append("Largest rank shifts between the two measures:")
    for i in big_shift:
        out.append(f"  {labels[i]:<30s} centred rank {rank_c[i]:2d}, Jaccard rank {rank_j[i]:2d}")
    out.append("")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "key_stats_baselines.json").write_text(json.dumps(_clean(key), indent=2), encoding="utf-8")
    (OUT / "baselines.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
