"""
baselines_lemma.py
==================
Lemma-based lexical baselines (MorphGNT SBLGNT lemmas), run through exactly
the pipeline used for the word-form baselines in ``baselines.py``: floors,
excess for the four sets, bootstrap CIs, Mann-Whitney, label permutation,
four permutation nulls, seed 42.

Measures
--------
* Jaccard (lemmas)  — multiset Jaccard on normalised lemmas (primary);
* TF-IDF (lemmas)   — cosine of word-level TF-IDF vectors of lemmas, fitted on
                      all unit lemma sequences of both traditions.

Residual tests
--------------
* the centred-cosine pair matrix regressed on Jaccard (lemmas) over all
  Matthew-Luke pairs; matched versus mismatched residual, random and
  length-and-theme-matched nulls (the primary test of the reading rule
  recorded in outputs/lemma/READING_RULE.md before this script was run);
* the same in the triple tradition; DT minus TT on the residual;
* multiple regressions on Jaccard (forms) + Jaccard (lemmas), Jaccard
  (forms) + TF-IDF (character 3-5-grams), and all three together;
* DT minus TT on the residual under separate fits per tradition, under the
  double-tradition slope for both, and under one pooled fit;
* the residual after lemma Jaccard under the alternative encoder
  configurations of robustness.py;
* ranks of the paraphrase cases discussed in Section 4.3.

Requires data/raw/morphgnt/ (python download_morphgnt.py).

Outputs (outputs/lemma/)
------------------------
    baselines_lemma.md, lemma_floor_table.csv, lemma_disagreements.csv,
    key_stats_lemma.json
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.analysis_core import (SEED, N_PERM, cos_matrix, floors, per_pericope_excess, compare_excess,  # noqa: E402
                                bootstrap_ci, permutation_tests, fmt_null_block, perm_pvalue)
from stcm.data_loader import SBLGNTLoader, TRIPLE_TRADITION, DOUBLE_TRADITION  # noqa: E402
from stcm.embeddings import EmbeddingPipeline  # noqa: E402
from stcm.metadata import THEMATIC_TAGS, FORM  # noqa: E402
from stcm.morphgnt import load_book, unit_lemmas, MORPHGNT_COMMIT  # noqa: E402
from baselines import jaccard_matrix, clean  # noqa: E402

OUT = _REPO / "outputs" / "lemma"
PARAPHRASE_CASES = ["Conditions of discipleship", "One taken, one left", "Golden rule", "Lord's Prayer"]


def multiset_jaccard(A, B):
    S = np.empty((len(A), len(B)))
    CA = [Counter(a) for a in A]
    CB = [Counter(b) for b in B]
    for i, ca in enumerate(CA):
        for j, cb in enumerate(CB):
            inter = sum((ca & cb).values())
            union = sum((ca | cb).values())
            S[i, j] = inter / union if union else 0.0
    return S


def residual_after(C, predictors):
    """Residual of C after an OLS fit on one or more pair matrices (all pairs)."""
    X = np.column_stack([np.ones(C.size)] + [P.ravel() for P in predictors])
    beta, *_ = np.linalg.lstsq(X, C.ravel(), rcond=None)
    fitted = X @ beta
    r2 = 1 - np.sum((C.ravel() - fitted) ** 2) / np.sum((C.ravel() - C.mean()) ** 2)
    return C - fitted.reshape(C.shape), beta, float(r2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = SBLGNTLoader().load()
    pipe = EmbeddingPipeline()
    double, triple = corpus.double_tradition, corpus.triple_tradition
    labels = [p.label for p in double]
    tags = [THEMATIC_TAGS[l] for l in labels]
    n, nt = len(double), len(triple)
    len_m = np.array([len(p.matthew) for p in double], float)
    len_l = np.array([len(p.luke) for p in double], float)
    tlen_m = np.array([len(p.matthew) for p in triple], float)
    tlen_l = np.array([len(p.luke) for p in triple], float)

    books = {g: load_book(g) for g in ("matthew", "mark", "luke")}
    dref = {l: (m, k) for l, m, k in DOUBLE_TRADITION}
    tref = {l: (m, k, lk) for l, m, k, lk in TRIPLE_TRADITION}
    LEM = {
        "dt": ([unit_lemmas(books["matthew"], dref[l][0]) for l in labels],
               [unit_lemmas(books["luke"], dref[l][1]) for l in labels]),
        "tt": ([unit_lemmas(books["matthew"], tref[p.label][0]) for p in triple],
               [unit_lemmas(books["luke"], tref[p.label][2]) for p in triple]),
        "mk": ([unit_lemmas(books["matthew"], tref[p.label][0]) for p in triple],
               [unit_lemmas(books["mark"], tref[p.label][1]) for p in triple]),
        "lk": ([unit_lemmas(books["luke"], tref[p.label][2]) for p in triple],
               [unit_lemmas(books["mark"], tref[p.label][1]) for p in triple]),
    }
    all_docs = [" ".join(x) for s in ("dt", "tt") for side in LEM[s] for x in side] + \
               [" ".join(x) for x in LEM["mk"][1]]
    tfidf = TfidfVectorizer(analyzer=lambda d: d.split(), sublinear_tf=True).fit(all_docs)

    def tfidf_lemma(A, B):
        a = tfidf.transform([" ".join(x) for x in A])
        b = tfidf.transform([" ".join(x) for x in B])
        return (a @ b.T).toarray()

    measures = {"Jaccard (lemmas)": multiset_jaccard, "TF-IDF (lemmas)": tfidf_lemma}
    SETS = [("dt", "Double tradition, Matt-Luke"), ("tt", "Triple tradition, Matt-Luke"),
            ("mk", "Triple tradition, Matt-Mark"), ("lk", "Triple tradition, Luke-Mark")]
    out = ["# Lemma-based lexical baselines", "",
           f"MorphGNT SBLGNT commit {MORPHGNT_COMMIT}; n_double = {n}, n_triple = {nt}; "
           f"permutations = {N_PERM}; seed = {SEED}. Reading rule: outputs/lemma/READING_RULE.md.", ""]
    key = {"morphgnt_commit": MORPHGNT_COMMIT, "n_double": n, "n_triple": nt}
    rows, Smat = [], {}
    for mname, fn in measures.items():
        out.append(f"## {mname}\n")
        out.append("| Set | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess [95% CI] |")
        out.append("|---|---|---|---|---|---|")
        E = {}
        for s, name in SETS:
            Sm = fn(*LEM[s])
            Smat[(mname, s)] = Sm
            f = floors(Sm)
            e, e_nn = per_pericope_excess(Sm)
            E[s] = (e, e_nn)
            lo, hi = bootstrap_ci(e, SEED + 11)
            lo2, hi2 = bootstrap_ci(e_nn, SEED + 12)
            out.append(f"| {name} | {f['matched']:.4f} | {f['floor']:.4f} | {f['nn']:.4f} | "
                       f"{f['excess']:.4f} [{lo:.4f}, {hi:.4f}] | {f['excess_nn']:.4f} [{lo2:.4f}, {hi2:.4f}] |")
            rows.append([mname, name, f"{f['matched']:.4f}", f"{f['floor']:.4f}", f"{f['nn']:.4f}",
                         f"{f['excess']:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{f['excess_nn']:.4f}", f"{lo2:.4f}", f"{hi2:.4f}"])
            key[f"{mname}_{s}"] = dict(**f, excess_ci=[lo, hi], excess_nn_ci=[lo2, hi2])
        out.append("")
        for a, b in (("dt", "tt"), ("dt", "lk"), ("dt", "mk")):
            r = compare_excess(E[a][0], E[b][0], SEED + 21)
            out.append(f"- {a.upper()} minus {b.upper()} excess: {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                       f"MWU p = {r['p_mwu']:.3g}")
            key[f"{mname}_{a}_minus_{b}"] = r
        nulls = permutation_tests(Smat[(mname, "dt")], len_m, len_l, tags)
        out.append("")
        out.append(fmt_null_block(f"{mname}: permutation tests (double tradition)", nulls))
        for nm in ("random", "thematic", "length", "length_theme"):
            key[f"{mname}_dt_{nm}_null_mean"] = float(nulls[nm].mean())
            key[f"{mname}_dt_{nm}_p"] = perm_pvalue(nulls["observed"], nulls[nm])
    with open(OUT / "lemma_floor_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measure", "set", "matched", "mean_floor", "nn_floor", "excess", "excess_ci_lo", "excess_ci_hi",
                    "excess_nn", "excess_nn_ci_lo", "excess_nn_ci_hi"])
        w.writerows(rows)

    # ------------------------------------------------------------------
    # Embedding residual after lemma overlap
    # ------------------------------------------------------------------
    M = np.array([pipe.embed_parts(p.matthew_parts) for p in double], float)
    L = np.array([pipe.embed_parts(p.luke_parts) for p in double], float)
    TM = np.array([pipe.embed_parts(p.matthew_parts) for p in triple], float)
    TK = np.array([pipe.embed_parts(p.mark_parts) for p in triple], float)
    TL = np.array([pipe.embed_parts(p.luke_parts) for p in triple], float)
    mu = np.vstack([M, L, TM, TK, TL]).mean(axis=0)
    C_dt, C_tt = cos_matrix(M, L, mu), cos_matrix(TM, TL, mu)
    JL_dt, JL_tt = Smat[("Jaccard (lemmas)", "dt")], Smat[("Jaccard (lemmas)", "tt")]
    JF_dt = jaccard_matrix([p.matthew for p in double], [p.luke for p in double])
    JF_tt = jaccard_matrix([p.matthew for p in triple], [p.luke for p in triple])
    texts_all = [clean(t) for p in corpus.all_pericopes for t in (p.matthew, p.mark, p.luke) if t]
    tf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts_all)
    TC = lambda A, B: (tf_char.transform([clean(t) for t in A]) @ tf_char.transform([clean(t) for t in B]).T).toarray()
    TC_dt = TC([p.matthew for p in double], [p.luke for p in double])
    TC_tt = TC([p.matthew for p in triple], [p.luke for p in triple])

    out.append("## Embedding residual after lemma overlap (primary test of the reading rule)\n")
    specs = [("lemma Jaccard", [JL_dt], [JL_tt]),
             ("lemma TF-IDF", [Smat[("TF-IDF (lemmas)", "dt")]], [Smat[("TF-IDF (lemmas)", "tt")]]),
             ("forms Jaccard + lemma Jaccard", [JF_dt, JL_dt], [JF_tt, JL_tt]),
             ("forms Jaccard + char TF-IDF", [JF_dt, TC_dt], [JF_tt, TC_tt]),
             ("forms Jaccard + lemma Jaccard + char TF-IDF", [JF_dt, JL_dt, TC_dt], [JF_tt, JL_tt, TC_tt])]
    key["residual"] = {}
    for k_, (name, P_dt, P_tt) in enumerate(specs):
        R_dt, beta, r2 = residual_after(C_dt, P_dt)
        R_tt, _, r2t = residual_after(C_tt, P_tt)
        f = floors(R_dt)
        nul = permutation_tests(R_dt, len_m, len_l, tags, seed=SEED + 5)
        nul_t = permutation_tests(R_tt, tlen_m, tlen_l, None, seed=SEED + 7)
        e_d, _ = per_pericope_excess(R_dt)
        e_t, _ = per_pericope_excess(R_tt)
        rr = compare_excess(e_d, e_t, SEED + 23)
        rec = dict(r2_all_pairs=r2, beta=[float(b) for b in beta], **f,
                   random_null_mean=float(nul["random"].mean()), random_null_sd=float(nul["random"].std()),
                   random_p=perm_pvalue(nul["observed"], nul["random"]),
                   thematic_p=perm_pvalue(nul["observed"], nul["thematic"]),
                   length_p=perm_pvalue(nul["observed"], nul["length"]),
                   length_theme_null_mean=float(nul["length_theme"].mean()),
                   length_theme_null_sd=float(nul["length_theme"].std()),
                   length_theme_p=perm_pvalue(nul["observed"], nul["length_theme"]),
                   tt_matched=float(np.diag(R_tt).mean()), tt_floor=floors(R_tt)["floor"], tt_r2=r2t,
                   tt_random_p=perm_pvalue(nul_t["observed"], nul_t["random"]),
                   tt_length_p=perm_pvalue(nul_t["observed"], nul_t["length"]),
                   dt_minus_tt=rr)
        key["residual"][name] = rec
        out.append(f"### Residual after {name}\n")
        out.append(f"- R² of the fit over all {n*n} pairs: {r2:.3f}; coefficients {', '.join(f'{b:+.3f}' for b in beta)}")
        out.append(f"- Matched mean {f['matched']:+.4f}; mismatched mean {f['floor']:+.4f}; excess {f['excess']:+.4f}")
        out.append(f"- Random null {rec['random_null_mean']:+.4f} (SD {rec['random_null_sd']:.4f}), p = {rec['random_p']:.3g}; "
                   f"thematic p = {rec['thematic_p']:.3g}; length p = {rec['length_p']:.3g}; "
                   f"length+theme null {rec['length_theme_null_mean']:+.4f} (SD {rec['length_theme_null_sd']:.4f}), "
                   f"p = {rec['length_theme_p']:.3g}")
        out.append(f"- Triple tradition: matched {rec['tt_matched']:+.4f}, mismatched {rec['tt_floor']:+.4f}, "
                   f"random p = {rec['tt_random_p']:.3g}, length p = {rec['tt_length_p']:.3g}")
        out.append(f"- DT minus TT on the residual: {rr['diff']:+.4f} [{rr['ci_lo']:+.4f}, {rr['ci_hi']:+.4f}], "
                   f"MWU p = {rr['p_mwu']:.3g}")
        out.append("")

    # DT minus TT on the residual under separate versus common fits.  With a
    # separate fit per tradition, the steeper triple-tradition slope removes
    # more of its excess; a common slope is the like-for-like comparison.
    out.append("## DT minus TT on the residual: separate versus common fits\n")
    out.append("| Predictor | Slope DT | Slope TT | Slope pooled | Separate fits | DT slope for both | Pooled fit |")
    out.append("|---|---|---|---|---|---|---|")
    key["dt_minus_tt_fit_comparison"] = {}
    for name, Pd, Pt in (("forms Jaccard", JF_dt, JF_tt), ("lemma Jaccard", JL_dt, JL_tt)):
        fd = stats.linregress(Pd.ravel(), C_dt.ravel())
        ft = stats.linregress(Pt.ravel(), C_tt.ravel())
        fp = stats.linregress(np.r_[Pd.ravel(), Pt.ravel()], np.r_[C_dt.ravel(), C_tt.ravel()])
        rec, cells = {"slope_dt": fd.slope, "slope_tt": ft.slope, "slope_pooled": fp.slope}, []
        for lab, (bd, bt) in (("separate", (fd, ft)), ("dt_slope", (fd, fd)), ("pooled", (fp, fp))):
            ed, _ = per_pericope_excess(C_dt - (bd.intercept + bd.slope * Pd))
            et, _ = per_pericope_excess(C_tt - (bt.intercept + bt.slope * Pt))
            r = compare_excess(ed, et, SEED + 23)
            rec[lab] = r
            cells.append(f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        key["dt_minus_tt_fit_comparison"][name] = rec
        out.append(f"| {name} | {fd.slope:.3f} | {ft.slope:.3f} | {fp.slope:.3f} | " + " | ".join(cells) + " |")
    out.append("")

    # The lemma residual under the alternative encoder configurations (the
    # embedding caches are filled by populate_cache.py).
    from populate_cache import CONFIGS
    out.append("## Residual after lemma Jaccard under the alternative encoders\n")
    out.append("| Configuration | Matched | Mismatched | Length+theme null | p |")
    out.append("|---|---|---|---|---|")
    key["lemma_residual_by_encoder"] = {}
    for cname, cfg in CONFIGS.items():
        pe = EmbeddingPipeline(config=cfg)
        Mx = np.array([pe.embed_parts(p.matthew_parts) for p in double], float)
        Lx = np.array([pe.embed_parts(p.luke_parts) for p in double], float)
        mux = np.vstack([Mx, Lx] + [np.array([pe.embed_parts(getattr(p, a)) for p in triple], float)
                                    for a in ("matthew_parts", "mark_parts", "luke_parts")]).mean(axis=0)
        Rx, _, _ = residual_after(cos_matrix(Mx, Lx, mux), [JL_dt])
        fx = floors(Rx)
        nx = permutation_tests(Rx, len_m, len_l, tags, seed=SEED + 5)
        px = perm_pvalue(nx["observed"], nx["length_theme"])
        key["lemma_residual_by_encoder"][cname] = dict(matched=fx["matched"], floor=fx["floor"],
                                                       length_theme_null_mean=float(nx["length_theme"].mean()),
                                                       length_theme_p=px, embedder=pe.embedder_id)
        out.append(f"| {cname} ({pe.embedder_id}) | {fx['matched']:+.4f} | {fx['floor']:+.4f} | "
                   f"{nx['length_theme'].mean():+.4f} | {px:.3g} |")
    out.append("")

    # Paraphrase cases: diagonal regression of centred cosine on lemma Jaccard
    cd, jl, jf = np.diag(C_dt), np.diag(JL_dt), np.diag(JF_dt)
    res_rows, key["cases"] = [], {}
    for meas, x in (("forms", jf), ("lemmas", jl)):
        b1, b0, r_d, _, _ = stats.linregress(x, cd)
        resid = cd - (b0 + b1 * x)
        rank_c = stats.rankdata(-cd).astype(int)
        rank_x = stats.rankdata(-x).astype(int)
        key[f"diag_r_{meas}"] = float(r_d)
        key[f"spearman_centred_vs_{meas}"] = float(stats.spearmanr(cd, x)[0])
        for lab in PARAPHRASE_CASES:
            i = labels.index(lab)
            key["cases"].setdefault(lab, {})[meas] = dict(jaccard=float(x[i]), rank_jaccard=int(rank_x[i]),
                                                          rank_centred=int(rank_c[i]), residual=float(resid[i]))
        if meas == "lemmas":
            order = np.argsort(resid)
            for i in order:
                res_rows.append([labels[i], FORM[labels[i]], f"{cd[i]:.4f}", int(rank_c[i]), f"{jf[i]:.4f}",
                                 f"{x[i]:.4f}", int(rank_x[i]), f"{resid[i]:+.4f}"])
    with open(OUT / "lemma_disagreements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "form", "centred_cos", "rank_centred", "jaccard_forms", "jaccard_lemmas",
                    "rank_jaccard_lemmas", "residual_centred_given_lemma_jaccard"])
        w.writerows(res_rows)
    out.append("## Paraphrase cases (matched pairs; residual of the centred cosine on Jaccard)\n")
    out.append("| Pericope | Jaccard forms (rank) | residual | Jaccard lemmas (rank) | residual | centred rank |")
    out.append("|---|---|---|---|---|---|")
    for lab in PARAPHRASE_CASES:
        f_, l_ = key["cases"][lab]["forms"], key["cases"][lab]["lemmas"]
        out.append(f"| {lab} | {f_['jaccard']:.3f} ({f_['rank_jaccard']}) | {f_['residual']:+.3f} | "
                   f"{l_['jaccard']:.3f} ({l_['rank_jaccard']}) | {l_['residual']:+.3f} | {l_['rank_centred']} |")
    out.append("")
    out.append(f"Spearman, centred cosine vs Jaccard: forms {key['spearman_centred_vs_forms']:.3f}, "
               f"lemmas {key['spearman_centred_vs_lemmas']:.3f}")

    def _clean(o):
        if isinstance(o, dict):
            return {kk: _clean(v) for kk, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "key_stats_lemma.json").write_text(json.dumps(_clean(key), indent=2), encoding="utf-8")
    (OUT / "baselines_lemma.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
