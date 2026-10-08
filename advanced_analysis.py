"""
advanced_analysis.py
====================
All statistics reported in the article, computed from the cached
Koine-Greek-BERT pericope embeddings (no model weights are needed once the
cache is populated; the sentence-level bootstrap in section 6 is the only
step that re-embeds text and therefore needs the model).

Analyses
--------
1. Calibration signatures (Table 1): Matt-Luke cosine in the triple
   tradition (Signature A) and residual correlation against the
   pericope-specific Markan text (Signature B), raw and mean-centred.
2. Anisotropy diagnosis and mean-centring (Ethayarajh 2019; cf. Su et al.
   2021): matched means, mismatched-pair floors, nearest-non-parallel
   floors (size-matched by subsampling), internal homogeneity, centred
   norms, and the sensitivity of the two-tradition comparison to the
   centring vector.
3. Permutation tests on the raw cosine, the centred cosine (primary) and
   the residual correlation, each against four nulls: random pairing,
   thematic pairing, length-matched pairing, and length-and-theme-matched
   pairing.  Matched pairs are similar in length while random pairs are
   not, so the length-matched nulls remove a confound the random null
   leaves in.
4. Genre floor for the residual signature (mismatched-pair residual
   correlation against the Mark centroid).
5. Directionality: ridge regression with intercept (within-set centring)
   in both directions, exact leave-one-out R^2 via the hat-matrix
   identity, the ridge penalty chosen by leave-one-out error, a jackknife
   standard error for the asymmetry delta-R2 (the percentile bootstrap is
   invalid for a leave-one-out statistic because duplicated observations
   enter their own held-out fold), and a sign-flip permutation test.
   Two controls accompany it: a length-balanced subset of the double
   tradition, and the known-direction case of the triple tradition (Mark
   -> Matthew, Mark -> Luke), which shows whether the statistic can detect
   a direction of copying at all.  The residual-norm comparison used in
   earlier versions is reported only as a diagnostic, because the residual
   norm is a near-deterministic function of passage length.
6. Centred-cosine ranking and per-pericope statistics: confound analyses
   (length in characters, literary form, compositional stratum, Goulder
   set, word overlap) and the sentence-level bootstrap.

Outputs
-------
    outputs/reports/advanced_analysis.md
    outputs/reports/centred_cosine_ranking.csv
    outputs/reports/calibration_table.csv
    outputs/reports/key_stats.json
"""
from __future__ import annotations

import csv
import pathlib
import re
import sys

import numpy as np
from scipy import stats

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.data_loader import SBLGNTLoader, DOUBLE_TRADITION, format_ref
from stcm.embeddings import EmbeddingPipeline
from stcm.metadata import (THEMATIC_TAGS, FORM, GOULDER_REDACTION_LABELS,
                           stratum_of, check_coverage)
from stcm.utils import residual_vector, word_tokens

SEED = 42
N_PERM = 1000
N_BOOT = 1000
N_SENT_BOOT = 200
LENGTH_TOL = 0.4          # |log(len_a / len_b)| tolerance for length-matched nulls
ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def perm_pvalue(observed, null):
    b = int(np.sum(np.asarray(null) >= observed))
    return (b + 1) / (len(null) + 1)


def pair_matrix(A, B, pair_stat):
    """S[i, j] = pair_stat(A_i, B_j) for all pairs."""
    n, m = len(A), len(B)
    S = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            S[i, j] = pair_stat(A[i], B[j])
    return S


def word_overlap(text_a: str, text_b: str) -> float:
    from collections import Counter
    ca, cb = Counter(word_tokens(text_a)), Counter(word_tokens(text_b))
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def bootstrap_ci(values, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def candidate_sets(labels, len_m, len_l, thematic=False, length=False):
    """
    For each Matthean pericope i, the indices j != i of admissible Lukan
    partners under the given null.  Length matching admits partners whose
    log length ratio to Matt_i is within LENGTH_TOL; if fewer than three
    qualify, the three closest in length are used.  Thematic matching
    restricts to the same THEMATIC_TAGS group (falling back to all
    pericopes for a singleton group).
    """
    n = len(labels)
    out = []
    for i in range(n):
        cand = [j for j in range(n) if j != i]
        if thematic:
            tag = THEMATIC_TAGS[labels[i]]
            same = [j for j in cand if THEMATIC_TAGS[labels[j]] == tag]
            if same:
                cand = same
        if length:
            ratio = {j: abs(np.log(len_m[i] / len_l[j])) for j in cand}
            close = [j for j in cand if ratio[j] <= LENGTH_TOL]
            if len(close) < 3:
                close = sorted(cand, key=lambda j: ratio[j])[:3]
            cand = close
        out.append(cand)
    return out


def permutation_tests(S, labels, len_m, len_l, seed=SEED):
    """Four nulls for the mean matched-pair statistic from the pair matrix S."""
    n = S.shape[0]
    rows = np.arange(n)
    observed = float(np.mean(np.diag(S)))
    res = {"observed": observed}

    rng = np.random.default_rng(seed)
    null_r = np.array([float(S[rows, rng.permutation(n)].mean()) for _ in range(N_PERM)])
    res["random"] = null_r

    specs = [("thematic", True, False), ("length", False, True), ("length_theme", True, True)]
    for k, (name, th, le) in enumerate(specs, start=1):
        cands = candidate_sets(labels, len_m, len_l, thematic=th, length=le)
        rng_k = np.random.default_rng(seed + 7 * k)
        null = []
        for _ in range(N_PERM):
            js = [rng_k.choice(c) for c in cands]
            null.append(float(S[rows, js].mean()))
        res[name] = np.array(null)
    return res


def fmt_block(title, r):
    lines = [f"## {title}", "", f"- Observed mean        : {r['observed']:.4f}"]
    for name, label in (("random", "Random null"), ("thematic", "Thematic null"),
                        ("length", "Length-matched null"), ("length_theme", "Length+theme null")):
        null = r[name]
        z = (r["observed"] - null.mean()) / (null.std() + 1e-12)
        lines.append(f"- {label:<21s}: mean {null.mean():.4f} (SD {null.std():.4f}); "
                     f"p = {perm_pvalue(r['observed'], null):.4g}; z = {z:.2f}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exact LOO ridge via the hat-matrix identity (with intercept)
# ---------------------------------------------------------------------------

def loo_r2(X, Y, alpha):
    """
    Exact leave-one-out R^2 of kernel ridge regression Y ~ X + intercept.

    The intercept is handled by centring X and Y on their column means,
    which is exact for ridge with an unpenalised intercept when the
    leave-one-out correction is applied to the centred problem with the
    hat matrix augmented by the 1/n intercept term:  H = 1/n + Gc (Gc +
    alpha I)^{-1}, where Gc is the Gram matrix of the centred predictors.
    The LOO residual is e_i = (y_i - yhat_i) / (1 - H_ii).  R^2 is computed
    against the leave-one-out mean of Y.
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T
    H = G @ np.linalg.inv(G + alpha * np.eye(n)) + np.full((n, n), 1.0 / n)
    resid = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
    sse = float(np.sum(resid ** 2))
    Ybar_loo = (Y.sum(axis=0)[None, :] - Y) / (n - 1)
    sst = float(np.sum((Y - Ybar_loo) ** 2))
    return 1.0 - sse / sst


def loo_sse(X, Y, alpha):
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T
    H = G @ np.linalg.inv(G + alpha * np.eye(n)) + np.full((n, n), 1.0 / n)
    resid = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
    return float(np.sum(resid ** 2))


def delta_r2(X, Y, alpha):
    """Predictability asymmetry: R^2(Y|X) - R^2(X|Y)."""
    return loo_r2(X, Y, alpha) - loo_r2(Y, X, alpha)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    corpus = SBLGNTLoader().load()
    pipe = EmbeddingPipeline()

    double = [p for p in corpus.double_tradition if p.matthew and p.luke]
    labels = [p.label for p in double]
    check_coverage(labels)
    M = [np.asarray(pipe.embed_text(p.matthew), dtype=np.float64) for p in double]
    L = [np.asarray(pipe.embed_text(p.luke), dtype=np.float64) for p in double]
    len_m = np.array([len(p.matthew) for p in double], dtype=float)
    len_l = np.array([len(p.luke) for p in double], dtype=float)

    triple = [p for p in corpus.triple_tradition if p.matthew and p.mark and p.luke]
    tlabels = [p.label for p in triple]
    TM = [np.asarray(pipe.embed_text(p.matthew), dtype=np.float64) for p in triple]
    TK = [np.asarray(pipe.embed_text(p.mark), dtype=np.float64) for p in triple]
    TL = [np.asarray(pipe.embed_text(p.luke), dtype=np.float64) for p in triple]
    tlen_m = np.array([len(p.matthew) for p in triple], dtype=float)
    tlen_l = np.array([len(p.luke) for p in triple], dtype=float)

    n, nt = len(M), len(TM)
    centroid = np.mean(TK, axis=0)
    all_vecs = np.array(M + L + TM + TK + TL)
    mu = all_vecs.mean(axis=0)

    def ccos(a, b):
        return cos(a - mu, b - mu)

    out = ["# STCM Advanced Analysis", "",
           f"(n_double = {n}, n_triple = {nt}, corpus vectors = {len(all_vecs)}, "
           f"permutations = {N_PERM}, bootstrap = {N_BOOT}, seed = {SEED})", "",
           "Lengths are in characters of the normalised Greek text. "
           f"Double tradition: Matthew median {np.median(len_m):.0f}, Luke median {np.median(len_l):.0f}; "
           f"triple tradition: Matthew median {np.median(tlen_m):.0f}, Luke median {np.median(tlen_l):.0f}.", ""]

    # ------------------------------------------------------------------
    # 1. Calibration signatures (Table 1)
    # ------------------------------------------------------------------
    siga_raw_v = np.array([cos(TM[i], TL[i]) for i in range(nt)])
    siga_cen_v = np.array([ccos(TM[i], TL[i]) for i in range(nt)])
    sigb_v = np.array([cos(residual_vector(TM[i], TK[i]), residual_vector(TL[i], TK[i]))
                       for i in range(nt)])
    mm_v = np.array([cos(TM[i], TK[i]) for i in range(nt)])
    lk_v = np.array([cos(TL[i], TK[i]) for i in range(nt)])
    out.append("## 1. Calibration signatures (triple tradition)\n")
    out.append("| Signature | Mean | SD | 95% bootstrap CI |")
    out.append("|---|---|---|---|")
    rows = [("A: Matt–Luke cosine (raw)", siga_raw_v, 1),
            ("A: Matt–Luke cosine (centred)", siga_cen_v, 2),
            ("B: residual correlation vs pericope-specific Mark", sigb_v, 3),
            ("Matt–Mark cosine (raw)", mm_v, 4), ("Luke–Mark cosine (raw)", lk_v, 5)]
    cal_rows = []
    for name, v, k in rows:
        lo, hi = bootstrap_ci(v, SEED + k)
        out.append(f"| {name} | {v.mean():.4f} | {v.std(ddof=1):.4f} | [{lo:.4f}, {hi:.4f}] |")
        cal_rows.append([name, f"{v.mean():.4f}", f"{v.std(ddof=1):.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    out.append("")
    with open(_REPO / "outputs" / "reports" / "calibration_table.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["signature", "mean", "sd", "ci_lower", "ci_upper"])
        w.writerows(cal_rows)
    out.append("Triple-tradition Matt–Luke centred cosine, lowest five (alignment check):")
    for i in np.argsort(siga_cen_v)[:5]:
        out.append(f"  {tlabels[i]:<28s} {siga_cen_v[i]:.4f}")
    out.append("")

    # ------------------------------------------------------------------
    # 2. Raw-cosine permutation tests
    # ------------------------------------------------------------------
    S_raw = pair_matrix(M, L, cos)
    raw = permutation_tests(S_raw, labels, len_m, len_l)
    out.append(fmt_block("2. Raw-cosine permutation tests (double tradition)", raw))

    # ------------------------------------------------------------------
    # 3. Anisotropy diagnosis and mean-centring
    # ------------------------------------------------------------------
    S_cen = pair_matrix(M, L, ccos)
    off = ~np.eye(n, dtype=bool)
    St_raw = pair_matrix(TM, TL, cos)
    St_cen = pair_matrix(TM, TL, ccos)
    offt = ~np.eye(nt, dtype=bool)

    def nn_floor(Sm):
        k = Sm.shape[0]
        return float(np.mean([np.delete(Sm[i], i).max() for i in range(k)]))

    def within(V, f):
        k = len(V); Sm = pair_matrix(V, V, f); o = ~np.eye(k, dtype=bool)
        return float(Sm[o].mean())

    table = {}
    for tag, Sd, St in (("raw", S_raw, St_raw), ("cen", S_cen, St_cen)):
        table[tag] = dict(
            tt_matched=float(np.diag(St).mean()), tt_floor=float(St[offt].mean()),
            tt_nn=nn_floor(St), dt_matched=float(np.diag(Sd).mean()),
            dt_floor=float(Sd[off].mean()), dt_nn=nn_floor(Sd))
    out.append("## 3. Anisotropy diagnosis and mean-centred similarities (Table 2)\n")
    out.append("| Quantity | Raw cosine | Mean-centred cosine |")
    out.append("|---|---|---|")
    r, c = table["raw"], table["cen"]
    out.append(f"| Triple tradition: Matt–Luke mean (matched; Signature A) | {r['tt_matched']:.4f} | {c['tt_matched']:.4f} |")
    out.append(f"| Triple tradition: mismatched-pair floor (mean) | {r['tt_floor']:.4f} | {c['tt_floor']:.4f} |")
    out.append(f"| Triple tradition: nearest non-parallel floor | {r['tt_nn']:.4f} | {c['tt_nn']:.4f} |")
    out.append(f"| Triple tradition: excess over mean floor | {r['tt_matched']-r['tt_floor']:.4f} | {c['tt_matched']-c['tt_floor']:.4f} |")
    out.append(f"| Triple tradition: excess over nearest floor | {r['tt_matched']-r['tt_nn']:.4f} | {c['tt_matched']-c['tt_nn']:.4f} |")
    out.append(f"| Double tradition: Matt–Luke mean (matched) | {r['dt_matched']:.4f} | {c['dt_matched']:.4f} |")
    out.append(f"| Double tradition: mismatched-pair floor (mean) | {r['dt_floor']:.4f} | {c['dt_floor']:.4f} |")
    out.append(f"| Double tradition: nearest non-parallel floor | {r['dt_nn']:.4f} | {c['dt_nn']:.4f} |")
    out.append(f"| Double tradition: excess over mean floor | {r['dt_matched']-r['dt_floor']:.4f} | {c['dt_matched']-c['dt_floor']:.4f} |")
    out.append(f"| Double tradition: excess over nearest floor | {r['dt_matched']-r['dt_nn']:.4f} | {c['dt_matched']-c['dt_nn']:.4f} |")
    out.append("")
    # Size-matched nearest-neighbour excess: the larger set is subsampled to
    # the smaller set's size (the NN floor is an order statistic).
    rng_ss = np.random.default_rng(SEED)
    big_is_triple = nt > n
    k_small = min(n, nt)
    for tag, Sm in (("raw", St_raw if big_is_triple else S_raw),
                    ("centred", St_cen if big_is_triple else S_cen)):
        exc = []
        for _ in range(N_BOOT):
            idx = rng_ss.choice(Sm.shape[0], size=k_small, replace=False)
            Ss = Sm[np.ix_(idx, idx)]
            exc.append(float(np.diag(Ss).mean()) - nn_floor(Ss))
        exc = np.array(exc)
        lo, hi = np.percentile(exc, [2.5, 97.5])
        which = "Triple" if big_is_triple else "Double"
        out.append(f"- {which} tradition subsampled to n={k_small}, excess over nearest floor "
                   f"({tag}): mean {exc.mean():+.4f}, SD {exc.std():.4f}, "
                   f"95% of draws [{lo:+.4f}, {hi:+.4f}]")
    out.append("")
    out.append("Internal homogeneity (mean similarity among a single evangelist's own "
               "pericopes within each tradition, raw cosine):")
    out.append(f"- Triple tradition: Matthew {within(TM, cos):.4f} | Luke {within(TL, cos):.4f}")
    out.append(f"- Double tradition: Matthew {within(M, cos):.4f} | Luke {within(L, cos):.4f}")
    out.append("")
    norm_dt = float(np.mean([np.linalg.norm(v - mu) for v in (M + L)]))
    norm_tt = float(np.mean([np.linalg.norm(v - mu) for v in (TM + TL)]))
    norm_mk = float(np.mean([np.linalg.norm(v - mu) for v in TK]))
    out.append(f"- Mean centred norm, double-tradition vectors   : {norm_dt:.4f}")
    out.append(f"- Mean centred norm, triple-tradition Matt+Luke : {norm_tt:.4f}")
    out.append(f"- Mean centred norm, Mark vectors               : {norm_mk:.4f}")
    out.append("")

    # 3c. Sensitivity to the centring vector
    def tradition_stats(A, B, centre):
        k = len(A)
        Sm = pair_matrix(A, B, lambda a, b: cos(a - centre, b - centre))
        offk = ~np.eye(k, dtype=bool)
        return float(np.diag(Sm).mean()), float(Sm[offk].mean())

    mu_dt = np.array(M + L).mean(axis=0)
    mu_tt = np.array(TM + TK + TL).mean(axis=0)
    schemes = [
        (f"All {len(all_vecs)} pericope vectors (used throughout)", mu),
        (f"Double-tradition vectors only ({2*n})", mu_dt),
        (f"Triple-tradition vectors only ({3*nt})", mu_tt),
        (f"Mark vectors only ({nt})", centroid),
        ("Mean of the two tradition means (balanced)", (mu_dt + mu_tt) / 2.0),
        (f"Matthew and Luke vectors only, both traditions ({2*n + 2*nt})",
         np.array(M + L + TM + TL).mean(axis=0)),
    ]
    out.append("### 3c. Sensitivity to the centring vector (Table 4)\n")
    out.append("| Centring vector | DT matched | DT contrast | TT matched | TT contrast |")
    out.append("|---|---|---|---|---|")
    for name, centre in schemes:
        dm, df = tradition_stats(M, L, centre)
        tm, tf = tradition_stats(TM, TL, centre)
        out.append(f"| {name} | {dm:.4f} | {dm - df:.4f} | {tm:.4f} | {tm - tf:.4f} |")
    out.append("")

    # ------------------------------------------------------------------
    # 4. Permutation tests on the centred cosine (primary statistic)
    # ------------------------------------------------------------------
    cen = permutation_tests(S_cen, labels, len_m, len_l)
    out.append(fmt_block("4. Permutation tests on the mean-centred cosine (primary statistic)", cen))
    # Length confound inside the pair matrix: why the length-matched null is needed.
    ratio = np.abs(np.log(len_m[:, None] / len_l[None, :]))
    rho_off, p_off = stats.spearmanr(S_cen[off], ratio[off])
    out.append(f"- Length mismatch of pairs, |log(len_Matt/len_Luke)|: matched pairs mean "
               f"{np.diag(ratio).mean():.3f}, mismatched pairs mean {ratio[off].mean():.3f}; "
               f"Spearman rho between centred cosine and length mismatch over all "
               f"mismatched pairs = {rho_off:.3f} (p = {p_off:.2g}).")
    out.append("")

    # ------------------------------------------------------------------
    # 5. Genre floor for the residual signature
    # ------------------------------------------------------------------
    RM = [residual_vector(m, centroid) for m in M]
    RL = [residual_vector(l, centroid) for l in L]
    S_res = pair_matrix(RM, RL, cos)
    res_matched = float(np.diag(S_res).mean())
    res_floor = float(S_res[off].mean())
    res_floor_sd = float(S_res[off].std())
    res = permutation_tests(S_res, labels, len_m, len_l)
    out.append("## 5. Genre floor for the residual signature\n")
    out.append(f"- Matched-pair mean residual correlation   : {res_matched:.4f}")
    out.append(f"- Mismatched-pair mean (genre floor)       : {res_floor:.4f} (SD {res_floor_sd:.4f})")
    out.append(f"- Matched-minus-floor contrast             : {res_matched - res_floor:.4f}")
    out.append("")
    out.append(fmt_block("5b. Permutation tests on the residual correlation alone", res))

    # ------------------------------------------------------------------
    # 6. Directionality
    # ------------------------------------------------------------------
    Mc = np.array([m - mu for m in M]); Mc /= np.linalg.norm(Mc, axis=1, keepdims=True)
    Lc = np.array([l - mu for l in L]); Lc /= np.linalg.norm(Lc, axis=1, keepdims=True)

    # Ridge penalty chosen by total leave-one-out error over both directions.
    sse = {a: loo_sse(Mc, Lc, a) + loo_sse(Lc, Mc, a) for a in ALPHA_GRID}
    alpha = min(sse, key=sse.get)
    r2_lm = loo_r2(Mc, Lc, alpha)
    r2_ml = loo_r2(Lc, Mc, alpha)
    d_obs = r2_lm - r2_ml

    # Jackknife SE for delta-R2.
    jk = np.array([delta_r2(np.delete(Mc, i, 0), np.delete(Lc, i, 0), alpha) for i in range(n)])
    jk_se = float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2)))
    ci_lo, ci_hi = d_obs - 1.96 * jk_se, d_obs + 1.96 * jk_se

    # Sign-flip permutation test under directional exchangeability.
    rng_s = np.random.default_rng(SEED + 13)
    null_d = []
    for _ in range(N_PERM):
        flip = rng_s.random(n) < 0.5
        Xp = np.where(flip[:, None], Lc, Mc)
        Yp = np.where(flip[:, None], Mc, Lc)
        null_d.append(delta_r2(Xp, Yp, alpha))
    null_d = np.array(null_d)
    p_two = float((np.sum(np.abs(null_d) >= abs(d_obs)) + 1) / (N_PERM + 1))

    # Sign-flip p across the alpha grid (specification sensitivity).
    def signflip_p(X, Y, a, seed):
        rng = np.random.default_rng(seed)
        d0 = delta_r2(X, Y, a)
        null = []
        for _ in range(N_PERM):
            flip = rng.random(X.shape[0]) < 0.5
            null.append(delta_r2(np.where(flip[:, None], Y, X), np.where(flip[:, None], X, Y), a))
        null = np.array(null)
        return d0, float((np.sum(np.abs(null) >= abs(d0)) + 1) / (N_PERM + 1))

    grid_rows = [(a,) + signflip_p(Mc, Lc, a, SEED + 17) for a in ALPHA_GRID]

    # Control 1: length-balanced subset of the double tradition (the
    # longer member of a pair is the better predictor for reasons that have
    # nothing to do with dependence).
    bal = np.abs(np.log(len_m / len_l)) < 0.25
    d_bal, p_bal = signflip_p(Mc[bal], Lc[bal], alpha, SEED + 19)

    # Control 2: known direction.  In the triple tradition Mark is the
    # source of both Matthew and Luke; if predictability asymmetry tracked
    # the direction of copying, delta-R2 for (Mark -> Matthew) and (Mark ->
    # Luke) should be clearly positive.
    def cnorm(V):
        A = np.array([v - mu for v in V]); return A / np.linalg.norm(A, axis=1, keepdims=True)
    Tm, Tk, Tl = cnorm(TM), cnorm(TK), cnorm(TL)
    d_km, p_km = signflip_p(Tk, Tm, alpha, SEED + 23)
    d_kl, p_kl = signflip_p(Tk, Tl, alpha, SEED + 29)
    d_ml, p_ml = signflip_p(Tm, Tl, alpha, SEED + 31)

    # Diagnostic: residual norm is a function of length.
    norm_m = np.array([np.linalg.norm(r) for r in RM])
    norm_l = np.array([np.linalg.norm(r) for r in RL])
    rho_len, p_len = stats.spearmanr(np.r_[norm_m, norm_l], np.r_[len_m, len_l])
    rho_diff, p_diff = stats.spearmanr(norm_m - norm_l, np.log(len_m / len_l))
    w_stat, w_p = stats.wilcoxon(norm_m, norm_l)

    out.append("## 6. Directionality analysis\n")
    out.append(f"- Ridge penalty alpha (min total LOO error) : {alpha}")
    out.append("- LOO error by alpha: " + "; ".join(f"{a}: {sse[a]:.3f}" for a in ALPHA_GRID))
    out.append(f"- Exact LOO ridge R^2, Luke | Matthew      : {r2_lm:.4f}")
    out.append(f"- Exact LOO ridge R^2, Matthew | Luke      : {r2_ml:.4f}")
    out.append(f"- delta-R2 (L|M minus M|L)                 : {d_obs:+.4f}")
    out.append(f"- Jackknife SE / 95% CI                    : {jk_se:.4f} / [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    out.append(f"- Sign-flip permutation p (two-sided)      : {p_two:.4g}")
    out.append("- delta-R2 and sign-flip p across the alpha grid: " +
               "; ".join(f"{a}: {d0:+.4f} (p = {pp:.3f})" for a, d0, pp in grid_rows))
    out.append("")
    out.append("Controls for the directionality statistic:")
    out.append(f"- Length-balanced double-tradition pairs (|log length ratio| < 0.25, n = {int(bal.sum())}): "
               f"delta-R2 = {d_bal:+.4f}, sign-flip p = {p_bal:.3f}")
    out.append(f"- Known direction, triple tradition (n = {nt}): Mark -> Matthew delta-R2 = {d_km:+.4f} "
               f"(p = {p_km:.3f}); Mark -> Luke delta-R2 = {d_kl:+.4f} (p = {p_kl:.3f}); "
               f"Matthew vs Luke (both copying Mark) delta-R2 = {d_ml:+.4f} (p = {p_ml:.3f})")
    out.append("")
    out.append("Residual-norm diagnostic (NOT used as evidence of directionality):")
    out.append(f"- Residual norm, Matthew mean {norm_m.mean():.4f}, Luke mean {norm_l.mean():.4f}; "
               f"Wilcoxon signed-rank W = {w_stat:.1f}, p = {w_p:.3g}")
    out.append(f"- Residual norm vs passage length (pooled): Spearman rho = {rho_len:.3f} (p = {p_len:.2g})")
    out.append(f"- Matt-minus-Luke residual norm vs log length ratio: rho = {rho_diff:.3f} (p = {p_diff:.3g})")
    out.append("")

    # ------------------------------------------------------------------
    # 7. Centred-cosine ranking and per-pericope statistics
    # ------------------------------------------------------------------
    ccos_vals = np.diag(S_cen).copy()
    raw_vals = np.diag(S_raw).copy()
    resid_vals = np.diag(S_res).copy()
    wo = np.array([word_overlap(p.matthew, p.luke) for p in double])
    order = np.argsort(ccos_vals)[::-1]
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    refs = {lab: (format_ref(m), format_ref(l)) for lab, m, l in DOUBLE_TRADITION}

    csv_path = _REPO / "outputs" / "reports" / "centred_cosine_ranking.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "label", "matt_ref", "luke_ref", "stratum", "form",
                    "matt_luke_cos", "centred_cos", "residual_sim",
                    "len_matt_chars", "len_luke_chars", "word_overlap"])
        for i in order:
            w.writerow([rank[i], labels[i], refs[labels[i]][0], refs[labels[i]][1],
                        stratum_of(labels[i]), FORM[labels[i]],
                        f"{raw_vals[i]:.4f}", f"{ccos_vals[i]:.4f}", f"{resid_vals[i]:.4f}",
                        int(len_m[i]), int(len_l[i]), f"{wo[i]:.4f}"])

    out.append("## 7. Centred-cosine ranking (primary ranking statistic)\n")
    out.append(f"- Mean (SD, sample)    : {ccos_vals.mean():.4f} ({ccos_vals.std(ddof=1):.4f})")
    out.append(f"- Range                : [{ccos_vals.min():.4f}, {ccos_vals.max():.4f}]")
    out.append("")
    out.append("Top five:")
    for i in order[:5]:
        out.append(f"  {rank[i]}. {labels[i]:<30s} centred={ccos_vals[i]:.4f} "
                   f"raw={raw_vals[i]:.4f} resid={resid_vals[i]:.4f}")
    out.append("Bottom five:")
    for i in order[-5:]:
        out.append(f"  {rank[i]}. {labels[i]:<30s} centred={ccos_vals[i]:.4f} "
                   f"raw={raw_vals[i]:.4f} resid={resid_vals[i]:.4f}")
    out.append("")

    # Confound analyses on the centred cosine
    lv = (len_m + len_l) / 2.0
    r_p, p_p = stats.pearsonr(lv, ccos_vals)
    r_s, p_s = stats.spearmanr(lv, ccos_vals)
    forms = [FORM[l] for l in labels]
    fgroups = {}
    for f, v in zip(forms, ccos_vals):
        fgroups.setdefault(f, []).append(v)
    big = [np.array(v) for v in fgroups.values() if len(v) >= 3]
    h_f, p_f = stats.kruskal(*big)
    strata = [stratum_of(l) for l in labels]
    q1 = ccos_vals[np.array([s == "Q1" for s in strata])]
    q2 = ccos_vals[np.array([s == "Q2" for s in strata])]
    u_st, p_u = stats.mannwhitneyu(q1, q2, alternative="two-sided")
    h_s, p_h = stats.kruskal(q1, q2)
    gmask = np.array([l in GOULDER_REDACTION_LABELS for l in labels])
    g, ng = ccos_vals[gmask], ccos_vals[~gmask]
    t_g, p_g = stats.ttest_ind(g, ng, equal_var=False)
    pooled = np.sqrt(((len(g) - 1) * g.var(ddof=1) + (len(ng) - 1) * ng.var(ddof=1)) / (n - 2))
    d_g = float((g.mean() - ng.mean()) / (pooled + 1e-12))
    r_w, p_w = stats.pearsonr(ccos_vals, wo)

    out.append("### 7b. Confound analyses on the centred cosine\n")
    out.append(f"- Length (characters): Pearson r = {r_p:.3f} (p = {p_p:.3f}); "
               f"Spearman rho = {r_s:.3f} (p = {p_s:.3f})")
    for f in ["proverbial", "discourse", "parable", "narrative", "liturgical"]:
        if f in fgroups:
            v = np.array(fgroups[f])
            out.append(f"- Form {f:<11s}: mean = {v.mean():.4f}, n = {len(v)}")
    out.append(f"- Form Kruskal-Wallis (n >= 3 groups): H = {h_f:.3f}, p = {p_f:.3f}")
    counts = {k: strata.count(k) for k in ("Q1", "Q2", "Q3", "U")}
    out.append(f"- Strata counts: {counts}")
    out.append(f"- Strata: Q1 mean = {q1.mean():.4f} (n = {len(q1)}), "
               f"Q2 mean = {q2.mean():.4f} (n = {len(q2)})")
    out.append(f"- Strata Mann-Whitney U = {u_st:.1f}, p = {p_u:.3f}; "
               f"Kruskal-Wallis H = {h_s:.3f}, p = {p_h:.3f}")
    out.append(f"- Goulder set: mean = {g.mean():.4f} (n = {len(g)}) vs "
               f"{ng.mean():.4f} (n = {len(ng)}); Welch t = {t_g:.3f}, "
               f"p = {p_g:.4f}, Cohen's d = {d_g:.3f}")
    out.append(f"- Word-overlap Pearson r = {r_w:.3f} (p = {p_w:.4g}); "
               f"unexplained variance ~ {100 * (1 - r_w**2):.0f}%")
    out.append("")

    # Sentence-level bootstrap on the centred cosine (needs the model)
    rng = np.random.default_rng(SEED + 99)
    stds = []
    for i, p in enumerate(double):
        ms = [s.strip() for s in re.split(r'[.;·]+', p.matthew) if s.strip()]
        ls = [s.strip() for s in re.split(r'[.;·]+', p.luke) if s.strip()]
        if len(ms) < 2 or len(ls) < 2:
            stds.append(0.0)
            continue
        vals = []
        for _ in range(N_SENT_BOOT):
            mi = rng.choice(len(ms), size=len(ms), replace=True)
            li = rng.choice(len(ls), size=len(ls), replace=True)
            em = pipe.embed_text(" ".join(ms[k] for k in mi))
            el = pipe.embed_text(" ".join(ls[k] for k in li))
            vals.append(ccos(np.asarray(em, dtype=np.float64),
                             np.asarray(el, dtype=np.float64)))
        stds.append(float(np.std(vals)))
    out.append("### 7c. Sentence-level bootstrap on the centred cosine\n")
    out.append(f"- Mean per-pericope bootstrap SD ({N_SENT_BOOT} resamples): {np.mean(stds):.4f}")
    out.append("")

    # Key statistics for the figure script and for documentation checks.
    import json
    key = {
        "n_double": n, "n_triple": nt,
        "centred_random_null_mean": float(cen["random"].mean()),
        "centred_random_null_sd": float(cen["random"].std()),
        "centred_matched_mean": float(ccos_vals.mean()),
        "centred_dt_floor": c["dt_floor"],
        "residual_genre_floor": res_floor,
        "residual_genre_floor_sd": res_floor_sd,
        "residual_matched_mean": res_matched,
        "sig_b_mean": float(sigb_v.mean()),
        "sentence_bootstrap_sd": float(np.mean(stds)),
        "alpha": alpha, "delta_r2": d_obs, "delta_r2_ci": [ci_lo, ci_hi], "signflip_p": p_two,
        "delta_r2_length_balanced": d_bal, "signflip_p_length_balanced": p_bal,
        "delta_r2_mark_matt": d_km, "signflip_p_mark_matt": p_km,
        "delta_r2_mark_luke": d_kl, "signflip_p_mark_luke": p_kl,
    }
    (_REPO / "outputs" / "reports" / "key_stats.json").write_text(
        json.dumps(key, indent=2), encoding="utf-8")

    rep = _REPO / "outputs" / "reports" / "advanced_analysis.md"
    rep.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nReport: {rep}\nRanking CSV: {csv_path}")


if __name__ == "__main__":
    main()
