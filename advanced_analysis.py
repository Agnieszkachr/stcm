"""
advanced_analysis.py
====================
Anisotropy-corrected similarity analyses, genre-floor quantification,
directionality inference, and the centred-cosine ranking for STCM.

All computations use the cached Koine-Greek-BERT pericope embeddings;
no model weights are required when the cache is populated.

Analyses
--------
1. Raw-cosine permutation tests (random + thematic nulls): weight-free
   significance tests of the Matt-Luke pairing signal.
2. Anisotropy diagnosis and mean-centring correction (Ethayarajh 2019;
   cf. Su et al. 2021): all similarity analyses are repeated after
   subtracting the corpus mean vector, and permutation tests are run on
   the mean-centred cosine, which serves as the primary ranking
   statistic throughout.
3. Genre-floor quantification for the residual signature: the mean
   residual correlation among MISMATCHED double-tradition pairs measures
   the shared discourse-vs-narrative displacement; matched pairs are
   compared against this floor.
4. Directionality analysis with formal uncertainty quantification:
   exact leave-one-out ridge regression (closed-form hat-matrix LOO),
   a percentile bootstrap confidence interval for the predictability
   asymmetry delta-R2, and a sign-flip permutation test of the null
   hypothesis of directional exchangeability.
5. Centred-cosine ranking and per-pericope statistics: confound
   analyses (length, literary form, compositional stratum, Goulder set,
   word overlap) and the sentence-level bootstrap, all computed on the
   mean-centred cosine.

Outputs
-------
    outputs/reports/advanced_analysis.md
    outputs/reports/centred_cosine_ranking.csv
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

from stcm.data_loader import SBLGNTLoader, DOUBLE_TRADITION
from stcm.embeddings import EmbeddingPipeline
from stcm.evaluation import THEMATIC_TAGS, GOULDER_REDACTION_LABELS, EvaluationEngine
from stcm.utils import cosine_similarity, residual_vector, load_pickle

SEED = 42
N_PERM = 1000
N_BOOT = 1000
RIDGE_ALPHA = 1.0

# Kloppenborg (1987, 100-101) strata; all unlisted pericopes are Q2.
# Pericope-level approximation of Kloppenborg's verse-level stratigraphy
# (Formation of Q, 1987). The formative stratum includes the discipleship
# speech Q 13:24; 14:26-27, 34-35; 17:33, hence Narrow gate, Conditions of
# discipleship and Salt of the earth are Q1. Pericopes straddling strata
# (e.g. Mission discourse with the Q2 woes 10:12-15) are assigned by their
# dominant component.
STRATUM = {
    "Anxieties about life": "Q1", "Lord's Prayer": "Q1",
    "Fear of God not men": "Q1", "Mission discourse": "Q1",
    "Beatitudes": "Q1", "Mustard seed and leaven": "Q1",
    "Harvest plentiful": "Q1", "Love of enemies": "Q1",
    "Narrow gate": "Q1", "Conditions of discipleship": "Q1",
    "Salt of the earth": "Q1",
    "Temptation narrative (full)": "Q3",
    "Serving two masters": "U", "Lost sheep": "U",
}

# Form-critical classification (matches generate_evaluation_figures.py).
FORM = {
    "John's preaching": "discourse", "Temptation narrative (full)": "narrative",
    "Beatitudes": "discourse", "Love of enemies": "discourse",
    "Lord's Prayer": "liturgical", "Anxieties about life": "discourse",
    "Narrow gate": "proverbial", "Centurion's servant": "narrative",
    "John's question from prison": "narrative", "Jesus on John": "discourse",
    "Woes on Galilean cities": "discourse", "Hidden from wise revealed": "proverbial",
    "Mission discourse": "discourse", "Harvest plentiful": "proverbial",
    "Sign of Jonah": "discourse", "Return of unclean spirit": "discourse",
    "Lamp of the body": "proverbial", "Leaven of Pharisees": "discourse",
    "Fear of God not men": "discourse", "Blasphemy Holy Spirit": "proverbial",
    "Thief in the night": "proverbial", "Faithful servant": "parable",
    "Not peace but sword": "proverbial", "Reading the signs": "proverbial",
    "Settling with opponent": "proverbial", "Mustard seed and leaven": "parable",
    "Many come from east west": "proverbial", "Lament over Jerusalem": "discourse",
    "Parable of Great Banquet": "parable", "Conditions of discipleship": "proverbial",
    "Salt of the earth": "proverbial", "Lost sheep": "parable",
    "Serving two masters": "proverbial", "Day of the Son of Man": "discourse",
    "Talents / Minas": "parable", "Judging twelve tribes": "proverbial",
}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def perm_pvalue(observed, null):
    b = int(np.sum(np.asarray(null) >= observed))
    return (b + 1) / (len(null) + 1)


def pair_matrix(matt_vecs, luke_vecs, pair_stat):
    """Precompute S[i, j] = pair_stat(matt_i, luke_j) for all pairs."""
    n = len(matt_vecs)
    S = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = pair_stat(matt_vecs[i], luke_vecs[j])
    return S


def permutation_tests(S, labels, seed=SEED):
    """Random and thematic permutation tests for the mean pair statistic,
    computed from the precomputed pair matrix S."""
    n = S.shape[0]
    rows = np.arange(n)
    observed = float(np.mean(np.diag(S)))

    rng = np.random.default_rng(seed)
    null_r = np.array([float(S[rows, rng.permutation(n)].mean())
                       for _ in range(N_PERM)])

    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(THEMATIC_TAGS.get(lab, "other"), []).append(i)
    rng_t = np.random.default_rng(seed + 7)
    null_t = []
    for _ in range(N_PERM):
        vals = []
        for i, lab in enumerate(labels):
            cand = [k for k in groups[THEMATIC_TAGS.get(lab, "other")] if k != i]
            if not cand:
                cand = [k for k in range(n) if k != i]
            j = rng_t.choice(cand)
            vals.append(S[i, j])
        null_t.append(float(np.mean(vals)))
    null_t = np.array(null_t)

    return {
        "observed": observed,
        "random_null_mean": float(null_r.mean()),
        "random_null_std": float(null_r.std()),
        "random_p": perm_pvalue(observed, null_r),
        "random_z": float((observed - null_r.mean()) / (null_r.std() + 1e-12)),
        "thematic_null_mean": float(null_t.mean()),
        "thematic_null_std": float(null_t.std()),
        "thematic_p": perm_pvalue(observed, null_t),
        "thematic_z": float((observed - null_t.mean()) / (null_t.std() + 1e-12)),
    }


def fmt_block(title, r):
    return (f"## {title}\n\n"
            f"- Observed mean        : {r['observed']:.4f}\n"
            f"- Random null mean     : {r['random_null_mean']:.4f} (SD {r['random_null_std']:.4f})\n"
            f"- Random p / z         : {r['random_p']:.4g} / {r['random_z']:.2f}\n"
            f"- Thematic null mean   : {r['thematic_null_mean']:.4f} (SD {r['thematic_null_std']:.4f})\n"
            f"- Thematic p / z       : {r['thematic_p']:.4g} / {r['thematic_z']:.2f}\n")


# ---------------------------------------------------------------------------
# Exact LOO ridge via the hat-matrix identity
# ---------------------------------------------------------------------------

def loo_r2(X, Y, alpha=RIDGE_ALPHA):
    """
    Exact leave-one-out R^2 for kernel ridge regression Y ~ X, using the
    closed-form LOO identity  e_i = (y_i - yhat_i) / (1 - H_ii)  with
    H = G (G + alpha I)^{-1},  G = X X^T.  Equivalent to refitting the
    ridge n times with one observation held out, but in closed form.
    """
    n = X.shape[0]
    G = X @ X.T
    H = G @ np.linalg.inv(G + alpha * np.eye(n))
    resid = (Y - H @ Y) / (1.0 - np.diag(H))[:, None]
    sse = float(np.sum(resid ** 2))
    # SST against the leave-one-out mean of Y
    Ybar_loo = (Y.sum(axis=0)[None, :] - Y) / (n - 1)
    sst = float(np.sum((Y - Ybar_loo) ** 2))
    return 1.0 - sse / sst


def delta_r2(X, Y, alpha=RIDGE_ALPHA):
    """Predictability asymmetry: R^2(Y|X) - R^2(X|Y)."""
    return loo_r2(X, Y, alpha) - loo_r2(Y, X, alpha)


def main():
    corpus = SBLGNTLoader().load()
    pipe = EmbeddingPipeline()
    cal = load_pickle(_REPO / "outputs" / "models" / "calibration_signatures.pkl")
    centroid = np.asarray(cal.centroid_mark, dtype=np.float64)

    double = [p for p in corpus.double_tradition if p.matthew and p.luke]
    labels = [p.label for p in double]
    M = [np.asarray(pipe.embed_text(p.matthew), dtype=np.float64) for p in double]
    L = [np.asarray(pipe.embed_text(p.luke), dtype=np.float64) for p in double]

    triple = [p for p in corpus.triple_tradition if p.matthew and p.mark and p.luke]
    TM = [np.asarray(pipe.embed_text(p.matthew), dtype=np.float64) for p in triple]
    TK = [np.asarray(pipe.embed_text(p.mark), dtype=np.float64) for p in triple]
    TL = [np.asarray(pipe.embed_text(p.luke), dtype=np.float64) for p in triple]

    out = ["# STCM Advanced Analysis",
           "",
           f"(n_double = {len(double)}, n_triple = {len(triple)}, "
           f"permutations = {N_PERM}, bootstrap = {N_BOOT}, seed = {SEED})", ""]

    # ------------------------------------------------------------------
    # 1. Raw-cosine permutation tests (uncorrected space)
    # ------------------------------------------------------------------
    S_raw = pair_matrix(M, L, cos)
    raw = permutation_tests(S_raw, labels)
    out.append(fmt_block("1. Raw-cosine permutation tests", raw))

    # ------------------------------------------------------------------
    # 2. Anisotropy diagnosis and mean-centring correction
    # ------------------------------------------------------------------
    all_vecs = np.array(M + L + TM + TK + TL)
    mu = all_vecs.mean(axis=0)

    def ccos(a, b):
        return cos(a - mu, b - mu)

    n = len(M)
    S_cen = pair_matrix(M, L, ccos)
    off = ~np.eye(n, dtype=bool)
    floor_raw = float(S_raw[off].mean())
    floor_cen = float(S_cen[off].mean())
    siga_raw = float(np.mean([cos(TM[i], TL[i]) for i in range(len(triple))]))
    siga_cen = float(np.mean([ccos(TM[i], TL[i]) for i in range(len(triple))]))
    dt_raw = float(np.diag(S_raw).mean())
    dt_cen = float(np.diag(S_cen).mean())

    out.append("## 2. Anisotropy diagnosis and mean-centred similarities\n")
    out.append("| Quantity | Raw cosine | Mean-centred cosine |")
    out.append("|---|---|---|")
    out.append(f"| Triple-tradition Matt-Luke mean (Sig-A) | {siga_raw:.4f} | {siga_cen:.4f} |")
    out.append(f"| Double-tradition Matt-Luke mean | {dt_raw:.4f} | {dt_cen:.4f} |")
    out.append(f"| Mismatched double-tradition pairs (floor) | {floor_raw:.4f} | {floor_cen:.4f} |")
    out.append(f"| Matched-minus-floor contrast | {dt_raw - floor_raw:.4f} | {dt_cen - floor_cen:.4f} |")
    out.append("")

    cen = permutation_tests(S_cen, labels)
    out.append(fmt_block("2b. Permutation tests on mean-centred cosine (primary statistic)", cen))

    # ------------------------------------------------------------------
    # 3. Genre floor for the residual signature
    # ------------------------------------------------------------------
    RM = [residual_vector(m, centroid) for m in M]
    RL = [residual_vector(l, centroid) for l in L]
    S_res = pair_matrix(RM, RL, cos)
    res_matched = float(np.diag(S_res).mean())
    res_floor = float(S_res[off].mean())
    res_floor_sd = float(S_res[off].std())

    res = permutation_tests(S_res, labels)
    out.append("## 3. Genre floor for the residual signature\n")
    out.append(f"- Matched-pair mean residual correlation   : {res_matched:.4f}")
    out.append(f"- Mismatched-pair mean (genre floor)       : {res_floor:.4f} (SD {res_floor_sd:.4f})")
    out.append(f"- Matched-minus-floor contrast             : {res_matched - res_floor:.4f}")
    out.append("")
    out.append(fmt_block("3b. Permutation tests on residual correlation alone", res))

    # ------------------------------------------------------------------
    # 4. Directionality with formal uncertainty quantification
    # ------------------------------------------------------------------
    norm_m = np.array([np.linalg.norm(r) for r in RM])
    norm_l = np.array([np.linalg.norm(r) for r in RL])
    w_stat, w_p = stats.wilcoxon(norm_m, norm_l)

    Mc = np.array([m - mu for m in M]); Mc /= np.linalg.norm(Mc, axis=1, keepdims=True)
    Lc = np.array([l - mu for l in L]); Lc /= np.linalg.norm(Lc, axis=1, keepdims=True)

    r2_lm = loo_r2(Mc, Lc)
    r2_ml = loo_r2(Lc, Mc)
    d_obs = r2_lm - r2_ml

    # Percentile bootstrap CI for delta-R2 (resampling pericopes).
    rng_b = np.random.default_rng(SEED + 11)
    boot = []
    for _ in range(N_BOOT):
        idx = rng_b.integers(0, n, size=n)
        boot.append(delta_r2(Mc[idx], Lc[idx]))
    boot = np.array(boot)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    # Sign-flip permutation test: under H0 of directional exchangeability,
    # swapping (Matt_i, Luke_i) -> (Luke_i, Matt_i) for a random subset of
    # pericopes leaves the joint distribution invariant.
    rng_s = np.random.default_rng(SEED + 13)
    null_d = []
    for _ in range(N_PERM):
        flip = rng_s.random(n) < 0.5
        Xp = np.where(flip[:, None], Lc, Mc)
        Yp = np.where(flip[:, None], Mc, Lc)
        null_d.append(delta_r2(Xp, Yp))
    null_d = np.array(null_d)
    p_two = float((np.sum(np.abs(null_d) >= abs(d_obs)) + 1) / (N_PERM + 1))

    out.append("## 4. Directionality analysis\n")
    out.append(f"- Residual norm, Matthew mean              : {norm_m.mean():.4f}")
    out.append(f"- Residual norm, Luke mean                 : {norm_l.mean():.4f}")
    out.append(f"- Wilcoxon signed-rank (paired)            : W = {w_stat:.1f}, p = {w_p:.4g}")
    out.append(f"- Exact LOO ridge R^2, Luke | Matthew      : {r2_lm:.4f}")
    out.append(f"- Exact LOO ridge R^2, Matthew | Luke      : {r2_ml:.4f}")
    out.append(f"- delta-R2 (L|M minus M|L)                 : {d_obs:+.4f}")
    out.append(f"- 95% percentile bootstrap CI              : [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    out.append(f"- Sign-flip permutation p (two-sided)      : {p_two:.4g}")
    out.append("")

    # ------------------------------------------------------------------
    # 5. Centred-cosine ranking and per-pericope statistics
    # ------------------------------------------------------------------
    ccos_vals = np.diag(S_cen).copy()
    raw_vals = np.diag(S_raw).copy()
    resid_vals = np.diag(S_res).copy()
    order = np.argsort(ccos_vals)[::-1]
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    csv_path = _REPO / "outputs" / "reports" / "centred_cosine_ranking.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "label", "matt_luke_cos", "centred_cos", "residual_sim"])
        for i in order:
            w.writerow([rank[i], labels[i], f"{raw_vals[i]:.4f}",
                        f"{ccos_vals[i]:.4f}", f"{resid_vals[i]:.4f}"])

    out.append("## 5. Centred-cosine ranking (primary ranking statistic)\n")
    out.append(f"- Mean (SD, sample)    : {ccos_vals.mean():.4f} ({ccos_vals.std(ddof=1):.4f})")
    out.append(f"- Range                : [{ccos_vals.min():.4f}, {ccos_vals.max():.4f}]")
    out.append("")
    out.append("Top five:")
    for i in order[:5]:
        out.append(f"  {rank[i]}. {labels[i]:<30s} centred={ccos_vals[i]:.4f} "
                   f"raw={raw_vals[i]:.4f} resid={resid_vals[i]:.4f}")
    out.append("Bottom three:")
    for i in order[-3:]:
        out.append(f"  {rank[i]}. {labels[i]:<30s} centred={ccos_vals[i]:.4f} "
                   f"raw={raw_vals[i]:.4f} resid={resid_vals[i]:.4f}")
    out.append("")

    # Confound analyses on the centred cosine -------------------------------
    lengths = {}
    for lab, m_ref, l_ref in DOUBLE_TRADITION:
        if m_ref and l_ref:
            lengths[lab] = ((m_ref[2] - m_ref[1] + 1) + (l_ref[2] - l_ref[1] + 1)) / 2.0
    lv = np.array([lengths[l] for l in labels])
    r_p, p_p = stats.pearsonr(lv, ccos_vals)
    r_s, p_s = stats.spearmanr(lv, ccos_vals)

    forms = [FORM.get(l, "proverbial") for l in labels]
    fgroups = {}
    for f, v in zip(forms, ccos_vals):
        fgroups.setdefault(f, []).append(v)
    big = [np.array(v) for v in fgroups.values() if len(v) >= 3]
    h_f, p_f = stats.kruskal(*big)

    strata = [STRATUM.get(l, "Q2") for l in labels]
    q1 = ccos_vals[np.array([s == "Q1" for s in strata])]
    q2 = ccos_vals[np.array([s == "Q2" for s in strata])]
    u_st, p_u = stats.mannwhitneyu(q1, q2, alternative="two-sided")
    h_s, p_h = stats.kruskal(q1, q2)

    gmask = np.array([l in GOULDER_REDACTION_LABELS for l in labels])
    g, ng = ccos_vals[gmask], ccos_vals[~gmask]
    t_g, p_g = stats.ttest_ind(g, ng, equal_var=False)
    pooled = np.sqrt((g.var() * len(g) + ng.var() * len(ng)) / n)
    d_g = float((g.mean() - ng.mean()) / (pooled + 1e-12))

    wo = np.array([EvaluationEngine._word_overlap(p.matthew, p.luke) for p in double])
    r_w, p_w = stats.pearsonr(ccos_vals, wo)

    out.append("### 5b. Confound analyses on the centred cosine\n")
    out.append(f"- Length: Pearson r = {r_p:.3f} (p = {p_p:.3f}); "
               f"Spearman rho = {r_s:.3f} (p = {p_s:.3f})")
    for f in ["proverbial", "discourse", "parable", "narrative", "liturgical"]:
        if f in fgroups:
            v = np.array(fgroups[f])
            out.append(f"- Form {f:<11s}: mean = {v.mean():.4f}, n = {len(v)}")
    out.append(f"- Form Kruskal-Wallis (n >= 3 groups): H = {h_f:.3f}, p = {p_f:.3f}")
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

    # Sentence-level bootstrap on the centred cosine ------------------------
    rng = np.random.default_rng(SEED + 99)
    stds = []
    for i, p in enumerate(double):
        ms = [s.strip() for s in re.split(r'[.;·]+', p.matthew) if s.strip()]
        ls = [s.strip() for s in re.split(r'[.;·]+', p.luke) if s.strip()]
        if len(ms) < 2 or len(ls) < 2:
            stds.append(0.0)
            continue
        vals = []
        for _ in range(200):
            mi = rng.choice(len(ms), size=len(ms), replace=True)
            li = rng.choice(len(ls), size=len(ls), replace=True)
            em = pipe.embed_text(" ".join(ms[k] for k in mi))
            el = pipe.embed_text(" ".join(ls[k] for k in li))
            vals.append(ccos(np.asarray(em, dtype=np.float64),
                             np.asarray(el, dtype=np.float64)))
        stds.append(float(np.std(vals)))
    out.append("### 5c. Sentence-level bootstrap on the centred cosine\n")
    out.append(f"- Mean per-pericope bootstrap SD (200 resamples): {np.mean(stds):.4f}")
    out.append("")

    rep = _REPO / "outputs" / "reports" / "advanced_analysis.md"
    rep.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nReport: {rep}\nRanking CSV: {csv_path}")


if __name__ == "__main__":
    main()
