"""
stcm/analysis_core.py
=====================
Shared helpers for the article statistics: pair matrices, floors,
per-pericope excess values and their uncertainty, permutation nulls and
the exact leave-one-out ridge used by the directionality analysis.

Everything here operates on numpy arrays only; no model is needed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

SEED = 42
N_PERM = 1000
N_BOOT = 1000
N_BOOT_DIFF = 2000
LENGTH_TOL = 0.4          # |log(len_a / len_b)| tolerance for length-matched nulls


# ---------------------------------------------------------------------------
# basic quantities
# ---------------------------------------------------------------------------

def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cos_matrix(A, B, centre=None) -> np.ndarray:
    """S[i, j] = cos(A_i - c, B_j - c) for all pairs (c = 0 if centre is None)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if centre is not None:
        A = A - centre
        B = B - centre
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def pair_matrix(A, B, pair_stat: Callable) -> np.ndarray:
    n, m = len(A), len(B)
    S = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            S[i, j] = pair_stat(A[i], B[j])
    return S


def perm_pvalue(observed: float, null) -> float:
    b = int(np.sum(np.asarray(null) >= observed))
    return (b + 1) / (len(null) + 1)


def bootstrap_ci(values, seed: int, n_boot: int = N_BOOT) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


# ---------------------------------------------------------------------------
# floors and per-pericope excess
# ---------------------------------------------------------------------------

def floors(S: np.ndarray) -> Dict[str, float]:
    """Matched mean, mean mismatched floor, nearest-non-parallel floor, excesses."""
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    matched = float(np.diag(S).mean())
    floor = float(S[off].mean())
    nn = float(np.mean([S[i, off[i]].max() for i in range(n)]))
    return dict(matched=matched, floor=floor, nn=nn,
                excess=matched - floor, excess_nn=matched - nn)


def per_pericope_excess(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    e_i   = S[i,i] - mean_{j != i} S[i,j]   (excess over the pericope's own mean floor)
    e_i^N = S[i,i] - max_{j != i} S[i,j]    (excess over its nearest non-parallel partner)
    """
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    d = np.diag(S)
    e = d - np.array([S[i, off[i]].mean() for i in range(n)])
    e_nn = d - np.array([S[i, off[i]].max() for i in range(n)])
    return e, e_nn


def compare_excess(e_a, e_b, seed: int, n_boot: int = N_BOOT_DIFF) -> Dict[str, float]:
    """
    Uncertainty for the difference of mean per-pericope excess between two
    sets of pericopes (a minus b): bootstrap-over-pericopes CI, Mann-Whitney
    U (two-sided) and a label-permutation test.
    """
    rng = np.random.default_rng(seed)
    e_a, e_b = np.asarray(e_a), np.asarray(e_b)
    diffs = np.array([rng.choice(e_a, len(e_a)).mean() - rng.choice(e_b, len(e_b)).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    u, p_u = stats.mannwhitneyu(e_a, e_b, alternative="two-sided")
    pooled = np.r_[e_a, e_b]
    obs = e_a.mean() - e_b.mean()
    null = []
    for _ in range(n_boot):
        perm = rng.permutation(pooled)
        null.append(perm[:len(e_a)].mean() - perm[len(e_a):].mean())
    null = np.array(null)
    p_perm = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_boot + 1))
    return dict(diff=float(obs), ci_lo=float(lo), ci_hi=float(hi),
                mwu=float(u), p_mwu=float(p_u), p_perm=p_perm,
                mean_a=float(e_a.mean()), mean_b=float(e_b.mean()),
                ci_a=bootstrap_ci(e_a, seed + 1), ci_b=bootstrap_ci(e_b, seed + 2))


def size_matched_nn_excess(S_big: np.ndarray, k_small: int, seed: int,
                           n_draw: int = N_BOOT) -> np.ndarray:
    """Excess over the nearest-neighbour floor with the larger set subsampled."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draw):
        idx = rng.choice(S_big.shape[0], size=k_small, replace=False)
        Ss = S_big[np.ix_(idx, idx)]
        out.append(floors(Ss)["excess_nn"])
    return np.array(out)


# ---------------------------------------------------------------------------
# permutation nulls
# ---------------------------------------------------------------------------

def candidate_sets(n: int, tags: Optional[Sequence[str]], len_a, len_b,
                   thematic: bool = False, length: bool = False) -> List[List[int]]:
    """
    For each left-hand pericope i, the indices j != i of admissible partners.
    Length matching admits partners whose log length ratio to i is within
    LENGTH_TOL; if fewer than three qualify, the three closest in length are
    used.  Thematic matching restricts to the same tag (falling back to all
    pericopes for a singleton group).
    """
    out = []
    for i in range(n):
        cand = [j for j in range(n) if j != i]
        if thematic and tags is not None:
            same = [j for j in cand if tags[j] == tags[i]]
            if same:
                cand = same
        if length:
            ratio = {j: abs(np.log(len_a[i] / len_b[j])) for j in cand}
            close = [j for j in cand if ratio[j] <= LENGTH_TOL]
            if len(close) < 3:
                close = sorted(cand, key=lambda j: ratio[j])[:3]
            cand = close
        out.append(cand)
    return out


def permutation_tests(S: np.ndarray, len_a, len_b, tags: Optional[Sequence[str]] = None,
                      seed: int = SEED, n_perm: int = N_PERM) -> Dict[str, np.ndarray]:
    """
    Nulls for the mean matched-pair statistic from the pair matrix S:
    random pairing, thematic pairing (if tags given), length-matched pairing,
    and length-and-theme-matched pairing (if tags given).
    """
    n = S.shape[0]
    rows = np.arange(n)
    res: Dict[str, np.ndarray] = {"observed": float(np.mean(np.diag(S)))}
    rng = np.random.default_rng(seed)
    res["random"] = np.array([float(S[rows, rng.permutation(n)].mean()) for _ in range(n_perm)])
    specs = [("length", False, True)]
    if tags is not None:
        specs = [("thematic", True, False), ("length", False, True), ("length_theme", True, True)]
    for k, (name, th, le) in enumerate(specs, start=1):
        cands = candidate_sets(n, tags, len_a, len_b, thematic=th, length=le)
        rng_k = np.random.default_rng(seed + 7 * k)
        null = []
        for _ in range(n_perm):
            js = [rng_k.choice(c) for c in cands]
            null.append(float(S[rows, js].mean()))
        res[name] = np.array(null)
    return res


def fmt_null_block(title: str, r: Dict[str, np.ndarray]) -> str:
    lines = [f"## {title}", "", f"- Observed mean        : {r['observed']:.4f}"]
    for name, label in (("random", "Random null"), ("thematic", "Thematic null"),
                        ("length", "Length-matched null"), ("length_theme", "Length+theme null")):
        if name not in r:
            continue
        null = r[name]
        lines.append(f"- {label:<21s}: mean {null.mean():.4f} (SD {null.std():.4f}); "
                     f"p = {perm_pvalue(r['observed'], null):.4g}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# exact leave-one-out ridge (kernel form, with intercept)
# ---------------------------------------------------------------------------

def _hat(X: np.ndarray, alpha: float) -> np.ndarray:
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T
    return G @ np.linalg.inv(G + alpha * np.eye(n)) + np.full((n, n), 1.0 / n)


def loo_r2(X: np.ndarray, Y: np.ndarray, alpha: float) -> float:
    """
    Exact leave-one-out R^2 of kernel ridge regression Y ~ X + intercept.
    The LOO residual is e_i = (y_i - yhat_i) / (1 - H_ii); R^2 is computed
    against the leave-one-out mean of Y.
    """
    n = X.shape[0]
    H = _hat(X, alpha)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    resid = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
    sse = float(np.sum(resid ** 2))
    Ybar_loo = (Y.sum(axis=0)[None, :] - Y) / (n - 1)
    sst = float(np.sum((Y - Ybar_loo) ** 2))
    return 1.0 - sse / sst


def loo_sse(X: np.ndarray, Y: np.ndarray, alpha: float) -> float:
    H = _hat(X, alpha)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    resid = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
    return float(np.sum(resid ** 2))


def effective_df(X: np.ndarray, alpha: float) -> float:
    """Trace of the hat matrix (effective degrees of freedom of the ridge fit)."""
    return float(np.trace(_hat(X, alpha)))


def delta_r2(X: np.ndarray, Y: np.ndarray, alpha: float) -> float:
    """Predictability asymmetry: R^2(Y|X) - R^2(X|Y)."""
    return loo_r2(X, Y, alpha) - loo_r2(Y, X, alpha)


def signflip_p(X: np.ndarray, Y: np.ndarray, alpha: float, seed: int,
               n_perm: int = N_PERM) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    d0 = delta_r2(X, Y, alpha)
    null = []
    for _ in range(n_perm):
        flip = rng.random(X.shape[0]) < 0.5
        null.append(delta_r2(np.where(flip[:, None], Y, X), np.where(flip[:, None], X, Y), alpha))
    null = np.array(null)
    return d0, float((np.sum(np.abs(null) >= abs(d0)) + 1) / (n_perm + 1))


# ---------------------------------------------------------------------------
# sentence resampling (shared by the bootstrap and the cache population)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    import re
    return [s.strip() for s in re.split(r"[.;·]+", text) if s.strip()]
