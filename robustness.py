"""
robustness.py
=============
The central contrast under alternative encoder configurations (the cache
for each is filled by ``populate_cache.py``):

* primary — Koine-Greek-BERT, last layer (the configuration used throughout);
* last4   — Koine-Greek-BERT, mean of the last four hidden layers;
* agb     — Ancient-Greek-BERT, the base model (its pre-training corpus
            includes PROIEL and therefore the Greek New Testament);
* greekbert — Greek BERT (Modern Greek), a model without biblical
            pre-training.

For each configuration the residual of the centred cosine after a linear
fit on word overlap (multiset Jaccard, over all Matthew-Luke pairs) is
also tested: matched minus mismatched residual, random-pairing null, and
the double-minus-triple contrast on the residual.

For each configuration: matched mean, floors and excess for the double
tradition (Matt-Luke) and the triple tradition (Matt-Luke, Luke-Mark,
Matt-Mark), raw and centred; the DT-minus-TT and DT-minus-Luke-Mark
contrasts with bootstrap CIs; rank agreement of the double-tradition
ranking with the primary configuration; the top and bottom pericopes.

Outputs
-------
    outputs/reports/robustness.md
    outputs/reports/robustness_table.csv
    outputs/reports/key_stats_robustness.json
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

from stcm.analysis_core import (SEED, cos_matrix, floors, per_pericope_excess, compare_excess, bootstrap_ci,
                                permutation_tests, perm_pvalue)
from scipy import stats as _st
from baselines import jaccard_matrix
from stcm.data_loader import SBLGNTLoader
from stcm.embeddings import EmbeddingPipeline
from populate_cache import CONFIGS

OUT = _REPO / "outputs" / "reports"


def main():
    corpus = SBLGNTLoader().load()
    double, triple = corpus.double_tradition, corpus.triple_tradition
    labels = [p.label for p in double]
    n, nt = len(double), len(triple)
    out = ["# Robustness of the central contrast to the encoder configuration", ""]
    key = {}
    rows = []
    primary_rank = None
    for cname, cfg in CONFIGS.items():
        pipe = EmbeddingPipeline(config=cfg)
        M = np.array([pipe.embed_parts(p.matthew_parts) for p in double], float)
        L = np.array([pipe.embed_parts(p.luke_parts) for p in double], float)
        TM = np.array([pipe.embed_parts(p.matthew_parts) for p in triple], float)
        TK = np.array([pipe.embed_parts(p.mark_parts) for p in triple], float)
        TL = np.array([pipe.embed_parts(p.luke_parts) for p in triple], float)
        mu = np.vstack([M, L, TM, TK, TL]).mean(axis=0)
        out.append(f"## {cname}: {pipe.embedder_id}\n")
        out.append("| Set | Space | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess |")
        out.append("|---|---|---|---|---|---|---|")
        E = {}
        k = {}
        for tag, c in (("raw", None), ("cen", mu)):
            mats = {"dt": cos_matrix(M, L, c), "tt": cos_matrix(TM, TL, c),
                    "lk": cos_matrix(TL, TK, c), "mk": cos_matrix(TM, TK, c)}
            for s, Sm in mats.items():
                f = floors(Sm)
                e, e_nn = per_pericope_excess(Sm)
                E[(tag, s)] = (e, e_nn)
                lo, hi = bootstrap_ci(e, SEED + 11)
                out.append(f"| {s.upper()} | {tag} | {f['matched']:.4f} | {f['floor']:.4f} | {f['nn']:.4f} | "
                           f"{f['excess']:.4f} [{lo:.4f}, {hi:.4f}] | {f['excess_nn']:.4f} |")
                rows.append([cname, s, tag, f"{f['matched']:.4f}", f"{f['floor']:.4f}", f"{f['nn']:.4f}",
                             f"{f['excess']:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{f['excess_nn']:.4f}"])
                k[f"{tag}_{s}"] = dict(**f, excess_ci=[lo, hi])
            for a, b in (("dt", "tt"), ("dt", "lk")):
                r = compare_excess(E[(tag, a)][0], E[(tag, b)][0], SEED + 21)
                r_nn = compare_excess(E[(tag, a)][1], E[(tag, b)][1], SEED + 22)
                out.append(f"- {a.upper()} minus {b.upper()} excess ({tag}): {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                           f"MWU p = {r['p_mwu']:.3g}; NN excess {r_nn['diff']:+.4f} [{r_nn['ci_lo']:+.4f}, {r_nn['ci_hi']:+.4f}]")
                k[f"{tag}_{a}_minus_{b}"] = r
                k[f"{tag}_{a}_minus_{b}_nn"] = r_nn
            unrelated = mats["dt"][~np.eye(n, dtype=bool)].mean()
            k[f"{tag}_dt_unrelated_mean"] = float(unrelated)
        # Residual after word overlap (Jaccard), double and triple tradition.
        if "J_dt" not in globals():
            globals()["J_dt"] = jaccard_matrix([p.matthew for p in double], [p.luke for p in double])
            globals()["J_tt"] = jaccard_matrix([p.matthew for p in triple], [p.luke for p in triple])
        res = {}
        for s, Cm, Jm in (("dt", cos_matrix(M, L, mu), J_dt), ("tt", cos_matrix(TM, TL, mu), J_tt)):
            sl, ic, r_all, _, _ = _st.linregress(Jm.ravel(), Cm.ravel())
            res[s] = Cm - (ic + sl * Jm)
            k[f"resid_{s}_r_all"] = float(r_all)
        fR = floors(res["dt"])
        nul = permutation_tests(res["dt"], np.array([len(p.matthew) for p in double], float),
                                np.array([len(p.luke) for p in double], float), None, seed=SEED + 5)
        e_rd, _ = per_pericope_excess(res["dt"])
        e_rt, _ = per_pericope_excess(res["tt"])
        rr = compare_excess(e_rd, e_rt, SEED + 23)
        k["resid_dt"] = dict(**fR, random_null_mean=float(nul["random"].mean()),
                             random_p=perm_pvalue(nul["observed"], nul["random"]),
                             length_p=perm_pvalue(nul["observed"], nul["length"]))
        k["resid_dt_minus_tt"] = rr
        out.append(f"- Residual after Jaccard (r over all pairs = {k['resid_dt_r_all']:.3f}): matched {fR['matched']:+.4f}, "
                   f"mismatched {fR['floor']:+.4f}; random-null p = {k['resid_dt']['random_p']:.3g}, "
                   f"length-null p = {k['resid_dt']['length_p']:.3g}; DT minus TT on the residual "
                   f"{rr['diff']:+.4f} [{rr['ci_lo']:+.4f}, {rr['ci_hi']:+.4f}]")
        cen = np.diag(cos_matrix(M, L, mu))
        rank = stats.rankdata(-cen).astype(int)
        if primary_rank is None:
            primary_rank = cen.copy()
            rho = 1.0
        else:
            rho, _ = stats.spearmanr(cen, primary_rank)
        order = np.argsort(-cen)
        out.append(f"- Spearman rho of the double-tradition ranking with the primary configuration: {rho:.3f}")
        out.append("- Top five: " + "; ".join(f"{labels[i]} ({cen[i]:.3f})" for i in order[:5]))
        out.append("- Bottom five: " + "; ".join(f"{labels[i]} ({cen[i]:.3f})" for i in order[-5:]))
        out.append("")
        k["spearman_vs_primary"] = float(rho)
        k["top5"] = [labels[i] for i in order[:5]]
        k["bottom5"] = [labels[i] for i in order[-5:]]
        k["embedder"] = pipe.embedder_id
        key[cname] = k
    with open(OUT / "robustness_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "set", "space", "matched", "mean_floor", "nn_floor", "excess", "excess_ci_lo",
                    "excess_ci_hi", "excess_nn"])
        w.writerows(rows)

    def _clean(o):
        if isinstance(o, dict):
            return {kk: _clean(v) for kk, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "key_stats_robustness.json").write_text(json.dumps(_clean(key), indent=2), encoding="utf-8")
    (OUT / "robustness.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
