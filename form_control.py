"""
form_control.py
===============
Is the double tradition's larger excess over its floor a property of the
*kind* of material (sayings against narrative) rather than of its
transmission?  The double tradition is mostly sayings; the triple
tradition mostly narrative, and evangelists may reproduce sayings more
closely than narrative whatever their source.

The triple tradition is split into units whose Markan text consists
predominantly of sayings of Jesus (SAYINGS_TT, an annotation made for this
control and listed in the article's appendix) and the rest.  The
per-pericope excess over the floor (the floor computed over the whole
triple tradition, as in Table 2) is then compared: double tradition
against triple-tradition sayings, triple-tradition sayings against
triple-tradition narrative, and double tradition against Luke-Mark
copying of sayings.

Outputs
-------
    outputs/reports/form_control.md
    outputs/reports/key_stats_form_control.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.analysis_core import SEED, cos_matrix, per_pericope_excess, compare_excess  # noqa: E402
from stcm.data_loader import SBLGNTLoader  # noqa: E402
from stcm.embeddings import EmbeddingPipeline  # noqa: E402
from stcm.metadata import SAYINGS_TT  # noqa: E402

OUT = _REPO / "outputs" / "reports"


def main():
    corpus = SBLGNTLoader().load()
    pipe = EmbeddingPipeline()
    D, T = corpus.double_tradition, corpus.triple_tradition

    def emb(ps, g):
        return np.array([pipe.embed_parts(p.parts(g)) for p in ps], float)
    M, L, TM, TK, TL = emb(D, "matthew"), emb(D, "luke"), emb(T, "matthew"), emb(T, "mark"), emb(T, "luke")
    mu = np.vstack([M, L, TM, TK, TL]).mean(axis=0)
    labels = [p.label for p in T]
    missing = SAYINGS_TT - set(labels)
    if missing:
        raise KeyError(f"SAYINGS_TT labels not in the triple tradition: {missing}")
    say = np.array([l in SAYINGS_TT for l in labels])
    out = ["# Form-of-material control", "",
           f"Triple-tradition units classed as sayings-dominant (n = {int(say.sum())}): "
           + "; ".join(l for l in labels if l in SAYINGS_TT), ""]
    key = {"n_sayings_tt": int(say.sum()), "sayings_tt": [l for l in labels if l in SAYINGS_TT]}
    for tag, c in (("raw", None), ("cen", mu)):
        e_dt, _ = per_pericope_excess(cos_matrix(M, L, c))
        e_tt, _ = per_pericope_excess(cos_matrix(TM, TL, c))
        e_lk, _ = per_pericope_excess(cos_matrix(TL, TK, c))
        e_mk, _ = per_pericope_excess(cos_matrix(TM, TK, c))
        out.append(f"## {tag}\n")
        out.append(f"- Mean excess: DT {e_dt.mean():.4f}; TT sayings {e_tt[say].mean():.4f}; "
                   f"TT narrative {e_tt[~say].mean():.4f}; Luke-Mark sayings {e_lk[say].mean():.4f}; "
                   f"Matt-Mark sayings {e_mk[say].mean():.4f}")
        k = dict(dt=float(e_dt.mean()), tt_sayings=float(e_tt[say].mean()),
                 tt_narrative=float(e_tt[~say].mean()), lk_sayings=float(e_lk[say].mean()),
                 mk_sayings=float(e_mk[say].mean()))
        for name, a, b, s in (("dt_minus_tt_sayings", e_dt, e_tt[say], 1),
                              ("tt_sayings_minus_tt_narrative", e_tt[say], e_tt[~say], 2),
                              ("dt_minus_lk_sayings", e_dt, e_lk[say], 3),
                              ("dt_minus_mk_sayings", e_dt, e_mk[say], 4)):
            r = compare_excess(a, b, SEED + s)
            out.append(f"- {name}: {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
                       f"Mann-Whitney p = {r['p_mwu']:.3g}")
            k[name] = r
        out.append("")
        key[tag] = k
    (OUT / "key_stats_form_control.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (OUT / "form_control.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
