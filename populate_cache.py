"""
populate_cache.py
=================
Embed every text the analysis needs and store the vectors in
``data/processed`` (the pipeline's cache), so that ``advanced_analysis.py``
and ``baselines.py`` can then run without the model.

Four encoder configurations are populated:

* primary   — Koine-Greek-BERT (pinned revision), last hidden layer;
* last4     — the same model, mean of the last four hidden layers;
* agb       — Ancient-Greek-BERT (the base model of Koine-Greek-BERT; its
              pre-training corpus includes the PROIEL treebank, which
              contains the Greek New Testament), last layer, diacritics
              stripped before tokenisation;
* greekbert — Greek BERT (nlpaueb, Modern Greek: Wikipedia, Europarl,
              OSCAR), a model without biblical pre-training, last layer,
              diacritics stripped as its model card prescribes.

For every pericope and gospel two encodings are produced: the range-wise
encoding used throughout (``embed_parts``) and, for composite units, the
single-sequence encoding of the concatenated ranges, which the sensitivity
table in the appendix compares against.

Usage
-----
    python populate_cache.py                 # all four configurations
    python populate_cache.py --config primary
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import pathlib

_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.config import EmbeddingConfig  # noqa: E402
from stcm.data_loader import SBLGNTLoader  # noqa: E402
from stcm.embeddings import EmbeddingPipeline  # noqa: E402

KOINE = "ABeZet/Koine-Greek-BERT"
KOINE_REV = "b81172ecb9baf65734c88ab65e2eb2491f12f3de"   # v1.0.1
AGB = "pranaydeeps/Ancient-Greek-BERT"
AGB_REV = "5e3e29ece1d63029baa226f11105b1e8277c4f07"
GRB = "nlpaueb/bert-base-greek-uncased-v1"
GRB_REV = "ec2b8f88dd215b5246f2f850413d5bff90d7540d"

CONFIGS = {
    "primary": EmbeddingConfig(model_name=KOINE, model_revision=KOINE_REV, layers="last",
                               strip_accents=False),
    "last4":   EmbeddingConfig(model_name=KOINE, model_revision=KOINE_REV, layers="last4",
                               strip_accents=False),
    "agb":     EmbeddingConfig(model_name=AGB, model_revision=AGB_REV, layers="last",
                               strip_accents=True),
    "greekbert": EmbeddingConfig(model_name=GRB, model_revision=GRB_REV, layers="last",
                                 strip_accents=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    corpus = SBLGNTLoader().load()
    names = [args.config] if args.config else list(CONFIGS)
    for name in names:
        pipe = EmbeddingPipeline(config=CONFIGS[name])
        t0 = time.time()
        n = 0
        for p in corpus.all_pericopes:
            for gospel in ("matthew", "mark", "luke"):
                parts = p.parts(gospel)
                if not parts:
                    continue
                pipe.embed_parts(parts)
                n += len(parts)
                if len(parts) > 1:
                    pipe.embed_text(" ".join(parts))   # single-sequence encoding
                    n += 1
        logging.info("%s: %d texts embedded or found in cache in %.0f s", name, n, time.time() - t0)


if __name__ == "__main__":
    main()
