# Methodology

STCM operates as a computational pipeline applying modern NLP (Natural Language Processing) to Ancient Greek text.

## 1. Text Representation (Embeddings)
We use `ABeZet/Koine-Greek-BERT`, a transformer model pre-trained on the Hellenistic and biblical Greek register to support semantic matching in ancient literature. Each sentence in a pericope is passed through the model with dropout disabled (evaluation mode). We extract the mean-pooled token vectors (last hidden state), resulting in a 768-dimensional float vector. A full pericope is represented by the mean of its constituent sentence vectors, normalised to unit length.

If the transformer model cannot be loaded, the pipeline stops with an error; a deterministic 256-dimensional character-trigram hashing embedder can be enabled explicitly (`STCM_ALLOW_FALLBACK=1`) for smoke-testing only.

An **internal validation** on known NT paraphrases (e.g., Feeding of the Five Thousand across Matt/Mark, the Baptism across Mark/Luke) confirms that the model discriminates known synoptic parallels from unrelated passages in the specific register of New Testament Greek.

## 2. Calibration (Triple Tradition)
We analyse the 49 passages where Matthew, Mark, and Luke overlap. Assuming Markan priority, Matthew and Luke are independent derivations from Mark. We compute:

- **Signature A:** The mean cosine similarity between Matthew and Luke (μ = 0.9472, σ = 0.0321, 95% CI [0.9376, 0.9561]), giving us an expected baseline of similarity for texts drawn from the same source.
- **Signature B:** The correlation of Matthew's residual (Matthew minus Mark) and Luke's residual (Luke minus Mark). This captures whether two authors make structurally similar modifications to a shared source (μ = 0.3693, σ = 0.1972).

For double-tradition pericopes (where Mark is absent), the residual is computed relative to the **Mark centroid** — the mean embedding of all Mark's triple-tradition pericopes. This centroid represents the typical direction of synoptic narrative as shaped by Mark in embedding space. However, Mark's triple-tradition material is overwhelmingly *narrative*, whereas the double-tradition material is predominantly *discourse* and *sayings*. Computing a residual against a narrative centroid for sayings material may introduce structural distortion, potentially inflating residuals. The magnitude of this effect is difficult to quantify without a discourse-specific calibration set.

## 3. Q-Scoring (Double Tradition)
For passages where Matthew and Luke share material absent from Mark (the Double Tradition), we compute a composite **Q-Score**:

```
Q-score = 0.8 × cos(Matt, Luke) + 0.2 × max(0, resid_sim)
```

Raw cosine similarity is the dominant signal, supplemented by a down-weighted residual-correlation component. Because the residual term is computed against a narrative Mark centroid and is therefore potentially genre-confounded, the composite is interpreted as a *relative ranking index*; the raw-cosine-only model (w₂ = 0) provides the confound-free significance check.

A **sensitivity analysis** across four weighting schemes — (0.8, 0.2), (0.7, 0.3), (0.9, 0.1), and (1.0, 0.0) — confirms that the top-scoring pericopes remain highly stable regardless of parametrisation (top-5 Jaccard = 0.833).

## 4. Latent Q Reconstruction
We estimate the hypothetical embedding of Q using an iterative, ridge-regularised centroid-shrinkage algorithm. For each pericope, the calibrated residual transforms are stripped from the Matthew and Luke embeddings, and a weighted centroid (including the previous estimate as a shrinkage regulariser) converges when the L2 norm of the update step falls below 1 × 10⁻⁶ (max 100 iterations). All 36 pericopes converge.

## 5. Evaluation Suite
The evaluation module performs eight analyses:

1. **Random permutation test** — 1000 random Matt–Luke pairings produce a null distribution; observed Q-scores are tested against this (empirical *p* = 0.001, 1,000 permutations).
2. **Top-10 permutation test** — As above, for the ten highest-scoring pericopes.
3. **Thematic-null permutation test** — A more demanding null that pairs each Matthean pericope with a *thematically similar* Lukan pericope (wisdom with wisdom, apocalyptic with apocalyptic, etc.), controlling for the possibility that topical overlap alone inflates similarity. Signal remains significant (empirical *p* = 0.001, 1,000 permutations).
4. **Weight sensitivity analysis** — Recomputes Q-scores under four alternative weighting schemes (including the raw-cosine-only model); reports top-5 Jaccard stability.
5. **Sentence-level bootstrap** — Resamples sentences within each pericope (200 resamples) to measure robustness to meaningful input perturbation (mean std = 0.0567).
6. **Word-overlap comparison** — Correlates embedding Q-scores with word-level Jaccard coefficients to demonstrate that embeddings capture information beyond lexical overlap.
7. **Goulder redaction test** — Compares Q-score distributions of pericopes Goulder (1989) identifies as demonstrating Lukan redaction of Matthew against the remainder (Welch's t-test, Cohen's d). The test is underpowered at current sample sizes (*n* = 10 vs. *n* = 26); the medium effect size leaves the result genuinely inconclusive.
8. **Internal BERT validation** — Tests model calibration on known NT paraphrases vs. unrelated passage pairs.
