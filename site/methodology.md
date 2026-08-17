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

## 3. Primary Statistic — Mean-Centred Cosine
Contextual embedding spaces are anisotropic: vectors crowd into a narrow cone, so even unrelated texts show high raw cosine similarity (Ethayarajh 2019). The primary statistic therefore subtracts the corpus mean vector, computed over all 219 pericope embeddings, before taking cosines (cf. Su et al. 2021):

```
centred cosine = cos(emb(Matt) − μ, emb(Luke) − μ)
```

This single, weight-free statistic carries **all inference and ranking**. Under it, unrelated pairs average 0.0589 while matched double-tradition pairs average 0.6567. The raw cosine is reported in parallel throughout and the residual correlation is analysed separately.

A legacy composite index (0.8 × cosine + 0.2 × max(0, residual)) survives in the pipeline outputs for backward compatibility with earlier versions of this project, together with its weight-sensitivity diagnostic. It is **not** used for inference or ranking.

## 4. Comparing the Two Traditions
The corpus mean is not neutral between the traditions: 147 of the 219 vectors are triple-tradition, so the origin of the centred space lies nearer that material and centring shortens those vectors more (mean centred norm 0.3036 against 0.3895). Setting the two matched means side by side would therefore flatter the double tradition. Each set is instead measured against **its own** mismatched-pair floor.

On that basis the result does not depend on the anisotropy correction at all. In the raw space the matched means are nearly identical (0.9452 and 0.9472), but the triple-tradition floor is far higher (0.9037 against 0.8510), because that material is internally homogeneous narrative inherited from Mark. The excess over floor is 0.0941 against 0.0435; after centring, 0.5979 against 0.4654.

Two controls close the remaining gaps:

- **Nearest-neighbour floor** — each pericope measured against the most similar passage it is *not* parallel to, which removes any advantage the double tradition might gain from being internally more varied. Because this is an order statistic and rises with pool size, the triple tradition is subsampled to *n* = 36 over 1,000 draws. Excess: 0.0380 against 0.0020 raw, 0.2658 against 0.1155 centred; **0 of 1,000** draws reach the double-tradition value.
- **Centring sensitivity** — the comparison is repeated under six definitions of the centring vector, including one that balances the traditions by construction. The ordering is invariant across all six.

## 5. Latent Q Reconstruction
We estimate the hypothetical embedding of Q using an iterative, ridge-regularised centroid-shrinkage algorithm. For each pericope, the calibrated residual transforms are stripped from the Matthew and Luke embeddings, and a weighted centroid (including the previous estimate as a shrinkage regulariser) converges when the L2 norm of the update step falls below 1 × 10⁻⁶ (max 100 iterations). All 36 pericopes converge.

## 6. Evaluation Suite
All tests below are computed on the mean-centred cosine, with the raw cosine reported in parallel:

1. **Random permutation test** — 1,000 random Matt–Luke pairings produce a null distribution; the observed mean is tested against it (0.6567 vs null 0.0737, *z* = 20.98, empirical *p* < 0.001).
2. **Top-10 permutation test** — As above, for the ten highest-ranked pericopes.
3. **Thematic-null permutation test** — A more demanding null pairing each Matthean pericope with a *thematically similar* Lukan one (wisdom with wisdom, apocalyptic with apocalyptic), controlling for topical overlap. The signal survives (null 0.1164, *z* = 25.37, *p* < 0.001).
4. **Raw-cosine permutation tests** — The same two tests on the uncorrected cosine; fully concordant (0.9452 vs nulls 0.8533 and 0.8585, both *p* < 0.001).
5. **Cross-tradition floor comparison** — Computes the mismatched-pair floor for the triple tradition as well as the double, so each set is read against its own baseline.
6. **Nearest-neighbour floor with size matching** — Controls for internal homogeneity; the triple tradition is subsampled to *n* = 36 over 1,000 draws.
7. **Centring-vector sensitivity** — Recomputes both traditions under six centring schemes.
8. **Sentence-level bootstrap** — Resamples sentences within each pericope (200 resamples); mean perturbation SD 0.1309 on the centred scale.
9. **Word-overlap comparison** — Correlates centred cosines with word-level Jaccard coefficients (*r* = 0.770; ~41% of variance unexplained).
10. **Goulder redaction test** — Compares centred-cosine distributions for the pericopes Goulder (1989) identifies as showing Lukan redaction of Matthew against the remainder (Welch's *t* = −1.177, *p* = 0.261, Cohen's *d* = −0.523). Underpowered at *n* = 10 vs *n* = 26; genuinely inconclusive.
11. **Directionality inference** — Exact leave-one-out ridge regression with a bootstrap CI and a sign-flip permutation test; no significant asymmetry (Δ*R*² = +0.030, *p* = 0.102).
12. **Internal BERT validation** — Model calibration on known NT paraphrases vs unrelated passage pairs (separation 0.149 in the uncorrected space).
