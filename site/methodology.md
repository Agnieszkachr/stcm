# Methodology

STCM operates as a computational pipeline applying modern NLP (Natural Language Processing) to Ancient Greek text.

## 0. Pericope Alignment
Pericopes are delimited by the Aland synopsis and, for the double tradition, the inventory of the International Q Project. Three rules are enforced by the loader (`stcm/data_loader.py`):

- each Matthew–Luke pair is trimmed to the verses that are actually parallel; where one evangelist scatters the material the unit is assembled from non-contiguous ranges (Luke 3:7–9 + 16–17 against Matt 3:7–12; Matt 23:4, 6–7, 13, 23–36 against Luke 11:39–52);
- no verse belongs to two units of the same set;
- no verse of Matthew or Luke belongs to both the calibration and the test set — Q sayings embedded in Markan contexts are cut out of the calibration units, and Mark–Q overlaps (Temptation, Beelzebul, Mission charge) are excluded from calibration altogether.

The result is 46 triple-tradition and 51 double-tradition units. The SBLGNT critical-apparatus sigla (⸀ ⸂ ⸃ …), which are editorial marks rather than text, are stripped before encoding.

## 1. Text Representation (Embeddings)
We use `ABeZet/Koine-Greek-BERT`, a transformer model pre-trained on the Hellenistic and biblical Greek register to support semantic matching in ancient literature. Each pericope is encoded as **one sequence**, so every token is contextualised by the whole passage, with dropout disabled (evaluation mode). We extract the attention-masked mean of the last hidden state, giving a 768-dimensional vector normalised to unit length. Texts longer than the 512-token window (one double-tradition and nine calibration texts) are split at sentence boundaries into the fewest chunks that fit, each chunk is encoded in full, and the chunk vectors are averaged with token-count weights; nothing is truncated. Embeddings are cached under a key that includes the model, the encoding-scheme version and the normalised text.

If the transformer model cannot be loaded, the pipeline stops with an error; a deterministic 256-dimensional character-trigram hashing embedder can be enabled explicitly (`STCM_ALLOW_FALLBACK=1`) for smoke-testing only.

An **internal validation** on known triple-tradition parallels (Feeding of the Five Thousand across Matt/Mark, the Paralytic across Matt/Luke, the Transfiguration and the Baptism across Mark/Luke) against unrelated pairs confirms that the model discriminates known synoptic parallels from unrelated passages in the specific register of New Testament Greek (raw 0.964 vs 0.850; centred 0.48–0.76 vs 0.02–0.04). No double-tradition passage is used for validation, since that is the test set.

## 2. Calibration (Triple Tradition)
We analyse the 46 passages where Matthew, Mark, and Luke overlap with a plain Markan parallel. Assuming Markan priority, Matthew and Luke are independent derivations from Mark. We compute:

- **Signature A:** The mean cosine similarity between Matthew and Luke (raw μ = 0.9571, σ = 0.0231, 95% CI [0.9500, 0.9637]; centred μ = 0.5681), giving us an expected baseline of similarity for texts drawn from the same source.
- **Signature B:** The correlation of Matthew's residual (Matthew minus Mark) and Luke's residual (Luke minus Mark). This captures whether two authors make structurally similar modifications to a shared source (μ = 0.3542, σ = 0.1359).

For double-tradition pericopes (where Mark is absent), the residual is computed relative to the **Mark centroid** — the mean embedding of all Mark's triple-tradition pericopes. Mark's material is overwhelmingly *narrative*, whereas the double tradition is predominantly *discourse*, so any two sayings passages show correlated displacement against this narrative baseline. That elevation is measured directly as the **genre floor** — the mean residual correlation among mismatched double-tradition pairs (0.198) — and the evidential quantity is the matched-pair excess over it (0.756 − 0.198 = 0.558). The residual *norm* is not used as evidence: it is a near-deterministic function of passage length (Spearman ρ = −0.82).

## 3. Primary Statistic — Mean-Centred Cosine
Contextual embedding spaces are anisotropic: vectors crowd into a narrow cone, so even unrelated texts show high raw cosine similarity (Ethayarajh 2019). The primary statistic therefore subtracts the corpus mean vector, computed over all 240 pericope embeddings, before taking cosines (cf. Su et al. 2021):

```
centred cosine = cos(emb(Matt) − μ, emb(Luke) − μ)
```

This single, weight-free statistic carries **all inference and ranking**. Under it, unrelated pairs average 0.0517 while matched double-tradition pairs average 0.7171. The raw cosine is reported in parallel throughout and the residual correlation is analysed separately.

A legacy composite index (0.8 × cosine + 0.2 × max(0, residual)) survives only behind `run_pipeline.py --legacy`. It is **not** used for inference or ranking.

## 4. Comparing the Two Traditions
The corpus mean is not neutral between the traditions: 138 of the 240 vectors are triple-tradition, so the origin of the centred space lies nearer that material and centring shortens those vectors more (mean centred norm 0.3078 against 0.4045). Setting the two matched means side by side would therefore flatter the double tradition. Each set is instead measured against **its own** mismatched-pair floor.

On that basis the result does not depend on the anisotropy correction at all. In the raw space the matched means are nearly identical (0.9509 and 0.9571), but the triple-tradition floor is far higher (0.9064 against 0.8378), because that material is internally homogeneous narrative inherited from Mark. The excess over floor is 0.1131 against 0.0506; after centring, 0.6654 against 0.5165.

Two controls close the remaining gaps:

- **Nearest-neighbour floor** — each pericope measured against the most similar passage it is *not* parallel to, which removes any advantage the double tradition might gain from being internally more varied. Because this is an order statistic and rises with pool size, the larger set (the double tradition, 51) is subsampled to the smaller (46) over 1,000 draws. Excess: 0.0473 against 0.0116 raw, 0.3371 against 0.1803 centred; **none** of the 1,000 draws falls to the triple-tradition value.
- **Centring sensitivity** — the comparison is repeated under six definitions of the centring vector, including one that balances the traditions by construction. The ordering is invariant across all six.

## 5. Evaluation Suite
All tests below are computed on the mean-centred cosine, with the raw cosine reported in parallel (`advanced_analysis.py`):

1. **Random permutation test** — 1,000 random Matt–Luke pairings produce a null distribution; the observed mean is tested against it (0.7171 vs null 0.0635, *z* = 27.3, empirical *p* < 0.001).
2. **Thematic-null permutation test** — A more demanding null pairing each Matthean pericope with a *thematically similar* Lukan one (wisdom with wisdom, apocalyptic with apocalyptic), controlling for topical overlap. The signal survives (null 0.1098, *z* = 31.6, *p* < 0.001).
3. **Length-matched nulls** — Correctly paired pericopes are of similar length (mean |log ratio| 0.22) whereas random pairs are not (0.85), and unrelated passages of unequal length are less similar than unrelated passages of equal length (ρ = −0.20 over the 2,550 mismatched pairs). Two nulls therefore pair each Matthean pericope with a Lukan one of comparable length (|log ratio| ≤ 0.4), from the whole set or from the same theme. The signal survives both (nulls 0.0924 and 0.1387; *z* = 28.3 and 29.2).
4. **Raw-cosine permutation tests** — The same four nulls on the uncorrected cosine; fully concordant (0.9509 vs nulls 0.8398–0.8542, all *p* < 0.001).
5. **Residual-correlation permutation tests** — The same four nulls on the residual correlation alone (0.7558 vs nulls 0.208–0.267, all *p* < 0.001).
6. **Cross-tradition floor comparison** — Mismatched-pair floor for the triple tradition as well as the double, so each set is read against its own baseline.
7. **Nearest-neighbour floor with size matching** — Controls for internal homogeneity; the larger set is subsampled to the smaller over 1,000 draws.
8. **Centring-vector sensitivity** — Recomputes both traditions under six centring schemes.
9. **Sentence-level bootstrap** — Resamples sentences within each pericope (200 resamples); mean perturbation SD 0.1227 on the centred scale.
10. **Covariates** — Length in characters (*r* = 0.06, n.s.), form-critical class (Kruskal–Wallis *p* = 0.029; discourse > narrative > proverbial > parable), Kloppenborg stratum (Q¹ vs Q² *p* = 0.956), word-level Jaccard overlap (*r* = 0.797; ~37% of variance unexplained).
11. **Goulder redaction test** — Compares centred-cosine distributions for the pericopes Goulder (1989) identifies as showing Lukan redaction of Matthew against the remainder (Welch's *t* = −1.498, *p* = 0.164, Cohen's *d* = −0.767). Underpowered at *n* = 10 vs *n* = 41; genuinely inconclusive.
12. **Directionality inference** — Leave-one-out ridge regression with intercept (hat-matrix identity), ridge penalty chosen by LOO error and reported across a grid, jackknife SE, sign-flip permutation test, plus two controls: a length-balanced subset and the known-direction case of the triple tradition (Mark → Matthew / Luke). The double-tradition asymmetry (Δ*R*² = +0.024, sign-flip *p* = 0.004) is regulariser-dependent, vanishes in length-balanced pairs (*p* = 0.58), and the statistic does not recover the known Markan direction (*p* = 0.73 / 0.68), so no directional inference is drawn.
13. **Internal BERT validation** (legacy suite) — Model calibration on known triple-tradition parallels vs unrelated passage pairs (separation 0.114 in the uncorrected space).
