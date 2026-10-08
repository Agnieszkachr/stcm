# STCM Evaluation Summary

> [!WARNING]
> **LEGACY REPORT — NOT THE STATISTICS REPORTED IN THE ARTICLE.**
> Every figure below is computed on the **retired composite Q-score**
> (0.8 x cosine + 0.2 x max(0, residual)), which no longer carries any
> inference or ranking in this project. The word-overlap correlation,
> sentence-level bootstrap and Goulder test therefore differ from the
> published values, because there they are computed on the mean-centred
> cosine. Do not compare these numbers with the article.
>
> The authoritative report is `outputs/reports/advanced_analysis.md`,
> produced by `advanced_analysis.py`. This file is retained only for
> backward compatibility with earlier versions of the pipeline.

## 1. Permutation Test: Mean Q-Score (Random Null)

- Observed mean Q-score : 0.9119
- Null mean             : 0.7150
- Null std              : 0.0069
- p-value               : 0.0000
- z-score               : 28.442
- Interpretation        : SIGNIFICANT (p < 0.05)

## 2. Permutation Test: Top-10 Mean Q-Score

- Observed top-10 mean  : 0.9752
- p-value               : 0.0000
- z-score               : 9.074
- Interpretation        : SIGNIFICANT (p < 0.05)

## 3. Thematic-Null Permutation Test (Circularity Check)

Pairs each Matthean pericope with a *thematically similar* Lukan pericope
(e.g., wisdom with wisdom, apocalyptic with apocalyptic) rather than a
random one.  This is a more demanding null that controls for topical
similarity in Koine Greek.

- Observed mean Q-score : 0.9119
- Thematic null mean    : 0.7271
- Thematic null std     : 0.0068
- p-value               : 0.0000
- z-score               : 27.183
- Interpretation        : SIGNIFICANT — signal exceeds thematic baseline

## 3b. Raw-Cosine (w2 = 0) Permutation Tests (Confound-Free)

Repeats both permutation tests on the mean raw cosine similarity alone,
excluding the potentially genre-confounded residual component.

- Observed mean cosine  : 0.9509
- Random null mean      : 0.8398
- Random null std       : 0.0041
- Random p-value        : 0.0000  (z=27.114)
- Thematic null mean    : 0.8470
- Thematic null std     : 0.0052
- Thematic p-value      : 0.0000  (z=19.855)

## 4. Weight Sensitivity Analysis

Tests Q-score stability across four alternative weighting schemes
(w_cosine, w_residual).  Top-5 Jaccard stability = 0.643
(1.0 = identical top-5 across all schemes).

  (0.80, 0.20) (default): mean_Q=0.9119  top-5: Serving two masters, Lament over Jerusalem, Harvest plentiful, Hidden from wise revealed, Return of unclean spirit
  (0.70, 0.30): mean_Q=0.8924  top-5: Serving two masters, Lament over Jerusalem, Harvest plentiful, Hidden from wise revealed, Return of unclean spirit
  (0.90, 0.10): mean_Q=0.9314  top-5: Serving two masters, Lament over Jerusalem, Harvest plentiful, Hidden from wise revealed, John's preaching
  (1.00, 0.00): mean_Q=0.9509  top-5: Serving two masters, Lament over Jerusalem, Harvest plentiful, John's preaching, Jesus on John

## 5. Sentence-Level Bootstrap Robustness

Resamples sentences within each pericope (with replacement) and recomputes
Q-scores to measure stability under meaningful input perturbation.

- mean_bootstrap_std=0.0529 (across 200 resamples per pericope)
- Interpretation        : MODERATE SENSITIVITY to sentence composition

## 6. Word-Overlap vs. Embedding Q-Score Comparison

Compares embedding-based Q-scores with traditional word-level agreement
(Jaccard coefficient) to demonstrate that embeddings capture information
beyond simple verbal overlap.

- Pearson r             : 0.794
- p-value               : 0.0000
- Residual variance     : 0.0012
- Interpretation        : Moderate correlation — embeddings partially track verbal agreement but capture additional semantic structure

## 7. Goulder Redaction Test

Compares Q-score distributions for pericopes Goulder (1989) identifies as
showing Lukan redaction of Matthew (n=10) against the
remainder (n=41).

- Goulder mean Q-score  : 0.8796
- Non-Goulder mean      : 0.9197
- Welch's t             : -1.417
- p-value               : 0.1861
- Cohen's d             : -0.736
- Interpretation        : No significant difference between Goulder-flagged and non-Goulder pericopes

## 8. Internal BERT Validation (Known NT Paraphrases)

Tests embedding-model calibration on known synoptic parallels (expected
high similarity) vs. unrelated passage pairs (expected low similarity).

  Feeding 5000: Matt vs Mark                    sim=0.977
  Paralytic: Matt vs Luke                       sim=0.973
  Transfiguration: Mark vs Luke                 sim=0.967
  Baptism: Mark vs Luke                         sim=0.939
  Genealogy vs Crucifixion                      sim=0.807
  Birth of Jesus vs Trial                       sim=0.893

- Paraphrase mean       : 0.964
- Control mean          : 0.850
- Separation            : 0.114
- Interpretation        : GOOD — model discriminates known parallels from unrelated text

## Caveats

- These results depend on the quality and coverage of the embedding model.
- Permutation tests assume exchangeability — valid for cosine-based
  statistics but may not hold if texts have structural dependencies.
- The thematic-null categories are broad; finer-grained genre tagging
  might yield a stricter baseline.
- A low p-value does NOT prove Q exists; it shows the embedding geometry
  is consistent with the Q hypothesis relative to both random and
  thematic baselines.