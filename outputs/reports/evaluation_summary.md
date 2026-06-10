# STCM Evaluation Summary

## 1. Permutation Test: Mean Q-Score (Random Null)

- Observed mean Q-score : 0.6204
- Null mean             : 0.4268
- Null std              : 0.0023
- p-value               : 0.0000
- z-score               : 83.646
- Interpretation        : SIGNIFICANT (p < 0.05)

## 2. Permutation Test: Top-10 Mean Q-Score

- Observed top-10 mean  : 0.6777
- p-value               : 0.0000
- z-score               : 44.983
- Interpretation        : SIGNIFICANT (p < 0.05)

## 3. Thematic-Null Permutation Test (Circularity Check)

Pairs each Matthean pericope with a *thematically similar* Lukan pericope
(e.g., wisdom with wisdom, apocalyptic with apocalyptic) rather than a
random one.  This is a more demanding null that controls for topical
similarity in Koine Greek.

- Observed mean Q-score : 0.6204
- Thematic null mean    : 0.4295
- Thematic null std     : 0.0024
- p-value               : 0.0000
- z-score               : 79.062
- Interpretation        : SIGNIFICANT — signal exceeds thematic baseline

## 4. Weight Sensitivity Analysis

Tests Q-score stability across nine alternative weighting schemes
(w_cosine, w_deviation, w_residual).  Top-5 Jaccard stability = 0.802
(1.0 = identical top-5 across all schemes).

  (0.50, 0.30, 0.20) (default): mean_Q=0.6204  top-5: Serving two masters, Lament over Jerusalem, Return of unclean spirit, Hidden from wise revealed, Jesus on John
  (0.60, 0.20, 0.20): mean_Q=0.7134  top-5: Serving two masters, Lament over Jerusalem, Return of unclean spirit, Hidden from wise revealed, Jesus on John
  (0.40, 0.40, 0.20): mean_Q=0.5274  top-5: Serving two masters, Lament over Jerusalem, Return of unclean spirit, Hidden from wise revealed, Jesus on John
  (0.40, 0.30, 0.30): mean_Q=0.5975  top-5: Serving two masters, Lament over Jerusalem, Return of unclean spirit, Hidden from wise revealed, Jesus on John
  (0.33, 0.33, 0.34): mean_Q=0.5605  top-5: Serving two masters, Lament over Jerusalem, Return of unclean spirit, Hidden from wise revealed, Jesus on John
  (0.70, 0.15, 0.15): mean_Q=0.7713  top-5: Serving two masters, Lament over Jerusalem, Return of unclean spirit, Jesus on John, Hidden from wise revealed
  (0.50, 0.50, 0.00): mean_Q=0.4800  top-5: Serving two masters, Jesus on John, Lament over Jerusalem, Return of unclean spirit, Anxieties about life
  (0.50, 0.00, 0.50): mean_Q=0.8309  top-5: Serving two masters, Lament over Jerusalem, Return of unclean spirit, Hidden from wise revealed, Lamp of the body
  (1.00, 0.00, 0.00): mean_Q=0.9452  top-5: Serving two masters, Jesus on John, Lament over Jerusalem, Return of unclean spirit, Anxieties about life

## 5. Sentence-Level Bootstrap Robustness

Resamples sentences within each pericope (with replacement) and recomputes
Q-scores to measure stability under meaningful input perturbation.

- mean_bootstrap_std=0.0449 (across 200 resamples per pericope)
- Interpretation        : ROBUST (mean std < 0.05)

## 6. Word-Overlap vs. Embedding Q-Score Comparison

Compares embedding-based Q-scores with traditional word-level agreement
(Jaccard coefficient) to demonstrate that embeddings capture information
beyond simple verbal overlap.

- Pearson r             : 0.764
- p-value               : 0.0000
- Residual variance     : 0.0013
- Interpretation        : Moderate correlation — embeddings partially track verbal agreement but capture additional semantic structure

## 7. Goulder Redaction Test

Compares Q-score distributions for pericopes Goulder (1989) identifies as
showing Lukan redaction of Matthew (n=10) against the
remainder (n=26).

- Goulder mean Q-score  : 0.5948
- Non-Goulder mean      : 0.6302
- Welch's t             : -1.381
- p-value               : 0.1932
- Cohen's d             : -0.655
- Interpretation        : No significant difference between Goulder-flagged and non-Goulder pericopes

## 8. Internal BERT Validation (Known NT Paraphrases)

Tests embedding-model calibration on known synoptic parallels (expected
high similarity) vs. unrelated passage pairs (expected low similarity).

  Lord's Prayer: Matt vs Luke                   sim=0.970
  Beatitudes: Matt vs Luke                      sim=0.951
  Feeding 5000: Matt vs Mark                    sim=0.975
  Baptism: Mark vs Luke                         sim=0.939
  Genealogy vs Crucifixion                      sim=0.761
  Sermon Mount vs Passion                       sim=0.860

- Paraphrase mean       : 0.959
- Control mean          : 0.810
- Separation            : 0.149
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