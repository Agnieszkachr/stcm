# STCM Advanced Analysis

(n_double = 51, n_triple = 46, corpus vectors = 240, permutations = 1000, bootstrap = 1000, seed = 42)

Lengths are in characters of the normalised Greek text. Double tradition: Matthew median 395, Luke median 335; triple tradition: Matthew median 719, Luke median 700.

## 1. Calibration signatures (triple tradition)

| Signature | Mean | SD | 95% bootstrap CI |
|---|---|---|---|
| A: Matt–Luke cosine (raw) | 0.9571 | 0.0231 | [0.9500, 0.9637] |
| A: Matt–Luke cosine (centred) | 0.5681 | 0.1735 | [0.5207, 0.6148] |
| B: residual correlation vs pericope-specific Mark | 0.3542 | 0.1359 | [0.3174, 0.3924] |
| Matt–Mark cosine (raw) | 0.9689 | 0.0176 | [0.9638, 0.9735] |
| Luke–Mark cosine (raw) | 0.9660 | 0.0157 | [0.9615, 0.9707] |

Triple-tradition Matt–Luke centred cosine, lowest five (alignment check):
  Rejection at Nazareth        0.0753
  Second passion prediction    0.1156
  Man with withered hand       0.2516
  Purpose of parables          0.2704
  Cleansing of temple          0.3180

## 2. Raw-cosine permutation tests (double tradition)

- Observed mean        : 0.9509
- Random null          : mean 0.8398 (SD 0.0041); p = 0.000999; z = 27.11
- Thematic null        : mean 0.8470 (SD 0.0052); p = 0.000999; z = 19.85
- Length-matched null  : mean 0.8466 (SD 0.0046); p = 0.000999; z = 22.77
- Length+theme null    : mean 0.8542 (SD 0.0036); p = 0.000999; z = 27.03

## 3. Anisotropy diagnosis and mean-centred similarities (Table 2)

| Quantity | Raw cosine | Mean-centred cosine |
|---|---|---|
| Triple tradition: Matt–Luke mean (matched; Signature A) | 0.9571 | 0.5681 |
| Triple tradition: mismatched-pair floor (mean) | 0.9064 | 0.0516 |
| Triple tradition: nearest non-parallel floor | 0.9455 | 0.3879 |
| Triple tradition: excess over mean floor | 0.0506 | 0.5165 |
| Triple tradition: excess over nearest floor | 0.0116 | 0.1803 |
| Double tradition: Matt–Luke mean (matched) | 0.9509 | 0.7171 |
| Double tradition: mismatched-pair floor (mean) | 0.8378 | 0.0517 |
| Double tradition: nearest non-parallel floor | 0.9042 | 0.3846 |
| Double tradition: excess over mean floor | 0.1131 | 0.6654 |
| Double tradition: excess over nearest floor | 0.0467 | 0.3325 |

- Double tradition subsampled to n=46, excess over nearest floor (raw): mean +0.0473, SD 0.0017, 95% of draws [+0.0439, +0.0506]
- Double tradition subsampled to n=46, excess over nearest floor (centred): mean +0.3371, SD 0.0108, 95% of draws [+0.3168, +0.3589]

Internal homogeneity (mean similarity among a single evangelist's own pericopes within each tradition, raw cosine):
- Triple tradition: Matthew 0.9093 | Luke 0.9074
- Double tradition: Matthew 0.8341 | Luke 0.8420

- Mean centred norm, double-tradition vectors   : 0.4045
- Mean centred norm, triple-tradition Matt+Luke : 0.3078
- Mean centred norm, Mark vectors               : 0.3040

### 3c. Sensitivity to the centring vector (Table 4)

| Centring vector | DT matched | DT contrast | TT matched | TT contrast |
|---|---|---|---|---|
| All 240 pericope vectors (used throughout) | 0.7171 | 0.6654 | 0.5681 | 0.5165 |
| Double-tradition vectors only (102) | 0.6990 | 0.7149 | 0.6736 | 0.3975 |
| Triple-tradition vectors only (138) | 0.7514 | 0.5734 | 0.5248 | 0.5459 |
| Mark vectors only (46) | 0.7605 | 0.5507 | 0.5448 | 0.5202 |
| Mean of the two tradition means (balanced) | 0.7126 | 0.6778 | 0.5805 | 0.5042 |
| Matthew and Luke vectors only, both traditions (194) | 0.7105 | 0.6839 | 0.5864 | 0.4985 |

## 4. Permutation tests on the mean-centred cosine (primary statistic)

- Observed mean        : 0.7171
- Random null          : mean 0.0635 (SD 0.0239); p = 0.000999; z = 27.34
- Thematic null        : mean 0.1098 (SD 0.0192); p = 0.000999; z = 31.58
- Length-matched null  : mean 0.0924 (SD 0.0220); p = 0.000999; z = 28.34
- Length+theme null    : mean 0.1387 (SD 0.0198); p = 0.000999; z = 29.24

- Length mismatch of pairs, |log(len_Matt/len_Luke)|: matched pairs mean 0.222, mismatched pairs mean 0.847; Spearman rho between centred cosine and length mismatch over all mismatched pairs = -0.201 (p = 1.1e-24).

## 5. Genre floor for the residual signature

- Matched-pair mean residual correlation   : 0.7558
- Mismatched-pair mean (genre floor)       : 0.1982 (SD 0.1557)
- Matched-minus-floor contrast             : 0.5576

## 5b. Permutation tests on the residual correlation alone

- Observed mean        : 0.7558
- Random null          : mean 0.2081 (SD 0.0199); p = 0.000999; z = 27.47
- Thematic null        : mean 0.2445 (SD 0.0175); p = 0.000999; z = 29.29
- Length-matched null  : mean 0.2283 (SD 0.0202); p = 0.000999; z = 26.16
- Length+theme null    : mean 0.2666 (SD 0.0177); p = 0.000999; z = 27.68

## 6. Directionality analysis

- Ridge penalty alpha (min total LOO error) : 1.0
- LOO error by alpha: 0.01: 87.896; 0.03: 86.017; 0.1: 81.854; 0.3: 77.517; 1.0: 76.659; 3.0: 81.417; 10.0: 89.370; 30.0: 94.579; 100.0: 97.166
- Exact LOO ridge R^2, Luke | Matthew      : 0.2331
- Exact LOO ridge R^2, Matthew | Luke      : 0.2095
- delta-R2 (L|M minus M|L)                 : +0.0236
- Jackknife SE / 95% CI                    : 0.0134 / [-0.0027, +0.0499]
- Sign-flip permutation p (two-sided)      : 0.003996
- delta-R2 and sign-flip p across the alpha grid: 0.01: +0.0627 (p = 0.069); 0.03: +0.0587 (p = 0.052); 0.1: +0.0500 (p = 0.025); 0.3: +0.0388 (p = 0.004); 1.0: +0.0236 (p = 0.006); 3.0: +0.0103 (p = 0.036); 10.0: +0.0020 (p = 0.289); 30.0: +0.0000 (p = 0.991); 100.0: -0.0001 (p = 0.245)

Controls for the directionality statistic:
- Length-balanced double-tradition pairs (|log length ratio| < 0.25, n = 30): delta-R2 = +0.0101, sign-flip p = 0.577
- Known direction, triple tradition (n = 46): Mark -> Matthew delta-R2 = +0.0044 (p = 0.730); Mark -> Luke delta-R2 = -0.0045 (p = 0.675); Matthew vs Luke (both copying Mark) delta-R2 = -0.0083 (p = 0.490)

Residual-norm diagnostic (NOT used as evidence of directionality):
- Residual norm, Matthew mean 0.4442, Luke mean 0.4285; Wilcoxon signed-rank W = 355.0, p = 0.00389
- Residual norm vs passage length (pooled): Spearman rho = -0.819 (p = 6.6e-26)
- Matt-minus-Luke residual norm vs log length ratio: rho = -0.342 (p = 0.0139)

## 7. Centred-cosine ranking (primary ranking statistic)

- Mean (SD, sample)    : 0.7171 (0.1737)
- Range                : [0.1464, 0.9788]

Top five:
  1. Serving two masters            centred=0.9788 raw=0.9945 resid=0.9814
  2. Harvest plentiful              centred=0.9549 raw=0.9896 resid=0.9610
  3. Lament over Jerusalem          centred=0.9441 raw=0.9921 resid=0.9573
  4. Hidden from wise revealed      centred=0.9153 raw=0.9882 resid=0.9260
  5. Return of unclean spirit       centred=0.9086 raw=0.9874 resid=0.9148
Bottom five:
  47. Lost sheep                     centred=0.5093 raw=0.9365 resid=0.6006
  48. Not peace but sword            centred=0.4408 raw=0.8700 resid=0.5140
  49. Judging twelve tribes          centred=0.4390 raw=0.8847 resid=0.5295
  50. Parable of Great Banquet       centred=0.1928 raw=0.9346 resid=0.2184
  51. Narrow gate                    centred=0.1464 raw=0.8245 resid=0.2203

### 7b. Confound analyses on the centred cosine

- Length (characters): Pearson r = 0.060 (p = 0.675); Spearman rho = 0.091 (p = 0.524)
- Form proverbial : mean = 0.6882, n = 25
- Form discourse  : mean = 0.8049, n = 15
- Form parable    : mean = 0.5730, n = 6
- Form narrative  : mean = 0.7502, n = 4
- Form liturgical : mean = 0.8541, n = 1
- Form Kruskal-Wallis (n >= 3 groups): H = 9.006, p = 0.029
- Strata counts: {'Q1': 20, 'Q2': 26, 'Q3': 1, 'U': 4}
- Strata: Q1 mean = 0.7166 (n = 20), Q2 mean = 0.7222 (n = 26)
- Strata Mann-Whitney U = 257.0, p = 0.956; Kruskal-Wallis H = 0.004, p = 0.947
- Goulder set: mean = 0.6138 (n = 10) vs 0.7423 (n = 41); Welch t = -1.498, p = 0.1643, Cohen's d = -0.767
- Word-overlap Pearson r = 0.797 (p = 2.675e-12); unexplained variance ~ 37%

### 7c. Sentence-level bootstrap on the centred cosine

- Mean per-pericope bootstrap SD (200 resamples): 0.1227
