# STCM Advanced Analysis

(n_double = 36, n_triple = 49, permutations = 1000, bootstrap = 1000, seed = 42)

## 1. Raw-cosine permutation tests

- Observed mean        : 0.9452
- Random null mean     : 0.8533 (SD 0.0044)
- Random p / z         : 0.000999 / 20.66
- Thematic null mean   : 0.8585 (SD 0.0048)
- Thematic p / z       : 0.000999 / 18.01

## 2. Anisotropy diagnosis and mean-centred similarities

| Quantity | Raw cosine | Mean-centred cosine |
|---|---|---|
| Triple tradition: Matt-Luke mean (matched; Sig-A) | 0.9472 | 0.4749 |
| Triple tradition: mismatched-pair floor (mean) | 0.9037 | 0.0096 |
| Triple tradition: nearest non-parallel floor | 0.9466 | 0.3816 |
| Triple tradition: excess over mean floor | 0.0435 | 0.4654 |
| Triple tradition: excess over nearest floor | 0.0006 | 0.0934 |
| Double tradition: Matt-Luke mean (matched) | 0.9452 | 0.6567 |
| Double tradition: mismatched-pair floor (mean) | 0.8510 | 0.0589 |
| Double tradition: nearest non-parallel floor | 0.9072 | 0.3909 |
| Double tradition: excess over mean floor | 0.0941 | 0.5979 |
| Double tradition: excess over nearest floor | 0.0380 | 0.2658 |

- Triple tradition subsampled to n=36, excess over nearest floor (raw): mean +0.0020, SD 0.0021, 95% of draws [-0.0018, +0.0062]
- Triple tradition subsampled to n=36, excess over nearest floor (centred): mean +0.1155, SD 0.0199, 95% of draws [+0.0794, +0.1549]

Internal homogeneity (mean similarity among a single evangelist's own pericopes within each tradition, raw cosine):
- Triple tradition: Matthew 0.9090 | Luke 0.9046
- Double tradition: Matthew 0.8447 | Luke 0.8622

- Mean centred norm, double-tradition vectors   : 0.3895
- Mean centred norm, triple-tradition Matt+Luke : 0.3036
- Mean centred norm, Mark vectors               : 0.3005

## 2c. Sensitivity to the centring vector

| Centring vector | DT matched | DT contrast | TT matched | TT contrast |
|---|---|---|---|---|
| All 219 pericope vectors (used throughout) | 0.6567 | 0.5979 | 0.4749 | 0.4654 |
| Double-tradition vectors only (72) | 0.6295 | 0.6578 | 0.5880 | 0.3609 |
| Triple-tradition vectors only (147) | 0.6899 | 0.5303 | 0.4527 | 0.4746 |
| Mark vectors only (49) | 0.7208 | 0.4824 | 0.4973 | 0.4345 |
| Mean of the two tradition means (balanced) | 0.6439 | 0.6254 | 0.4984 | 0.4465 |
| Matthew and Luke vectors only, both traditions (170) | 0.6455 | 0.6205 | 0.4915 | 0.4524 |

## 2b. Permutation tests on mean-centred cosine (primary statistic)

- Observed mean        : 0.6567
- Random null mean     : 0.0737 (SD 0.0278)
- Random p / z         : 0.000999 / 20.98
- Thematic null mean   : 0.1164 (SD 0.0213)
- Thematic p / z       : 0.000999 / 25.37

## 3. Genre floor for the residual signature

- Matched-pair mean residual correlation   : 0.7166
- Mismatched-pair mean (genre floor)       : 0.2290 (SD 0.1522)
- Matched-minus-floor contrast             : 0.4876

## 3b. Permutation tests on residual correlation alone

- Observed mean        : 0.7166
- Random null mean     : 0.2413 (SD 0.0224)
- Random p / z         : 0.000999 / 21.27
- Thematic null mean   : 0.2721 (SD 0.0196)
- Thematic p / z       : 0.000999 / 22.64

## 4. Directionality analysis

- Residual norm, Matthew mean              : 0.4453
- Residual norm, Luke mean                 : 0.4058
- Wilcoxon signed-rank (paired)            : W = 68.0, p = 6.657e-06
- Exact LOO ridge R^2, Luke | Matthew      : 0.1628
- Exact LOO ridge R^2, Matthew | Luke      : 0.1331
- delta-R2 (L|M minus M|L)                 : +0.0297
- 95% percentile bootstrap CI              : [-0.0041, +0.0346]
- Sign-flip permutation p (two-sided)      : 0.1019

## 5. Centred-cosine ranking (primary ranking statistic)

- Mean (SD, sample)    : 0.6567 (0.1981)
- Range                : [0.1799, 0.9833]

Top five:
  1. Serving two masters            centred=0.9833 raw=0.9945 resid=0.9853
  2. Lament over Jerusalem          centred=0.9150 raw=0.9876 resid=0.9363
  3. Return of unclean spirit       centred=0.8869 raw=0.9856 resid=0.8924
  4. Lamp of the body               centred=0.8827 raw=0.9657 resid=0.8971
  5. Hidden from wise revealed      centred=0.8618 raw=0.9797 resid=0.8949
Bottom three:
  34. Leaven of Pharisees            centred=0.3286 raw=0.9119 resid=0.4432
  35. Parable of Great Banquet       centred=0.2795 raw=0.9560 resid=0.3923
  36. Narrow gate                    centred=0.1799 raw=0.8235 resid=0.2754

### 5b. Confound analyses on the centred cosine

- Length: Pearson r = -0.028 (p = 0.870); Spearman rho = -0.089 (p = 0.606)
- Form proverbial : mean = 0.6306, n = 14
- Form discourse  : mean = 0.7015, n = 13
- Form parable    : mean = 0.5911, n = 5
- Form narrative  : mean = 0.6272, n = 3
- Form liturgical : mean = 0.8580, n = 1
- Form Kruskal-Wallis (n >= 3 groups): H = 2.526, p = 0.471
- Strata: Q1 mean = 0.6701 (n = 11), Q2 mean = 0.6420 (n = 22)
- Strata Mann-Whitney U = 124.0, p = 0.924; Kruskal-Wallis H = 0.013, p = 0.909
- Goulder set: mean = 0.5849 (n = 10) vs 0.6843 (n = 26); Welch t = -1.177, p = 0.2609, Cohen's d = -0.523
- Word-overlap Pearson r = 0.770 (p = 4.046e-08); unexplained variance ~ 41%

### 5c. Sentence-level bootstrap on the centred cosine

- Mean per-pericope bootstrap SD (200 resamples): 0.1309
