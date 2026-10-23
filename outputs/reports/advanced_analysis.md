# STCM Advanced Analysis

(n_double = 51, n_triple = 45, corpus vectors = 237, permutations = 1000, bootstrap = 1000, seed = 42; encoder = ABeZet/Koine-Greek-BERT@b81172ecb9baf65734c88ab65e2eb2491f12f3de (last))

Lengths are in characters of the normalised Greek text. Double tradition: Matthew median 395, Luke median 335; triple tradition: Matthew median 727, Mark median 764, Luke median 685. Texts exceeding the 512-token window (split at sentence boundaries): 9 in the triple tradition, 0 in the double tradition.

## 1. Calibration signatures (triple tradition)

| Signature | Mean | SD | 95% bootstrap CI |
|---|---|---|---|
| A: Matt-Luke cosine (raw) | 0.9574 | 0.0233 | [0.9508, 0.9634] |
| A: Matt-Luke cosine (centred) | 0.5781 | 0.1645 | [0.5312, 0.6263] |
| Matt-Mark cosine (raw) | 0.9686 | 0.0177 | [0.9634, 0.9732] |
| Matt-Mark cosine (centred) | 0.6791 | 0.1327 | [0.6391, 0.7161] |
| Luke-Mark cosine (raw) | 0.9659 | 0.0166 | [0.9608, 0.9703] |
| Luke-Mark cosine (centred) | 0.6512 | 0.1431 | [0.6076, 0.6896] |
| B: residual correlation vs pericope-specific Mark | 0.3638 | 0.1169 | [0.3317, 0.3966] |

Triple-tradition Matt-Luke centred cosine, lowest five (alignment check):
  Second passion prediction    0.1060
  Man with withered hand       0.2543
  Rejection at Nazareth        0.2583
  Purpose of parables          0.2664
  Cleansing of temple          0.3189

## 2. Floors and excess over floor (Table 2)

| Set | Space | Matched | Mean floor | NN floor | Excess (mean floor) [95% CI] | Excess (NN floor) [95% CI] |
|---|---|---|---|---|---|---|
| Double tradition, Matt-Luke | raw | 0.9504 | 0.8389 | 0.9050 | 0.1115 [0.1021, 0.1200] | 0.0454 [0.0367, 0.0536] |
| Triple tradition, Matt-Luke | raw | 0.9574 | 0.9056 | 0.9460 | 0.0518 [0.0456, 0.0575] | 0.0114 [0.0057, 0.0172] |
| Triple tradition, Matt-Mark | raw | 0.9686 | 0.9108 | 0.9454 | 0.0578 [0.0527, 0.0626] | 0.0232 [0.0187, 0.0274] |
| Triple tradition, Luke-Mark | raw | 0.9659 | 0.9071 | 0.9414 | 0.0587 [0.0536, 0.0645] | 0.0245 [0.0195, 0.0300] |
| Double tradition, Matt-Luke | cen | 0.7082 | 0.0518 | 0.3815 | 0.6563 [0.6085, 0.7018] | 0.3267 [0.2725, 0.3807] |
| Triple tradition, Matt-Luke | cen | 0.5781 | 0.0529 | 0.3990 | 0.5251 [0.4747, 0.5736] | 0.1791 [0.1222, 0.2366] |
| Triple tradition, Matt-Mark | cen | 0.6791 | 0.0700 | 0.4079 | 0.6091 [0.5702, 0.6503] | 0.2712 [0.2356, 0.3089] |
| Triple tradition, Luke-Mark | cen | 0.6512 | 0.0653 | 0.3899 | 0.5859 [0.5395, 0.6347] | 0.2613 [0.2173, 0.3119] |

Contrasts between sets (difference of mean per-pericope excess, a minus b; bootstrap CI over pericopes, Mann-Whitney U, label-permutation p):

| Contrast | Space | Floor | Diff | 95% CI | MWU p | perm p |
|---|---|---|---|---|---|---|
| DT - TT | raw | mean | +0.0597 | [+0.0496, +0.0706] | 3.16e-14 | 0.0005 |
| DT - TT | raw | NN | +0.0340 | [+0.0237, +0.0447] | 3.98e-08 | 0.0005 |
| DT - LK | raw | mean | +0.0528 | [+0.0423, +0.0631] | 7.73e-13 | 0.0005 |
| DT - LK | raw | NN | +0.0209 | [+0.0108, +0.0312] | 0.000152 | 0.001 |
| DT - MK | raw | mean | +0.0537 | [+0.0439, +0.0636] | 2.1e-13 | 0.0005 |
| DT - MK | raw | NN | +0.0223 | [+0.0132, +0.0318] | 4.61e-05 | 0.0005 |
| LK - TT | raw | mean | +0.0069 | [-0.0006, +0.0151] | 0.262 | 0.094 |
| LK - TT | raw | NN | +0.0131 | [+0.0057, +0.0214] | 0.011 | 0.003 |
| MK - TT | raw | mean | +0.0060 | [-0.0011, +0.0137] | 0.249 | 0.133 |
| MK - TT | raw | NN | +0.0118 | [+0.0049, +0.0192] | 0.00358 | 0.0025 |
| DT - TT | cen | mean | +0.1312 | [+0.0633, +0.2003] | 0.000147 | 0.001 |
| DT - TT | cen | NN | +0.1476 | [+0.0715, +0.2273] | 0.000623 | 0.0005 |
| DT - LK | cen | mean | +0.0704 | [+0.0037, +0.1411] | 0.0292 | 0.057 |
| DT - LK | cen | NN | +0.0654 | [-0.0048, +0.1388] | 0.0756 | 0.091 |
| DT - MK | cen | mean | +0.0472 | [-0.0149, +0.1086] | 0.0686 | 0.157 |
| DT - MK | cen | NN | +0.0555 | [-0.0083, +0.1241] | 0.0831 | 0.0975 |
| LK - TT | cen | mean | +0.0608 | [-0.0055, +0.1323] | 0.0917 | 0.09 |
| LK - TT | cen | NN | +0.0822 | [+0.0117, +0.1586] | 0.0623 | 0.0295 |
| MK - TT | cen | mean | +0.0840 | [+0.0258, +0.1499] | 0.0181 | 0.014 |
| MK - TT | cen | NN | +0.0921 | [+0.0281, +0.1607] | 0.0325 | 0.007 |

- Double tradition subsampled to n=45, excess over nearest floor (raw): mean +0.0463, SD 0.0019, 95% of draws [+0.0427, +0.0502]; the other tradition's value is +0.0114
- Double tradition subsampled to n=45, excess over nearest floor (cen): mean +0.3332, SD 0.0121, 95% of draws [+0.3116, +0.3574]; the other tradition's value is +0.1791

Internal homogeneity (mean raw cosine among one evangelist's own pericopes within a tradition):
- Triple tradition: Matthew 0.9101 | Mark 0.9151 | Luke 0.9049
- Double tradition: Matthew 0.8353 | Luke 0.8431
- Mean centred norm: double-tradition vectors 0.4031; triple-tradition Matt+Luke 0.3096; Mark 0.3045

### 2c. Sensitivity to the centring vector (Table 4)

| Centring vector | DT matched | DT excess | TT matched | TT excess | Luke-Mark excess |
|---|---|---|---|---|---|
| All 237 pericope vectors (used throughout) | 0.7082 | 0.6563 | 0.5781 | 0.5251 | 0.5859 |
| Double-tradition vectors only (102) | 0.6893 | 0.7053 | 0.6792 | 0.4037 | 0.4444 |
| Triple-tradition vectors only (135) | 0.7450 | 0.5642 | 0.5357 | 0.5571 | 0.6335 |
| Mark vectors only (45) | 0.7547 | 0.5413 | 0.5554 | 0.5320 | 0.6245 |
| Mean of the two tradition means (balanced) | 0.7037 | 0.6678 | 0.5892 | 0.5133 | 0.5713 |
| Matthew and Luke vectors only, both traditions (192) | 0.7012 | 0.6746 | 0.5957 | 0.5060 | 0.5592 |

## 3a. Raw-cosine permutation tests (double tradition)

- Observed mean        : 0.9504
- Random null          : mean 0.8409 (SD 0.0041); p = 0.000999
- Thematic null        : mean 0.8481 (SD 0.0052); p = 0.000999
- Length-matched null  : mean 0.8468 (SD 0.0046); p = 0.000999
- Length+theme null    : mean 0.8542 (SD 0.0036); p = 0.000999

## 3b. Mean-centred cosine permutation tests (primary statistic)

- Observed mean        : 0.7082
- Random null          : mean 0.0635 (SD 0.0238); p = 0.000999
- Thematic null        : mean 0.1100 (SD 0.0192); p = 0.000999
- Length-matched null  : mean 0.0856 (SD 0.0218); p = 0.000999
- Length+theme null    : mean 0.1316 (SD 0.0196); p = 0.000999

- Length mismatch |log(len_Matt/len_Luke)|: matched pairs mean 0.222, mismatched pairs mean 0.847; Spearman rho between centred cosine and length mismatch over mismatched pairs = -0.171 (p = 4.1e-18).

## 3c. Mean-centred cosine permutation tests (triple tradition, Matt-Luke)

- Observed mean        : 0.5781
- Random null          : mean 0.0634 (SD 0.0255); p = 0.000999
- Length-matched null  : mean 0.1047 (SD 0.0222); p = 0.000999

## 4. Pericope-specific residual correlation in the triple tradition

What Matthew adds to Mark correlated with what Luke adds to Mark, pericope by pericope (the geometric form of the minor agreements).

- Matched mean (Signature B)          : 0.3638
- Mismatched-pair floor               : 0.0580
- Nearest non-parallel floor          : 0.2725
- Excess over mean floor [95% CI]     : 0.3058 [0.2717, 0.3359]
- Excess over NN floor                : 0.0913

## 4b. Permutation tests on the residual correlation (triple tradition)

- Observed mean        : 0.3638
- Random null          : mean 0.0648 (SD 0.0161); p = 0.000999
- Length-matched null  : mean 0.0631 (SD 0.0147); p = 0.000999

Centroid residual (double tradition) versus Mark-centred cosine:
- residual-correlation matched mean 0.7500, floor 0.2019; Mark-centred cosine matched mean 0.7547, floor 0.2134; Spearman between residual correlation and the primary centred cosine over the 51 pericopes = 0.977

## 5. Double-tradition statistics with and without the Mark-Q overlap units

Units with a Markan parallel to all or part of their text (n = 16): John's preaching (Mark 1:7-8, partial); Temptation narrative (full) (Mark 1:12-13, full); Judging and the speck (Mark 4:24, partial); Mission discourse (Mark 6:7-13, partial); Whoever receives you (Mark 9:37, full); Sign of Jonah (Mark 8:11-12, partial); Lamp on a stand (Mark 4:21, full); Woes against Pharisees (Mark 12:38-39, partial); Fear of God not men (Mark 4:22, partial); Blasphemy Holy Spirit (Mark 3:28-29, full); Mustard seed and leaven (Mark 4:30-32, partial); Conditions of discipleship (Mark 8:34, partial); Salt of the earth (Mark 9:50, full); Law, prophets and divorce (Mark 10:11-12, partial); Scandals, forgiveness, faith (Mark 9:42; 11:23, partial); Day of the Son of Man (Mark 13:21, partial)

| Subset | n | Space | Matched | Mean floor | Excess | NN excess |
|---|---|---|---|---|---|---|
| all | 51 | raw | 0.9504 | 0.8389 | 0.1115 | 0.0454 |
| without overlaps | 35 | raw | 0.9501 | 0.8356 | 0.1145 | 0.0524 |
| overlaps only | 16 | raw | 0.9511 | 0.8463 | 0.1048 | 0.0470 |
- Non-overlap DT minus TT excess (raw): +0.0627 [+0.0485, +0.0756], MWU p = 2.87e-11
| all | 51 | cen | 0.7082 | 0.0518 | 0.6563 | 0.3267 |
| without overlaps | 35 | cen | 0.7049 | 0.0382 | 0.6667 | 0.3658 |
| overlaps only | 16 | cen | 0.7154 | 0.0836 | 0.6318 | 0.3529 |
- Non-overlap DT minus TT excess (cen): +0.1415 [+0.0561, +0.2218], MWU p = 0.000498

## 6. Composite units: range-wise versus single-sequence encoding

| Pericope | Set | Composite side | Centred cos (range-wise, used) | Centred cos (single sequence) | Difference |
|---|---|---|---|---|---|
| John's preaching | double | luke | 0.7673 | 0.8537 | +0.0864 |
| Love of enemies | double | matthew, luke | 0.6521 | 0.7347 | +0.0827 |
| Judging and the speck | double | luke | 0.8618 | 0.8448 | -0.0170 |
| Blind guides | double | matthew | 0.6102 | 0.5843 | -0.0258 |
| Centurion's servant | double | matthew | 0.6238 | 0.6591 | +0.0353 |
| Jesus on John | double | matthew | 0.6710 | 0.7958 | +0.1248 |
| Woes against Pharisees | double | matthew | 0.8028 | 0.8336 | +0.0308 |
| Law, prophets and divorce | double | matthew | 0.6102 | 0.6061 | -0.0041 |
| Scandals, forgiveness, faith | double | matthew | 0.4605 | 0.6046 | +0.1440 |
| Day of the Son of Man | double | luke | 0.6132 | 0.5820 | -0.0312 |
| Rejection at Nazareth | triple | luke | 0.2583 | 0.2943 | +0.0360 |
| Eschatological discourse | triple | matthew | 0.8695 | 0.8741 | +0.0046 |

Corpus-level contrasts under single-sequence encoding of all composites:
- DT minus TT (cen): +0.1404 [+0.0697, +0.2075]; DT minus LK (cen): +0.0783 [+0.0070, +0.1386]
- DT minus TT (raw): +0.0613 [+0.0513, +0.0719]; DT minus LK (raw): +0.0543 [+0.0437, +0.0640]

## 7. Directionality analysis

- Ridge penalty alpha (min total LOO error) : 1.0; effective df of the fit tr(H) = 20.8 of n = 51
- LOO error by alpha: 0.01: 90.323; 0.03: 88.346; 0.1: 83.939; 0.3: 79.263; 1.0: 78.052; 3.0: 82.454; 10.0: 89.931; 30.0: 94.822; 100.0: 97.249
- Exact LOO ridge R^2, Luke | Matthew      : 0.2189
- Exact LOO ridge R^2, Matthew | Luke      : 0.1954
- delta-R2 (L|M minus M|L)                 : +0.0235
- Jackknife SE / 95% CI                    : 0.0137 / [-0.0033, +0.0503]
- Sign-flip permutation p (two-sided)      : 0.004995
- delta-R2 and sign-flip p across the alpha grid: 0.01: +0.0570 (p = 0.095); 0.03: +0.0530 (p = 0.081); 0.1: +0.0453 (p = 0.036); 0.3: +0.0363 (p = 0.003); 1.0: +0.0235 (p = 0.003); 3.0: +0.0107 (p = 0.027); 10.0: +0.0021 (p = 0.239); 30.0: +0.0000 (p = 0.919); 100.0: -0.0001 (p = 0.261)

Controls:
- Length-balanced double-tradition pairs (|log length ratio| < 0.25, n = 30): delta-R2 = +0.0143, sign-flip p = 0.382
- Known direction, triple tradition (n = 45): Mark -> Matthew delta-R2 = +0.0060 (p = 0.615); Mark -> Luke delta-R2 = -0.0023 (p = 0.839); Matthew vs Luke (both copying Mark) delta-R2 = -0.0061 (p = 0.583)

Residual-norm diagnostic (NOT used as evidence of directionality):
- Residual norm, Matthew mean 0.4438, Luke mean 0.4277; Wilcoxon W = 355.0, p = 0.00389; residual norm vs length (pooled): Spearman rho = -0.823 (p = 2.4e-26)

## 8. Centred-cosine ranking (primary ranking statistic)

- Mean (SD, sample)    : 0.7082 (0.1751)
- Range                : [0.1416, 0.9787]
Top five:
  1. Serving two masters            centred=0.9787 raw=0.9945 jaccard=0.9643
  2. Harvest plentiful              centred=0.9547 raw=0.9896 jaccard=0.7000
  3. Lament over Jerusalem          centred=0.9437 raw=0.9921 jaccard=0.7302
  4. Hidden from wise revealed      centred=0.9146 raw=0.9882 jaccard=0.6628
  5. Return of unclean spirit       centred=0.9083 raw=0.9874 jaccard=0.7286
Bottom five:
  47. Scandals, forgiveness, faith   centred=0.4605 raw=0.9530 jaccard=0.2412
  48. Not peace but sword            centred=0.4381 raw=0.8700 jaccard=0.1379
  49. Judging twelve tribes          centred=0.4347 raw=0.8847 jaccard=0.2857
  50. Parable of Great Banquet       centred=0.2047 raw=0.9346 jaccard=0.1377
  51. Narrow gate                    centred=0.1416 raw=0.8245 jaccard=0.1231

### 8b. Confound analyses on the centred cosine

- Length (characters): Pearson r = 0.002 (p = 0.988); Spearman rho = 0.020 (p = 0.890)
- Form proverbial : mean = 0.6826, median = 0.6814, n = 25
- Form discourse  : mean = 0.7854, median = 0.8028, n = 15
- Form parable    : mean = 0.5745, median = 0.6072, n = 6
- Form narrative  : mean = 0.7426, median = 0.7472, n = 4
- Form liturgical : mean = 0.8537, median = 0.8537, n = 1
- Form Kruskal-Wallis (n >= 3 groups): H = 6.414, p = 0.093
- Pairwise Mann-Whitney, Holm-adjusted p: discourse vs narrative: 0.942; discourse vs proverbial: 0.369; discourse vs parable: 0.140; narrative vs proverbial: 0.942; narrative vs parable: 0.686; proverbial vs parable: 0.942
- Strata counts: {'Q1': 19, 'Q2': 26, 'Q3': 1, 'U': 5}; Q1 mean = 0.7142 (n = 19), Q2 mean = 0.7125 (n = 26); Mann-Whitney U = 249.0, p = 0.973
- Goulder set: mean = 0.6067 (n = 10) vs 0.7329 (n = 41); Welch t = -1.488, p = 0.1666, Cohen's d = -0.745
- Word overlap (multiset Jaccard): Pearson r = 0.781 (p = 1.37e-11), Spearman rho = 0.870; unexplained variance ~ 39%

### 8c. Sentence-level bootstrap on the centred cosine

- Pericopes with >= 2 sentences in both versions: 47 (excluded: Golden rule, Whoever receives you, Lamp on a stand, Blasphemy Holy Spirit)
- Per-pericope bootstrap SD (200 resamples): mean 0.1334, median 0.1174
