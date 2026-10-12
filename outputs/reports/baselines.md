# Lexical baselines

(n_double = 51, n_triple = 45, permutations = 1000, seed = 42)

## Jaccard

| Set | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess [95% CI] |
|---|---|---|---|---|---|
| Double tradition, Matt-Luke | 0.3882 | 0.0756 | 0.1452 | 0.3125 [0.2642, 0.3647] | 0.2430 [0.1915, 0.2981] |
| Triple tradition, Matt-Luke | 0.2858 | 0.1189 | 0.1860 | 0.1669 [0.1454, 0.1874] | 0.0999 [0.0767, 0.1232] |
| Triple tradition, Matt-Mark | 0.3730 | 0.1217 | 0.1866 | 0.2513 [0.2205, 0.2823] | 0.1864 [0.1559, 0.2173] |
| Triple tradition, Luke-Mark | 0.3323 | 0.1181 | 0.1812 | 0.2141 [0.1869, 0.2448] | 0.1510 [0.1245, 0.1815] |

- DT minus TT excess: +0.1456 [+0.0911, +0.2051], MWU p = 5.74e-05; NN excess +0.1431 [+0.0902, +0.2020]
- DT minus LK excess: +0.0984 [+0.0414, +0.1616], MWU p = 0.0199; NN excess +0.0920 [+0.0352, +0.1557]
- DT minus MK excess: +0.0612 [+0.0026, +0.1262], MWU p = 0.322; NN excess +0.0566 [-0.0010, +0.1204]

## Jaccard: permutation tests (double tradition)

- Observed mean        : 0.3882
- Random null          : mean 0.0816 (SD 0.0078); p = 0.000999
- Thematic null        : mean 0.0814 (SD 0.0037); p = 0.000999
- Length-matched null  : mean 0.0919 (SD 0.0034); p = 0.000999
- Length+theme null    : mean 0.0920 (SD 0.0032); p = 0.000999

- Rank agreement with the centred cosine: Spearman rho = 0.870 (double tradition), 0.733 (triple tradition); Pearson r = 0.781 (double tradition)

## TF-IDF char 3-5

| Set | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess [95% CI] |
|---|---|---|---|---|---|
| Double tradition, Matt-Luke | 0.5451 | 0.0680 | 0.1481 | 0.4771 [0.4230, 0.5309] | 0.3970 [0.3383, 0.4609] |
| Triple tradition, Matt-Luke | 0.4183 | 0.1156 | 0.1946 | 0.3027 [0.2724, 0.3325] | 0.2237 [0.1900, 0.2553] |
| Triple tradition, Matt-Mark | 0.5484 | 0.1201 | 0.2042 | 0.4283 [0.3946, 0.4627] | 0.3442 [0.3088, 0.3810] |
| Triple tradition, Luke-Mark | 0.4726 | 0.1154 | 0.1871 | 0.3572 [0.3234, 0.3945] | 0.2854 [0.2523, 0.3216] |

- DT minus TT excess: +0.1744 [+0.1116, +0.2450], MWU p = 3.57e-05; NN excess +0.1733 [+0.1095, +0.2429]
- DT minus LK excess: +0.1199 [+0.0548, +0.1906], MWU p = 0.0077; NN excess +0.1115 [+0.0490, +0.1815]
- DT minus MK excess: +0.0488 [-0.0170, +0.1206], MWU p = 0.34; NN excess +0.0528 [-0.0132, +0.1241]

## TF-IDF char 3-5: permutation tests (double tradition)

- Observed mean        : 0.5451
- Random null          : mean 0.0770 (SD 0.0104); p = 0.000999
- Thematic null        : mean 0.0752 (SD 0.0039); p = 0.000999
- Length-matched null  : mean 0.0704 (SD 0.0032); p = 0.000999
- Length+theme null    : mean 0.0765 (SD 0.0035); p = 0.000999

- Rank agreement with the centred cosine: Spearman rho = 0.808 (double tradition), 0.640 (triple tradition); Pearson r = 0.771 (double tradition)

## Embedding-specific increment

- Centred cosine ~ Jaccard over all 2601 Matthew-Luke pairs: slope 1.513, intercept -0.059, r = 0.501 (r^2 = 0.251)
- Residual matrix: matched mean +0.1800, mismatched mean -0.0036, excess +0.1836, NN excess -0.1223
## Permutation tests on the residualised centred cosine

- Observed mean        : 0.1800
- Random null          : mean -0.0003 (SD 0.0185); p = 0.000999
- Thematic null        : mean 0.0460 (SD 0.0180); p = 0.000999
- Length-matched null  : mean 0.0067 (SD 0.0204); p = 0.000999
- Length+theme null    : mean 0.0498 (SD 0.0177); p = 0.000999

- Centred cosine ~ TF-IDF over all pairs: r = 0.500; residual matched mean +0.1014, mismatched -0.0020, excess +0.1035; random-null p = 0.000999, length+theme p = 0.033

- Triple tradition: centred cosine ~ Jaccard r = 0.597; residual matched +0.0968, mismatched -0.0022, random-null p = 0.000999

- Residualised DT minus TT excess: +0.0846 [+0.0205, +0.1465], MWU p = 0.00237

Matched pairs: centred cosine regressed on Jaccard (r = 0.781). Pericopes the embedding rates lower than their word overlap predicts:
  Narrow gate                    centred 0.142 (rank 51), Jaccard 0.123 (rank 50), residual -0.383
  Parable of Great Banquet       centred 0.205 (rank 50), Jaccard 0.138 (rank 49), residual -0.330
  Judging twelve tribes          centred 0.435 (rank 49), Jaccard 0.286 (rank 33), residual -0.202
  Jesus on John                  centred 0.671 (rank 33), Jaccard 0.547 (rank 13), residual -0.147
  Scandals, forgiveness, faith   centred 0.461 (rank 47), Jaccard 0.241 (rank 40), residual -0.146
  John's preaching               centred 0.767 (rank 20), Jaccard 0.682 (rank 6), residual -0.144
Pericopes the embedding rates higher than their word overlap predicts:
  Lord's Prayer                  centred 0.854 (rank 12), Jaccard 0.346 (rank 24), residual +0.175
  Conditions of discipleship     centred 0.714 (rank 26), Jaccard 0.154 (rank 46), residual +0.168
  Woes against Pharisees         centred 0.803 (rank 18), Jaccard 0.324 (rank 28), residual +0.139
  One taken, one left            centred 0.686 (rank 30), Jaccard 0.159 (rank 45), residual +0.137
  Golden rule                    centred 0.779 (rank 19), Jaccard 0.308 (rank 29), residual +0.127
  Mission discourse              centred 0.753 (rank 22), Jaccard 0.298 (rank 31), residual +0.108

Largest rank shifts between the two measures:
  Jesus on John                  centred rank 33, Jaccard rank 13
  Conditions of discipleship     centred rank 26, Jaccard rank 46
  Judging twelve tribes          centred rank 49, Jaccard rank 33
  One taken, one left            centred rank 30, Jaccard rank 45
  John's preaching               centred rank 20, Jaccard rank  6
  Centurion's servant            centred rank 39, Jaccard rank 26
  Lord's Prayer                  centred rank 12, Jaccard rank 24
  Golden rule                    centred rank 19, Jaccard rank 29
