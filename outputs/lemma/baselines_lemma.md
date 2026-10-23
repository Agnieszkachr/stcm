# Lemma-based lexical baselines

MorphGNT SBLGNT commit aaed91e57c8e4a8dc9a2383e129ca5e75fe6393d; n_double = 51, n_triple = 45; permutations = 1000; seed = 42. Reading rule: outputs/lemma/READING_RULE.md.

## Jaccard (lemmas)

| Set | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess [95% CI] |
|---|---|---|---|---|---|
| Double tradition, Matt-Luke | 0.4981 | 0.1389 | 0.2423 | 0.3592 [0.3173, 0.4053] | 0.2558 [0.2133, 0.3043] |
| Triple tradition, Matt-Luke | 0.4113 | 0.2011 | 0.3084 | 0.2102 [0.1846, 0.2363] | 0.1030 [0.0752, 0.1277] |
| Triple tradition, Matt-Mark | 0.4943 | 0.2037 | 0.3092 | 0.2906 [0.2557, 0.3246] | 0.1851 [0.1532, 0.2171] |
| Triple tradition, Luke-Mark | 0.4630 | 0.1999 | 0.3049 | 0.2631 [0.2315, 0.2965] | 0.1581 [0.1273, 0.1909] |

- DT minus TT excess: +0.1490 [+0.0980, +0.2036], MWU p = 2.81e-06
- DT minus LK excess: +0.0962 [+0.0436, +0.1562], MWU p = 0.0066
- DT minus MK excess: +0.0686 [+0.0135, +0.1296], MWU p = 0.123

## Jaccard (lemmas): permutation tests (double tradition)

- Observed mean        : 0.4981
- Random null          : mean 0.1458 (SD 0.0095); p = 0.000999
- Thematic null        : mean 0.1478 (SD 0.0061); p = 0.000999
- Length-matched null  : mean 0.1796 (SD 0.0047); p = 0.000999
- Length+theme null    : mean 0.1737 (SD 0.0045); p = 0.000999

## TF-IDF (lemmas)

| Set | Matched | Mean floor | NN floor | Excess [95% CI] | NN excess [95% CI] |
|---|---|---|---|---|---|
| Double tradition, Matt-Luke | 0.6064 | 0.0882 | 0.1986 | 0.5183 [0.4637, 0.5704] | 0.4078 [0.3512, 0.4631] |
| Triple tradition, Matt-Luke | 0.4936 | 0.1351 | 0.2412 | 0.3585 [0.3220, 0.3943] | 0.2524 [0.2121, 0.2901] |
| Triple tradition, Matt-Mark | 0.6289 | 0.1379 | 0.2531 | 0.4909 [0.4559, 0.5309] | 0.3758 [0.3363, 0.4178] |
| Triple tradition, Luke-Mark | 0.5427 | 0.1333 | 0.2383 | 0.4094 [0.3707, 0.4493] | 0.3044 [0.2669, 0.3419] |

- DT minus TT excess: +0.1597 [+0.0961, +0.2286], MWU p = 5.39e-05
- DT minus LK excess: +0.1089 [+0.0445, +0.1783], MWU p = 0.00504
- DT minus MK excess: +0.0273 [-0.0394, +0.0975], MWU p = 0.463

## TF-IDF (lemmas): permutation tests (double tradition)

- Observed mean        : 0.6064
- Random null          : mean 0.0981 (SD 0.0113); p = 0.000999
- Thematic null        : mean 0.0961 (SD 0.0049); p = 0.000999
- Length-matched null  : mean 0.0904 (SD 0.0046); p = 0.000999
- Length+theme null    : mean 0.0955 (SD 0.0044); p = 0.000999

## Embedding residual after lemma overlap (primary test of the reading rule)

### Residual after lemma Jaccard

- R² of the fit over all 2601 pairs: 0.204; coefficients -0.093, +1.082
- Matched mean +0.2623; mismatched mean -0.0052; excess +0.2676
- Random null -0.0002 (SD 0.0190), p = 0.000999; thematic p = 0.000999; length p = 0.000999; length+theme null +0.0353 (SD 0.0182), p = 0.000999
- Triple tradition: matched +0.2148, mismatched -0.0049, random p = 0.000999, length p = 0.000999
- DT minus TT on the residual: +0.0479 [-0.0012, +0.0950], MWU p = 0.0298

### Residual after lemma TF-IDF

- R² of the fit over all 2601 pairs: 0.221; coefficients -0.034, +1.000
- Matched mean +0.1353; mismatched mean -0.0027; excess +0.1380
- Random null -0.0004 (SD 0.0190), p = 0.000999; thematic p = 0.000999; length p = 0.000999; length+theme null +0.0686 (SD 0.0192), p = 0.002
- Triple tradition: matched +0.0249, mismatched -0.0006, random p = 0.103, length p = 0.875
- DT minus TT on the residual: +0.1126 [+0.0498, +0.1732], MWU p = 0.000606

### Residual after forms Jaccard + lemma Jaccard

- R² of the fit over all 2601 pairs: 0.251; coefficients -0.060, +1.482, +0.027
- Matched mean +0.1799; mismatched mean -0.0036; excess +0.1835
- Random null -0.0003 (SD 0.0185), p = 0.000999; thematic p = 0.000999; length p = 0.000999; length+theme null +0.0493 (SD 0.0177), p = 0.000999
- Triple tradition: matched +0.0963, mismatched -0.0022, random p = 0.000999, length p = 0.000999
- DT minus TT on the residual: +0.0850 [+0.0211, +0.1464], MWU p = 0.0021

### Residual after forms Jaccard + char TF-IDF

- R² of the fit over all 2601 pairs: 0.268; coefficients -0.050, +0.823, +0.608
- Matched mean +0.1071; mismatched mean -0.0021; excess +0.1093
- Random null -0.0003 (SD 0.0182), p = 0.000999; thematic p = 0.000999; length p = 0.000999; length+theme null +0.0575 (SD 0.0179), p = 0.00999
- Triple tradition: matched -0.0459, mismatched +0.0010, random p = 0.995, length p = 1
- DT minus TT on the residual: +0.1562 [+0.0926, +0.2182], MWU p = 5.7e-06

### Residual after forms Jaccard + lemma Jaccard + char TF-IDF

- R² of the fit over all 2601 pairs: 0.268; coefficients -0.051, +0.800, +0.020, +0.608
- Matched mean +0.1070; mismatched mean -0.0021; excess +0.1092
- Random null -0.0003 (SD 0.0182), p = 0.000999; thematic p = 0.000999; length p = 0.000999; length+theme null +0.0572 (SD 0.0179), p = 0.00799
- Triple tradition: matched -0.0457, mismatched +0.0010, random p = 0.995, length p = 1
- DT minus TT on the residual: +0.1559 [+0.0931, +0.2179], MWU p = 4.78e-06

## DT minus TT on the residual: separate versus common fits

| Predictor | Slope DT | Slope TT | Slope pooled | Separate fits | DT slope for both | Pooled fit |
|---|---|---|---|---|---|---|
| forms Jaccard | 1.513 | 2.553 | 1.590 | +0.085 [+0.021, +0.147] | -0.089 [-0.151, -0.031] | -0.100 [-0.165, -0.039] |
| lemma Jaccard | 1.082 | 1.453 | 1.048 | +0.048 [-0.001, +0.095] | -0.030 [-0.078, +0.017] | -0.025 [-0.073, +0.022] |

## Residual after lemma Jaccard under the alternative encoders

| Configuration | Matched | Mismatched | Length+theme null | p |
|---|---|---|---|---|
| primary (ABeZet/Koine-Greek-BERT@b81172ecb9baf65734c88ab65e2eb2491f12f3de (last)) | +0.2623 | -0.0052 | +0.0353 | 0.000999 |
| last4 (ABeZet/Koine-Greek-BERT@b81172ecb9baf65734c88ab65e2eb2491f12f3de (last4)) | +0.1982 | -0.0040 | +0.0522 | 0.000999 |
| agb (pranaydeeps/Ancient-Greek-BERT@5e3e29ece1d63029baa226f11105b1e8277c4f07 (last)) | +0.2636 | -0.0053 | +0.0283 | 0.000999 |
| greekbert (nlpaueb/bert-base-greek-uncased-v1@ec2b8f88dd215b5246f2f850413d5bff90d7540d (last)) | +0.1874 | -0.0037 | +0.0408 | 0.000999 |

## Paraphrase cases (matched pairs; residual of the centred cosine on Jaccard)

| Pericope | Jaccard forms (rank) | residual | Jaccard lemmas (rank) | residual | centred rank |
|---|---|---|---|---|---|
| Conditions of discipleship | 0.154 (46) | +0.168 | 0.286 (48) | +0.170 | 26 |
| One taken, one left | 0.159 (45) | +0.137 | 0.378 (38) | +0.071 | 30 |
| Golden rule | 0.308 (29) | +0.127 | 0.360 (41) | +0.178 | 19 |
| Lord's Prayer | 0.346 (24) | +0.175 | 0.438 (27) | +0.192 | 12 |

Spearman, centred cosine vs Jaccard: forms 0.870, lemmas 0.828