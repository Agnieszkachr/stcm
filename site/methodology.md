# Methodology

## 1. Data and alignment

Greek text: SBLGNT, with its inline critical-apparatus sigla removed. Pericopes follow the Aland synopsis; the double tradition follows the IQP inventory. Four rules are enforced by `stcm/data_loader.py`:

1. each pair is trimmed to the verses that are actually parallel (non-contiguous ranges are allowed, e.g. Luke 3:7–9 + 16–17);
2. no verse belongs to two units of the same set;
3. no Matthew or Luke verse belongs to both sets; Mark–Q overlaps are excluded from the calibration set;
4. a calibration unit must be the Markan episode itself in each gospel (Luke 5:1–11 is excluded; the Nazareth sermon is restricted to Luke 4:16, 22, 24).

Result: **45 triple-tradition** units (positive control) and **51 double-tradition** units (test set), 16 of which have a Markan parallel to all or part of their text (`MARK_Q_OVERLAP` in `stcm/metadata.py`).

## 2. Representation

- Model: `ABeZet/Koine-Greek-BERT`, pinned to revision `b81172e` (v1.0.1), run under transformers 4.57.3 (tokenisation independent of Python's hash seed). Its normaliser lower-cases and strips diacritics, so inference is on unaccented text. Of the 15,000 added vocabulary entries, 9,257 are reachable whole-word forms and 7,074 are new to the base vocabulary.
- Each contiguous verse range is encoded as one sequence; the passage vector is the attention-masked mean of the last hidden layer, L2-normalised.
- Composite units are encoded **range by range** and averaged with token-count weights, so the model never sees a splice the evangelist did not write. Texts over 512 tokens are chunked at sentence boundaries and combined the same way. Nothing is truncated.
- Cache keys include model, revision, layer setting, accent handling and encoding scheme.

## 3. Primary statistic

The **mean-centred cosine** cos(*x* − *μ*, *y* − *μ*), with *μ* the mean of all 237 pericope vectors. Centring corrects the anisotropy of contextual embeddings (unrelated pairs fall from ≈ 0.84 to ≈ 0.05). The raw cosine is reported in parallel.

## 4. Floors, excess and uncertainty

For each matched pair, the **mean floor** is the average similarity of the left text to all non-parallel right texts, and the **nearest-neighbour floor** the maximum. The evidential quantity is the per-pericope **excess** over the floor, computed for four sets:

| Set | What it calibrates |
|---|---|
| Double tradition, Matt–Luke | the disputed relation |
| Triple tradition, Matt–Luke | two independent copies of one source (two-step) |
| Triple tradition, Luke–Mark and Matt–Mark | a copy and its source (one-step) |

Set means carry bootstrap CIs over pericopes; differences carry a bootstrap CI, a Mann–Whitney test and a label-permutation test. The nearest-neighbour comparison is repeated with the larger set subsampled to the smaller.

## 5. Controls and robustness

- **Permutation nulls** (double tradition): random, thematic (7 categories), length-matched (|log length ratio| ≤ 0.4), length-and-theme-matched; 1,000 draws each.
- **Mark–Q overlaps**: double-tradition statistics with and without the 16 overlap units.
- **Kind of material** (`form_control.py`): 11 sayings-dominant triple-tradition units against the double tradition.
- **Centring vector**: six alternatives (Table 4 of the article).
- **Encoder** (`robustness.py`): Koine-Greek-BERT mean of last four layers; Ancient-Greek-BERT (its pre-training includes PROIEL, hence the Greek NT); Greek BERT (Modern Greek, no biblical pre-training). For each, the residual after word overlap is also tested.
- **Composite encoding**: range-wise vs single-sequence (`composite_sensitivity.csv`).

## 6. Lexical baselines (`baselines.py`)

Multiset Jaccard on accent-stripped word forms and character 3–5-gram TF-IDF cosine are run through the identical floor/excess/null pipeline. The centred-cosine pair matrix is regressed on the Jaccard matrix over all Matthew–Luke pairs and the residual is tested for matched-vs-mismatched separation; the pericopes on which the measures disagree most are listed in `baseline_disagreements.csv`.

## 7. Shared deviation from Mark

For each triple-tradition pericope, the residual of Matthew's vector orthogonal to Mark's is correlated with Luke's; mismatched-pair floor and random/length-matched nulls as above. This is the geometric counterpart of the minor agreements.

## 8. Covariates

Length (characters), five form classes, Kloppenborg strata (pericope-level approximation), the Goulder set, word overlap, and a sentence-level bootstrap (200 resamples; pericopes with fewer than two sentences in either version are excluded).

## 9. Directionality

Kernel ridge with intercept in both directions, exact leave-one-out *R*² via the hat-matrix identity, penalty chosen by LOO error, jackknife CI, sign-flip permutation test, a penalty sweep, a length-balanced subset, and the **known-direction control** (Mark → Matthew, Mark → Luke). The statistic fails that control, so no directional inference is drawn.

