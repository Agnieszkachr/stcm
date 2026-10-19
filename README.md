# STCM — Synoptic Transform Calibration Model

> Testing a transformer representation of Koine Greek against known literary
> dependence in the synoptic gospels, and measuring the double tradition with it.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21761951.svg)](https://doi.org/10.5281/zenodo.21761951)

📖 **[Documentation & Results](https://agnieszkachr.github.io/stcm/)**

The pipeline calibrates a domain-adapted Koine Greek transformer (`ABeZet/Koine-Greek-BERT`, revision `b81172e`) on 45 triple-tradition pericopes — comparing both **two-step copying** (Matthew–Luke) and **one-step copying** (Luke–Mark, Matthew–Mark) as well as the **shared deviation from Mark** (minor agreements) — and evaluates 51 double-tradition pericopes with range-wise encoding of composite units, bootstrap confidence intervals, four permutation nulls, **lexical baselines** (multiset Jaccard and character n-gram TF-IDF), a **sayings-versus-narrative control**, **Mark–Q-overlap flags**, and **three alternative encoder configurations** (last-four-layer mean; Ancient-Greek-BERT; Greek BERT, a Modern Greek model without biblical pre-training).

## What the project shows

1. **The representation passes positive and negative controls.** Where Matthew and Luke both copy Mark, the mean-centred cosine is 0.578 between them and 0.679 / 0.651 between each and Mark; mismatched pairs fall to ≈ 0.05.
2. **The double tradition** (51 pairs, mean 0.708) exceeds random, thematic and length-matched nulls (*p* < 0.001). Measured as excess over each set's own floor it is **closer than two-step Markan copying** (+0.131, 95% CI [0.063, 0.200]; robust to every control) and **at least as close as one-step copying** (Luke–Mark: +0.070 [0.004, 0.141] centred, +0.053 [0.042, 0.063] raw; not stable across encoders). Part of the two-step contrast is a **sayings-versus-narrative** effect.
3. **Most of the pattern is visible in word overlap** (Jaccard ranks correlate at ρ = 0.87 and reproduce the ordering). The embedding's own contribution is a **residual after word overlap** that still separates true parallels (*p* < 0.001); the passages it rates above their word overlap are paraphrases.
4. **Minor agreements:** in the triple tradition, what Matthew and Luke each do to Mark is correlated pericope by pericope (0.364 vs a floor of 0.058, *p* < 0.001).
5. **No directional inference:** the predictability-asymmetry statistic fails its positive control (it does not recover the known direction of Markan copying).

## Key results

| Metric | Value |
|--------|-------|
| Embedding model | `ABeZet/Koine-Greek-BERT` @ `b81172e` (v1.0.1); mean-pooled last layer; diacritics stripped by the model's normaliser |
| Corpus | 51 double-tradition pairs (16 with a Markan parallel, flagged), 45 triple-tradition triples (237 vectors) |
| Calibration (centred) | Matt–Luke 0.578 [0.531, 0.626]; Matt–Mark 0.679 [0.639, 0.716]; Luke–Mark 0.651 [0.608, 0.690] |
| **Centred cosine, double tradition** | matched 0.708 (SD 0.175) vs floor 0.052; nulls: random 0.064, thematic 0.110, length 0.086, length+theme 0.132; all *p* < 0.001 |
| **Excess over own floor (centred)** | DT 0.656 [0.609, 0.702]; TT Matt–Luke 0.525 [0.475, 0.574]; Luke–Mark 0.586 [0.540, 0.635]; Matt–Mark 0.609 [0.570, 0.650] |
| DT − TT (two-step) | centred +0.131 [0.063, 0.200]; raw +0.060 [0.050, 0.071]; without Mark–Q overlaps +0.142; Ancient-Greek-BERT +0.098; Greek BERT (no biblical pre-training) +0.127; last-4 layers +0.126; Jaccard +0.146 (no pre-training at all) |
| DT − Luke–Mark (one-step) | centred +0.070 [0.004, 0.141]; raw +0.053 [0.042, 0.063] |
| Sayings control | DT − TT sayings units (n = 11): centred +0.077 [−0.061, 0.230], raw +0.060 [0.043, 0.078] |
| Lexical baselines | Jaccard excess DT 0.313 vs TT 0.167; Spearman with centred cosine 0.87; residual after Jaccard: matched +0.180 vs mismatched −0.004, *p* < 0.001 |
| Shared deviation from Mark | residual correlation 0.364 vs floor 0.058 (*p* < 0.001) |
| Covariates | length *r* = 0.002; form KW *p* = 0.093; strata *p* = 0.97; Goulder *p* = 0.17 (*d* = −0.75) |
| Sentence bootstrap | median per-pericope SD 0.117 (47 pericopes) |
| Directionality | Δ*R*² = +0.024 (sign-flip *p* = 0.005) but penalty-dependent, absent in length-balanced pairs (*p* = 0.38), and fails the known-direction control (Mark → Matthew *p* = 0.62): **no inference drawn** |

**Top 5 by centred cosine:** Serving Two Masters (0.979), Harvest Plentiful (0.955), Lament over Jerusalem (0.944), Hidden from Wise (0.915), Return of the Unclean Spirit (0.908). **Last:** Narrow Gate (0.142).

![Excess over floor by set](outputs/figures/evaluation/fig2_excess_by_set.png)

## Quickstart

```bash
uv venv && uv pip install -r requirements.txt     # or: pip install -r requirements.txt
python download_sblgnt.py                         # SBLGNT source texts
python populate_cache.py                          # embed everything (4 encoder configurations)
python advanced_analysis.py                       # Tables 1–4 and all embedding statistics
python baselines.py                               # Table 5: lexical baselines
python form_control.py                            # sayings-versus-narrative control
python robustness.py                              # Appendix C2: alternative encoders
python tools/make_article_tables.py               # Appendix Tables A1, B1, C1–C3 (markdown)
python generate_evaluation_figures.py             # Figures 1–5
python -m pytest tests                            # integrity checks
# or all of it:
python run_pipeline.py
```

### Output files

```
outputs/reports/
├── advanced_analysis.md          # Tables 1–4, nulls, shared deviation, overlaps, composite sensitivity, directionality, covariates
├── baselines.md                  # Table 5 and the embedding-specific increment
├── form_control.md               # sayings vs narrative
├── robustness.md                 # alternative encoders
├── centred_cosine_ranking.csv    # Table A1 (all 51, with refs, form, theme, stratum, excess, Jaccard, Markan parallel)
├── triple_tradition_table.csv    # Table B1
├── floor_table.csv, per_pericope_excess.csv, composite_sensitivity.csv, baseline_disagreements.csv
├── calibration_table.csv, pair_values_centred.csv
└── key_stats.json, key_stats_baselines.json, key_stats_form_control.json, key_stats_robustness.json
```

*(Auxiliary outputs `evaluation_summary.md`, `q_score_distribution.csv`, `system_validation.txt`, and `outputs/figures/q_score_histogram.png` belong to the optional composite-index module; they are not tracked in the repository and are generated only when running `python run_pipeline.py --legacy`.)*

## Architecture

```
stcm/
├── stcm/
│   ├── config.py          # configuration (STCM_* env overrides; pinned model revision; layer setting)
│   ├── utils.py           # Greek normalisation, SBLGNT parsing
│   ├── data_loader.py     # alignment tables (multi-range refs, four enforced rules, per-range parts)
│   ├── metadata.py        # themes, forms, strata, Goulder set, Mark–Q overlaps, sayings units, text-critical notes
│   ├── embeddings.py      # encoder: range-wise and chunked encoding, disk cache
│   ├── analysis_core.py   # floors, excess, CIs, nulls, LOO ridge
│   ├── calibration.py     # triple-tradition calibration object
│   └── scoring.py, reconstruction.py, evaluation.py   # optional modules (--legacy)
├── populate_cache.py, advanced_analysis.py, baselines.py, form_control.py, robustness.py
├── tools/make_article_tables.py
├── generate_evaluation_figures.py
├── tests/test_integrity.py
├── site/ (MkDocs source) · docs/ (built site)
└── run_pipeline.py, download_sblgnt.py
```

## Reproducibility

Deterministic given the SBLGNT files, the pinned model revisions (Koine-Greek-BERT `b81172ecb9baf65734c88ab65e2eb2491f12f3de` = v1.0.1, `tokenizer.json` SHA-256 `80c4ce56314d9a035d82645f3eaba11e094c12bb16ee452cabf645b10737d9e3`; Ancient-Greek-BERT `5e3e29ece1d63029baa226f11105b1e8277c4f07`; Greek BERT `ec2b8f88dd215b5246f2f850413d5bff90d7540d`), Python 3.11+ with `requirements.txt`, and seed 42 (resampling and permutations only). The published numbers were computed with transformers 4.57.3 and tokenizers 0.22.2; the v1.0.1 tokenizer gives hash-seed-independent input ids under transformers 4.x and 5.x alike, and two independent runs produced identical vectors. Cache keys include the model, revision, layer setting, accent handling and encoding version, so vectors from any other configuration are never reused.

## Limitations

- Mutual similarity does not discriminate between Q and Luke's use of Matthew; the directionality test fails its control.
- Part of the double-versus-triple contrast reflects sayings versus narrative; with 11 sayings units in the triple tradition the remainder cannot be settled.
- All floors are internal to the synoptic corpus.
- Koine-Greek-BERT and its base model Ancient-Greek-BERT (via PROIEL) were pre-trained on the NT. The double-vs-triple contrast is unaffected (word overlap reproduces it without pre-training); the embedding's increment over word overlap is reproduced by Greek BERT (no biblical pre-training: matched-minus-mismatched residual +0.163), but exposure to gospel text through web corpora cannot be excluded.
- Pericope delimitation and composite encoding move individual ranks (Appendix C of the article), not the corpus-level contrasts.
- Thematic, form and sayings annotations are the author's own.

## Data source

Greek text: [SBLGNT](https://github.com/LogosBible/SBLGNT), under the [SBLGNT EULA](https://www.sblgnt.com/license/).

## Citation

Cite the concept DOI [10.5281/zenodo.21761951](https://doi.org/10.5281/zenodo.21761951), which resolves to the latest release.

```bibtex
@software{stcm2026,
  author    = {Ziemińska, Agnieszka},
  title     = {STCM: Synoptic Transform Calibration Model},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21761951},
  url       = {https://github.com/Agnieszkachr/stcm},
  note      = {Testing a transformer representation of Koine Greek against known literary dependence}
}
```

## License

MIT — see [LICENSE](LICENSE).
