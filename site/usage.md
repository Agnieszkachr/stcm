# Usage

The entire STCM process is driven by `run_pipeline.py`. It is designed to be resumable and modular. Commands below use `uv run` (see [Installation](installation.md)); with a pip-installed environment, use `python` instead.

## Standard Run

To execute the entire pipeline from start to finish:

```bash
uv run run_pipeline.py
```

This will:
1. Load the SBLGNT texts and check the alignment tables
2. Load the embedding model lazily (only on a cache miss)
3. Calibrate on the Triple Tradition (45 pericopes)
4. Run `advanced_analysis.py` (Tables 1–4: floors, excess with CIs, nulls, shared deviation from Mark, Mark–Q overlaps, composite sensitivity, directionality, covariates, bootstrap)
5. Run `baselines.py` (lexical baselines), `form_control.py` (sayings vs narrative) and `robustness.py` (alternative encoders)
6. Draw Figures 1–5
7. Save all output to `outputs/`

To fill the embedding cache for all four encoder configurations first (the analyses then run without the model, except the sentence bootstrap):

```bash
uv run populate_cache.py
```

## Optional modules

The auxiliary composite Q-score and latent-Q reconstruction modules are not part of the article. They run only on request:

```bash
uv run run_pipeline.py --legacy
```

## Resuming from Cache

If a previous run was interrupted or you only want to re-run the final evaluation steps, use the `--resume` flag. It reads `logs/STEP_LOG.md` to determine the last completed step and picks up from there.

```bash
uv run run_pipeline.py --resume
```

## Overriding Configuration

STCM uses environment variables for configuration. You can change the behavior without editing code:

**Change the embedding model:**
```bash
# Linux / macOS
STCM_EMBEDDING_MODEL=bert-base-multilingual-cased uv run run_pipeline.py

# Windows PowerShell
$env:STCM_EMBEDDING_MODEL="bert-base-multilingual-cased"
uv run run_pipeline.py
```

**Use the mean of the last four layers, or another model revision:**
```bash
STCM_LAYERS=last4 uv run advanced_analysis.py
STCM_MODEL_REVISION=<git sha> uv run advanced_analysis.py
```

**Increase log verbosity:**
```bash
STCM_LOG_LEVEL=DEBUG uv run run_pipeline.py
```

## Understanding Outputs

Look in the `outputs/` directory for your results:
- `reports/advanced_analysis.md` — Tables 1–4 and every embedding statistic in the article.
- `reports/baselines.md`, `reports/form_control.md`, `reports/robustness.md` — Table 5, the form control and the encoder comparison (Appendix C).
- `reports/centred_cosine_ranking.csv` — The per-pericope ranking (refs, stratum, form, theme, raw/centred cosine, excess, lengths, word overlap, Markan parallel, text-critical note).
- `reports/triple_tradition_table.csv`, `per_pericope_excess.csv`, `floor_table.csv`, `composite_sensitivity.csv`, `baseline_disagreements.csv` — appendix tables.
- `reports/key_stats*.json` — machine-readable numbers (the figure script and the tests read these).
- `figures/evaluation/fig1–fig5` — The article figure set.
- `reports/evaluation_summary.md`, `reports/q_score_distribution.csv`, `reports/system_validation.txt`, `figures/q_score_histogram.png` — optional outputs (not tracked in the repository; generated only with `--legacy`).
