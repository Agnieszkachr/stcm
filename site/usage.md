# Usage

The entire STCM process is driven by `run_pipeline.py`. It is designed to be resumable and modular.

## Standard Run

To execute the entire pipeline from start to finish:

```bash
python run_pipeline.py
```

This will:
1. Load the SBLGNT texts and check the alignment tables (no verse in two units; no verse in both sets)
2. Load the embedding model lazily (only on a cache miss)
3. Calibrate on the Triple Tradition (46 pericopes)
4. Run the advanced analysis on the Double Tradition (51 pericopes): centred-cosine statistics, four permutation nulls, floors, covariates, directionality with controls
5. Draw Figures 1–6
6. Save all output to `outputs/`

## Legacy steps

The retired composite Q-score, the latent-Q reconstruction and the old evaluation suite are not part of the article. They run only on request:

```bash
python run_pipeline.py --legacy
```

## Resuming from Cache

If a previous run was interrupted or you only want to re-run the final evaluation steps, use the `--resume` flag. It reads `logs/STEP_LOG.md` to determine the last completed step and picks up from there.

```bash
python run_pipeline.py --resume
```

## Overriding Configuration

STCM uses environment variables for configuration. You can change the behavior without editing code:

**Change the embedding model:**
```bash
# Linux / macOS
STCM_EMBEDDING_MODEL=bert-base-multilingual-cased python run_pipeline.py

# Windows PowerShell
$env:STCM_EMBEDDING_MODEL="bert-base-multilingual-cased"
python run_pipeline.py
```

**Increase log verbosity:**
```bash
STCM_LOG_LEVEL=DEBUG python run_pipeline.py
```

## Understanding Outputs

Look in the `outputs/` directory for your results:
- `reports/advanced_analysis.md` — Every statistic in the article: calibration table, anisotropy table, permutation nulls, genre floor, directionality with controls, covariates, bootstrap.
- `reports/centred_cosine_ranking.csv` — The per-pericope ranking with verse references, stratum, form, lengths and word overlap.
- `reports/key_stats.json` — The handful of numbers the figure script needs.
- `figures/evaluation/fig1–fig6` — The article figure set.
- `reports/evaluation_summary.md`, `reports/q_score_distribution.csv`, `reports/system_validation.txt` — legacy outputs (`--legacy` only).
