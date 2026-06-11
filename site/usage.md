# Usage

The entire STCM process is driven by `run_pipeline.py`. It is designed to be resumable and modular.

## Standard Run

To execute the entire pipeline from start to finish:

```bash
python run_pipeline.py
```

This will:
1. Load the SBLGNT texts
2. Load the embedding model (or fallback if offline)
3. Calibrate on the Triple Tradition
4. Score the Double Tradition (Q material)
5. Reconstruct latent Q embeddings
6. Run 1000 permutations for robustness evaluation
7. Save all output to `outputs/`

## Speeding up Execution

The permutation tests in the evaluation step take the longest. To skip them during development or quick checks:

```bash
python run_pipeline.py --skip-eval
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
- `reports/q_score_distribution.csv` — Contains the raw scores for every analyzed double-tradition pericope.
- `reports/evaluation_summary.md` — Contains the p-values and z-scores showing statistical significance.
- `figures/q_score_histogram.png` — A visualization of how Q material aligns with the Triple Tradition baseline.
