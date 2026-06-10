"""
run_pipeline.py
===============
End-to-end STCM pipeline runner.

Steps executed:
  1. Load & validate SBLGNT corpus
  2. Embed all pericopes
  3. Calibrate on triple tradition
  4. Score double tradition (Q candidates)
  5. Reconstruct latent Q embeddings
  6. Evaluate robustness
  7. Save all outputs and STEP_LOG.md

Usage
-----
    python run_pipeline.py [--skip-eval] [--model ABeZet/Koine-Greek-BERT]

The pipeline is resumable: if outputs/models/calibration_signatures.pkl exists,
calibration is skipped and loaded from disk.
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time

# Ensure the repo root is on sys.path
_REPO = pathlib.Path(__file__).parent
sys.path.insert(0, str(_REPO))

from stcm.config import STCMConfig, default_config
from stcm.data_loader import SBLGNTLoader
from stcm.embeddings import EmbeddingPipeline
from stcm.calibration import calibrate_and_save, CalibrationResult
from stcm.scoring import QScorer
from stcm.reconstruction import reconstruct_and_save
from stcm.evaluation import evaluate_and_save
from stcm.utils import load_pickle, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    log = logging.getLogger("stcm.pipeline")
    sep = "─" * 60
    log.info(sep)
    log.info(msg)
    log.info(sep)


def append_step_log(step: int, description: str, files: list[str]) -> None:
    log_path = _REPO / "logs" / "STEP_LOG.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [
        f"\n## ✅ STEP {step} COMPLETE — {ts}\n",
        f"**{description}**\n",
        "Files created/modified:\n",
    ]
    for f in files:
        lines.append(f"- `{f}`\n")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.writelines(lines)


def last_completed_step() -> int:
    """Read STEP_LOG.md and return the highest completed step number."""
    log_path = _REPO / "logs" / "STEP_LOG.md"
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8")
    import re
    matches = re.findall(r"## ✅ STEP (\d+)", text)
    if not matches:
        return 0
    return max(int(m) for m in matches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="STCM Pipeline Runner")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the evaluation step (faster)")
    parser.add_argument("--model", default=None,
                        help="Override embedding model (e.g. ABeZet/Koine-Greek-BERT)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed steps (detect from STEP_LOG.md)")
    args = parser.parse_args()

    # Setup
    cfg = STCMConfig()
    cfg.paths.ensure_all()
    logger = cfg.setup_logging()
    cfg.seed_everything()

    if args.model:
        cfg.embedding.model_name = args.model

    last_step = last_completed_step() if args.resume else 0

    banner("STCM — Synoptic Transform Calibration Model")
    logger.info("Python %s | resume from step %d", sys.version.split()[0], last_step)
    logger.info("Embedding model: %s", cfg.embedding.model_name)

    # -----------------------------------------------------------------------
    # STEP 1: Load corpus
    # -----------------------------------------------------------------------
    t0 = time.time()
    if last_step < 1:
        banner("STEP 1 — Loading SBLGNT corpus")
        loader = SBLGNTLoader(cfg.paths.data_raw)
        corpus = loader.load()
        logger.info(
            "Corpus: %d triple + %d double tradition pericopes",
            len(corpus.triple_tradition),
            len(corpus.double_tradition),
        )
        append_step_log(1, "SBLGNT corpus loaded", [
            "data/raw/matthew.txt",
            "data/raw/mark.txt",
            "data/raw/luke.txt",
        ])
        logger.info("STEP 1 COMPLETE (%.1f s)", time.time() - t0)
    else:
        logger.info("STEP 1 already complete — loading corpus …")
        loader = SBLGNTLoader(cfg.paths.data_raw)
        corpus = loader.load()

    # -----------------------------------------------------------------------
    # STEP 2: Build embedding pipeline
    # -----------------------------------------------------------------------
    t0 = time.time()
    banner("STEP 2 — Initialising embedding pipeline")
    pipeline = EmbeddingPipeline(config=cfg.embedding, cache_dir=cfg.paths.data_processed)
    logger.info("Embedding pipeline ready. STEP 2 COMPLETE (%.1f s)", time.time() - t0)
    if last_step < 2:
        append_step_log(2, "Embedding pipeline initialised", ["data/processed/ (cache)"])

    # -----------------------------------------------------------------------
    # STEP 3: Calibration (triple tradition)
    # -----------------------------------------------------------------------
    t0 = time.time()
    cal_path = cfg.paths.outputs_models / "calibration_signatures.pkl"
    if last_step >= 3 and cal_path.exists():
        banner("STEP 3 — Loading cached calibration result")
        calibration: CalibrationResult = load_pickle(cal_path)
        logger.info("Loaded calibration from cache.")
    else:
        banner("STEP 3 — Calibrating on triple tradition")
        calibration = calibrate_and_save(
            corpus=corpus,
            pipeline=pipeline,
            config=cfg.calibration,
            out_path=cal_path,
        )
        append_step_log(3, "Triple-tradition calibration complete", [
            "outputs/models/calibration_signatures.pkl",
        ])
    logger.info(
        "Sig-A: mean=%.4f std=%.4f | Sig-B: mean=%.4f std=%.4f",
        calibration.sig_a.mean, calibration.sig_a.std,
        calibration.sig_b.mean, calibration.sig_b.std,
    )
    logger.info("STEP 3 COMPLETE (%.1f s)", time.time() - t0)

    # -----------------------------------------------------------------------
    # STEP 4: Scoring (double tradition)
    # -----------------------------------------------------------------------
    t0 = time.time()
    banner("STEP 4 — Scoring double-tradition pericopes")
    scorer = QScorer(
        calibration=calibration,
        pipeline=pipeline,
        config=cfg.scoring,
    )
    report = scorer.batch_score_double_tradition(corpus)
    scorer.generate_distribution_report(report)
    append_step_log(4, "Double-tradition scoring complete", [
        "outputs/reports/q_score_distribution.csv",
        "outputs/figures/q_score_histogram.png",
    ])
    logger.info("STEP 4 COMPLETE (%.1f s)", time.time() - t0)

    # Print top 10
    logger.info("Top 10 Q-scored pericopes:")
    for i, s in enumerate(report.top_k(10), 1):
        logger.info("  %2d. %-40s Q=%.4f", i, s.label, s.q_score)

    # -----------------------------------------------------------------------
    # STEP 5: Reconstruction
    # -----------------------------------------------------------------------
    t0 = time.time()
    banner("STEP 5 — Reconstructing latent Q embeddings")
    rec_path = cfg.paths.outputs_models / "reconstructed_q_embeddings.pkl"
    reconstruction = reconstruct_and_save(
        corpus=corpus,
        calibration=calibration,
        pipeline=pipeline,
        config=cfg.reconstruction,
        out_path=rec_path,
    )
    logger.info(
        "Reconstruction: conv_rate=%.2f mean_var=%.6f",
        reconstruction.convergence_rate,
        reconstruction.mean_variance,
    )
    append_step_log(5, "Latent Q embedding reconstruction complete", [
        "outputs/models/reconstructed_q_embeddings.pkl",
    ])
    logger.info("STEP 5 COMPLETE (%.1f s)", time.time() - t0)

    # -----------------------------------------------------------------------
    # STEP 6: Evaluation
    # -----------------------------------------------------------------------
    eval_result = None
    if not args.skip_eval:
        t0 = time.time()
        banner("STEP 6 — Robustness evaluation (permutation tests)")
        eval_result = evaluate_and_save(
            report=report,
            corpus=corpus,
            calibration=calibration,
            pipeline=pipeline,
            config=cfg.evaluation,
        )
        logger.info(
            "Permutation test p-value: %.4f (z=%.3f)",
            eval_result.permutation_mean_q.p_value,
            eval_result.permutation_mean_q.z_score,
        )
        append_step_log(6, "Robustness evaluation complete", [
            "outputs/reports/evaluation_summary.md",
        ])
        logger.info("STEP 6 COMPLETE (%.1f s)", time.time() - t0)
    else:
        logger.info("Skipping STEP 6 (--skip-eval set).")

    # -----------------------------------------------------------------------
    # STEP 7: System validation report
    # -----------------------------------------------------------------------
    banner("STEP 7 — Generating system validation report")
    _write_validation_report(
        report, reconstruction, cfg,
        eval_result=eval_result if not args.skip_eval else None,
    )
    append_step_log(7, "System validation report generated", [
        "outputs/reports/system_validation.txt",
    ])

    banner("✅  PIPELINE COMPLETE")
    logger.info(
        "All outputs in: %s", cfg.paths.outputs_reports.parent
    )


def _write_validation_report(report, reconstruction, cfg: STCMConfig,
                              eval_result=None) -> None:
    lines = [
        "STCM System Validation Report",
        "=" * 50,
        f"Embedding model       : {cfg.embedding.model_name}",
        f"Double-tradition n    : {len(report.scores)}",
        f"Q-score mean          : {report.q_scores.mean():.4f}",
        f"Q-score std           : {report.q_scores.std():.4f}",
        f"Q-score range         : [{report.q_scores.min():.4f}, {report.q_scores.max():.4f}]",
        f"Calibration Sig-A mean: {report.calibration.sig_a.mean:.4f}",
        f"Calibration Sig-B mean: {report.calibration.sig_b.mean:.4f}",
        f"Reconstruction conv % : {reconstruction.convergence_rate * 100:.1f}%",
        f"Reconstruction var    : {reconstruction.mean_variance:.6f}",
        "",
        "Top 5 Q pericopes:",
    ]
    for i, s in enumerate(report.top_k(5), 1):
        lines.append(f"  {i}. {s.label:<40} Q={s.q_score:.4f}")

    # Include evaluation suite results if available
    if eval_result is not None:
        lines.append("")
        lines.append("Evaluation Suite Results:")
        lines.append("-" * 50)
        pm = eval_result.permutation_mean_q
        lines.append(f"Random perm. p-value  : {pm.p_value:.4f} (z={pm.z_score:.3f})")
        if eval_result.thematic_null is not None:
            tn = eval_result.thematic_null
            lines.append(f"Thematic-null p-value : {tn.p_value:.4f} (z={tn.z_score:.3f})")
        if eval_result.sensitivity is not None:
            lines.append(f"Weight sensitivity    : top-5 Jaccard = {eval_result.sensitivity.top5_stability:.3f}")
        if eval_result.bootstrap_robustness is not None:
            lines.append(f"Bootstrap mean std    : {eval_result.bootstrap_robustness.mean_bootstrap_std:.4f}")
        if eval_result.word_overlap is not None:
            lines.append(f"Word-overlap Pearson r: {eval_result.word_overlap.correlation:.3f}")
        if eval_result.goulder_test is not None:
            gt = eval_result.goulder_test
            lines.append(f"Goulder test p-value  : {gt.p_value:.4f} (d={gt.effect_size:.3f})")
        if eval_result.bert_validation is not None:
            bv = eval_result.bert_validation
            lines.append(f"BERT validation gap   : {bv.separation:.3f}")

    # Full pericope table
    lines.append("")
    lines.append("All 36 pericopes (ranked by Q-score):")
    lines.append("-" * 50)
    for i, s in enumerate(report.top_k(36), 1):
        lines.append(
            f"  {i:2d}. {s.label:<40} Q={s.q_score:.4f}  "
            f"cos={s.matt_luke_cos:.3f}  resid={s.residual_sim:.3f}"
        )

    out = cfg.paths.outputs_reports / "system_validation.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    logging.getLogger("stcm.pipeline").info("Validation report: %s", out)


if __name__ == "__main__":
    main()
