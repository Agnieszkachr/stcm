"""
stcm/calibration.py
===================
Triple-tradition calibration module.

The core idea
-------------
Triple-tradition pericopes appear in Matthew, Mark, and Luke.  Under the
Two-Source Hypothesis (2SH), Matthew and Luke each depend on Mark.  We can
therefore define two *embedding signatures*:

Signature A — Independent tradition (baseline)
    Computed from the three-way spread of Matt / Mark / Luke embeddings
    in triple-tradition pericopes.  Captures typical between-gospel cosine
    distances when texts share a common source.

Signature B — Dependence signature
    Computed from the *residual* of Matthew relative to Mark, and Luke
    relative to Mark, in triple-tradition pericopes.  Represents the
    directional transform each evangelist applies to their Markan source.

Both signatures are used in scoring.py to evaluate double-tradition (Q)
pericopes.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from stcm.config import CalibrationConfig, default_config
from stcm.data_loader import Pericope, SynopticCorpus
from stcm.embeddings import EmbeddingPipeline
from stcm.utils import (
    angular_distance,
    bootstrap_ci,
    cosine_similarity,
    residual_vector,
    save_pickle,
    load_pickle,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SignatureStats:
    """Summary statistics for a calibration signature."""
    mean: float
    std: float
    median: float
    ci_lower: float
    ci_upper: float
    n: int
    raw_values: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n": self.n,
        }


@dataclass
class CalibrationResult:
    """Full calibration output."""
    # Signature A: cosine similarities among triple-tradition triples
    sig_a: SignatureStats
    # Signature B: cosine similarities of Matt vs Luke residuals (w.r.t. Mark)
    sig_b: SignatureStats
    # Per-pericope details
    pericope_labels: List[str]
    matt_mark_sims: np.ndarray   # cosine(Matt, Mark) per triple-tradition pericope
    luke_mark_sims: np.ndarray   # cosine(Luke, Mark)
    matt_luke_sims: np.ndarray   # cosine(Matt, Luke)
    residual_sims: np.ndarray    # cosine(resid_Matt_vs_Mark, resid_Luke_vs_Mark)
    # Mean embedding vectors per gospel (centroid of triple-tradition pericopes).
    # NOTE: centroid_mark is used in scoring.py for double-tradition
    # pericopes where no Markan parallel exists.  It represents the mean
    # direction of all Mark embeddings in the triple tradition — a proxy for
    # "the Markan direction in embedding space."  Residuals computed relative
    # to this centroid in the double tradition capture what Matthew and Luke
    # contribute beyond that average Markan baseline direction.
    centroid_matt: np.ndarray = field(repr=False)
    centroid_mark: np.ndarray = field(repr=False)
    centroid_luke: np.ndarray = field(repr=False)
    # Mean residual transform vectors
    mean_matt_residual: np.ndarray = field(repr=False)
    mean_luke_residual: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Calibration engine
# ---------------------------------------------------------------------------

class CalibrationEngine:
    """
    Compute calibration signatures from triple-tradition pericopes.

    Parameters
    ----------
    pipeline : EmbeddingPipeline
    config   : CalibrationConfig
    """

    def __init__(
        self,
        pipeline: Optional[EmbeddingPipeline] = None,
        config: Optional[CalibrationConfig] = None,
    ) -> None:
        self._pipe = pipeline or EmbeddingPipeline()
        self._cfg = config or default_config.calibration
        self._rng = np.random.default_rng(self._cfg.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_independent_signature(
        self, pericopes: List[Pericope]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Compute Matt–Mark, Luke–Mark, and Matt–Luke cosine similarities
        for each triple-tradition pericope.

        Returns
        -------
        (mm_sims, lm_sims, ml_sims, labels)
        """
        triple = [p for p in pericopes if p.tradition == "triple"
                  and p.matthew and p.mark and p.luke]
        log.info("Computing independent signature on %d pericopes.", len(triple))
        mm_sims, lm_sims, ml_sims, labels = [], [], [], []
        for p in triple:
            e_m = self._pipe.embed_text(p.matthew)
            e_mk = self._pipe.embed_text(p.mark)
            e_l = self._pipe.embed_text(p.luke)
            mm_sims.append(cosine_similarity(e_m, e_mk))
            lm_sims.append(cosine_similarity(e_l, e_mk))
            ml_sims.append(cosine_similarity(e_m, e_l))
            labels.append(p.label)
            log.debug("  %s: Matt-Mark=%.3f Luke-Mark=%.3f Matt-Luke=%.3f",
                      p.label, mm_sims[-1], lm_sims[-1], ml_sims[-1])
        return (
            np.array(mm_sims), np.array(lm_sims), np.array(ml_sims), labels
        )

    def compute_dependence_signature(
        self, pericopes: List[Pericope]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute residual vectors of Matt and Luke relative to Mark, then
        measure their cosine similarity.

        Returns
        -------
        (residual_sims, mean_matt_residual, mean_luke_residual)

        Notes
        -----
        Residual of Matt relative to Mark = component of emb(Matt) orthogonal
        to emb(Mark).  This captures what Matthew *adds* beyond Mark.
        If Matthew and Luke independently drew on Q, their residuals (relative
        to their shared Markan source) should be correlated.
        """
        triple = [p for p in pericopes if p.tradition == "triple"
                  and p.matthew and p.mark and p.luke]
        log.info("Computing dependence signature on %d pericopes.", len(triple))
        resid_sims, matt_residuals, luke_residuals = [], [], []
        for p in triple:
            e_m = self._pipe.embed_text(p.matthew)
            e_mk = self._pipe.embed_text(p.mark)
            e_l = self._pipe.embed_text(p.luke)
            r_matt = residual_vector(e_m, e_mk)
            r_luke = residual_vector(e_l, e_mk)
            sim = cosine_similarity(r_matt, r_luke)
            resid_sims.append(sim)
            matt_residuals.append(r_matt)
            luke_residuals.append(r_luke)
        mean_mr = np.mean(matt_residuals, axis=0).astype(np.float32)
        mean_lr = np.mean(luke_residuals, axis=0).astype(np.float32)
        return np.array(resid_sims), mean_mr, mean_lr

    def estimate_transform_vectors(
        self, pericopes: List[Pericope]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean embedding centroids for Matt, Mark, Luke across
        triple-tradition pericopes.

        Returns
        -------
        (centroid_matt, centroid_mark, centroid_luke)
        """
        triple = [p for p in pericopes if p.tradition == "triple"
                  and p.matthew and p.mark and p.luke]
        matt_vecs, mark_vecs, luke_vecs = [], [], []
        for p in triple:
            matt_vecs.append(self._pipe.embed_text(p.matthew))
            mark_vecs.append(self._pipe.embed_text(p.mark))
            luke_vecs.append(self._pipe.embed_text(p.luke))
        c_m = np.mean(matt_vecs, axis=0).astype(np.float32)
        c_mk = np.mean(mark_vecs, axis=0).astype(np.float32)
        c_l = np.mean(luke_vecs, axis=0).astype(np.float32)
        return c_m, c_mk, c_l

    def cross_validate_signatures(
        self, pericopes: List[Pericope], n_folds: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        5-fold cross-validation of Signature A (Matt–Luke cosine).

        Returns
        -------
        dict with keys 'fold_means' and 'fold_stds'
        """
        triple = [p for p in pericopes if p.tradition == "triple"
                  and p.matthew and p.mark and p.luke]
        n = len(triple)
        if n < n_folds:
            log.warning("Too few pericopes for %d-fold CV; skipping.", n_folds)
            return {"fold_means": np.array([]), "fold_stds": np.array([])}
        indices = np.arange(n)
        self._rng.shuffle(indices)
        fold_size = n // n_folds
        fold_means, fold_stds = [], []
        for k in range(n_folds):
            test_idx = indices[k * fold_size : (k + 1) * fold_size]
            test = [triple[i] for i in test_idx]
            sims = []
            for p in test:
                e_m = self._pipe.embed_text(p.matthew)
                e_l = self._pipe.embed_text(p.luke)
                sims.append(cosine_similarity(e_m, e_l))
            fold_means.append(np.mean(sims))
            fold_stds.append(np.std(sims))
        log.info("CV fold means: %s", [f"{m:.3f}" for m in fold_means])
        return {
            "fold_means": np.array(fold_means),
            "fold_stds": np.array(fold_stds),
        }

    def _make_sig_stats(
        self, values: np.ndarray, label: str
    ) -> SignatureStats:
        ci_lo, ci_hi = bootstrap_ci(
            values,
            n_bootstrap=self._cfg.n_bootstrap,
            ci_level=self._cfg.ci_level,
            rng=self._rng,
        )
        ss = SignatureStats(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            median=float(np.median(values)),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n=len(values),
            raw_values=values,
        )
        log.info(
            "%s: mean=%.4f std=%.4f 95%%CI=[%.4f, %.4f] n=%d",
            label, ss.mean, ss.std, ss.ci_lower, ss.ci_upper, ss.n,
        )
        return ss

    def calibrate(self, corpus: SynopticCorpus) -> CalibrationResult:
        """
        Run full calibration on a SynopticCorpus.

        Parameters
        ----------
        corpus : SynopticCorpus

        Returns
        -------
        CalibrationResult
        """
        pericopes = corpus.all_pericopes
        n_triple = len([p for p in pericopes if p.tradition == "triple"])
        if n_triple < self._cfg.min_pericopes:
            raise ValueError(
                f"Only {n_triple} triple-tradition pericopes found; "
                f"minimum is {self._cfg.min_pericopes}."
            )

        # Signature A
        mm_sims, lm_sims, ml_sims, labels = self.compute_independent_signature(
            pericopes
        )
        # Use Matt–Luke as the primary independent-tradition metric
        sig_a = self._make_sig_stats(ml_sims, "Signature A (Matt–Luke, triple)")

        # Signature B
        resid_sims, mean_mr, mean_lr = self.compute_dependence_signature(pericopes)
        sig_b = self._make_sig_stats(resid_sims, "Signature B (residual similarity)")

        # Centroids
        c_m, c_mk, c_l = self.estimate_transform_vectors(pericopes)

        result = CalibrationResult(
            sig_a=sig_a,
            sig_b=sig_b,
            pericope_labels=labels,
            matt_mark_sims=mm_sims,
            luke_mark_sims=lm_sims,
            matt_luke_sims=ml_sims,
            residual_sims=resid_sims,
            centroid_matt=c_m,
            centroid_mark=c_mk,
            centroid_luke=c_l,
            mean_matt_residual=mean_mr,
            mean_luke_residual=mean_lr,
        )
        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def calibrate_and_save(
    corpus: SynopticCorpus,
    pipeline: Optional[EmbeddingPipeline] = None,
    config: Optional[CalibrationConfig] = None,
    out_path: Optional[pathlib.Path] = None,
) -> CalibrationResult:
    """
    Run calibration and pickle the result.

    Parameters
    ----------
    corpus   : SynopticCorpus
    pipeline : EmbeddingPipeline (or None to build a new one)
    config   : CalibrationConfig (or None for default)
    out_path : where to save; defaults to outputs/models/calibration_signatures.pkl

    Returns
    -------
    CalibrationResult
    """
    out_path = out_path or (
        default_config.paths.outputs_models / "calibration_signatures.pkl"
    )
    engine = CalibrationEngine(pipeline=pipeline, config=config)
    result = engine.calibrate(corpus)
    save_pickle(result, out_path)
    log.info("Calibration result saved to %s", out_path)
    return result
