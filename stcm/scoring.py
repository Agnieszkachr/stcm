"""
stcm/scoring.py
===============
Double-tradition (Q) scoring module.

For each double-tradition pericope (Matt + Luke, no Mark), the scorer
computes a *Q-score*: a composite measure of how well the Matt–Luke
embedding relationship matches the calibrated dependence signatures from
the triple tradition.

Interpretation
--------------
High Q-score  → Matt–Luke similarity pattern matches the calibrated
                independent-source signature (consistent with shared Q source)
Low Q-score   → pattern is inconsistent with a shared written source
                (could indicate oral tradition, independent composition, or
                late harmonisation)

The scorer does NOT claim to prove Q exists — it quantifies the degree to
which the embedding geometry supports the Q hypothesis relative to a
calibrated baseline.
"""
from __future__ import annotations

import csv
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stcm.calibration import CalibrationResult
from stcm.config import ScoringConfig, default_config
from stcm.data_loader import Pericope, SynopticCorpus
from stcm.embeddings import EmbeddingPipeline
from stcm.utils import angular_distance, cosine_similarity, residual_vector

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PericopyScore:
    """Score for a single double-tradition pericope."""
    label: str
    matt_luke_cos: float      # raw cosine similarity
    matt_luke_ang: float      # angular distance
    # How far is this pericope's cosine from the calibrated mean?
    deviation_from_sig_a: float
    # Similarity of residuals (Matt-vs-centroid, Luke-vs-centroid)
    residual_sim: float
    # Final composite Q-score (higher = more Q-like)
    q_score: float
    # Normalised Q-score in [0, 1]
    q_score_norm: float = 0.0
    # Bayesian posterior (if enabled)
    bayesian_score: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "matt_luke_cos": round(self.matt_luke_cos, 4),
            "matt_luke_ang": round(self.matt_luke_ang, 4),
            "deviation_from_sig_a": round(self.deviation_from_sig_a, 4),
            "residual_sim": round(self.residual_sim, 4),
            "q_score": round(self.q_score, 4),
            "q_score_norm": round(self.q_score_norm, 4),
        }


@dataclass
class ScoringReport:
    """Full scoring report for the double tradition."""
    scores: List[PericopyScore]
    calibration: CalibrationResult
    config: ScoringConfig = field(repr=False)

    @property
    def q_scores(self) -> np.ndarray:
        return np.array([s.q_score for s in self.scores])

    @property
    def q_scores_norm(self) -> np.ndarray:
        return np.array([s.q_score_norm for s in self.scores])

    def top_k(self, k: int = 10) -> List[PericopyScore]:
        """Return k pericopes with highest Q-scores."""
        return sorted(self.scores, key=lambda s: s.q_score, reverse=True)[:k]

    def bottom_k(self, k: int = 10) -> List[PericopyScore]:
        """Return k pericopes with lowest Q-scores."""
        return sorted(self.scores, key=lambda s: s.q_score)[:k]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class QScorer:
    """
    Score double-tradition pericopes against calibration signatures.

    Parameters
    ----------
    pipeline    : EmbeddingPipeline
    calibration : CalibrationResult from the triple-tradition calibration
    config      : ScoringConfig
    """

    def __init__(
        self,
        calibration: CalibrationResult,
        pipeline: Optional[EmbeddingPipeline] = None,
        config: Optional[ScoringConfig] = None,
    ) -> None:
        self._cal = calibration
        self._pipe = pipeline or EmbeddingPipeline()
        self._cfg = config or default_config.scoring

    # ------------------------------------------------------------------
    # Individual pericope scoring
    # ------------------------------------------------------------------

    def score_pericope(self, pericope: Pericope) -> PericopyScore:
        """
        Score a single double-tradition pericope.

        Parameters
        ----------
        pericope : Pericope with tradition == 'double' and matthew + luke set

        Returns
        -------
        PericopyScore
        """
        if not pericope.matthew or not pericope.luke:
            raise ValueError(f"Pericope '{pericope.label}' missing Matthew or Luke text.")

        e_m = self._pipe.embed_text(pericope.matthew)
        e_l = self._pipe.embed_text(pericope.luke)

        # Raw similarity
        cos = cosine_similarity(e_m, e_l)
        ang = angular_distance(e_m, e_l)

        # Deviation from Signature A mean (triple-tradition Matt–Luke baseline)
        dev_a = cos - self._cal.sig_a.mean

        # Residual similarity relative to the Mark centroid.
        # NOTE: For double-tradition pericopes, Mark
        # is by definition absent.  The centroid_mark used here is the mean
        # embedding of Mark's triple-tradition pericopes, computed during
        # calibration.  It serves as a proxy for the "Markan direction" in
        # embedding space — i.e., the typical location of synoptic narrative
        # as shaped by Mark.  The residual relative to this centroid captures
        # what Matthew and Luke contribute *beyond* that Markan baseline
        # direction.  If these residuals are correlated, the "surplus" content
        # in both gospels points in the same direction, consistent with a
        # shared non-Markan source.  This is an approximation: the centroid
        # is a global average, not a pericope-specific Markan parallel.
        r_m = residual_vector(e_m, self._cal.centroid_mark)
        r_l = residual_vector(e_l, self._cal.centroid_mark)
        resid_sim = cosine_similarity(r_m, r_l)

        # Composite Q-score (2-component model: w1=0.8, w3=0.2)
        # Logic:
        #   Higher cosine → more similar → more Q-like
        #   Higher residual sim → correlated "surplus" content
        q = (
            0.8 * cos                        # raw similarity component
            + 0.2 * max(0.0, resid_sim)      # residual correlation bonus
        )

        score = PericopyScore(
            label=pericope.label,
            matt_luke_cos=cos,
            matt_luke_ang=ang,
            deviation_from_sig_a=dev_a,
            residual_sim=resid_sim,
            q_score=q,
        )

        if self._cfg.use_bayesian:
            score.bayesian_score = self._bayesian_score(cos)

        log.debug(
            "  %-40s cos=%.3f dev_a=%+.3f resid=%.3f Q=%.3f",
            pericope.label, cos, dev_a, resid_sim, q,
        )
        return score

    def _bayesian_score(self, obs_cos: float) -> float:
        """
        Bayesian posterior probability that this pericope comes from the
        calibrated Q distribution rather than the null (random) distribution.

        Simplified: likelihood ratio using Gaussian approximation.
        """
        from scipy import stats  # type: ignore
        mu_q = self._cal.sig_a.mean
        sigma_q = max(self._cal.sig_a.std, 1e-4)
        # Null: cosine sims expected ~0 for independent random texts
        mu_null = 0.0
        sigma_null = 0.2

        like_q = stats.norm.pdf(obs_cos, mu_q, sigma_q)
        like_null = stats.norm.pdf(obs_cos, mu_null, sigma_null)
        w = self._cfg.prior_weight
        posterior = (like_q * (1 - w)) / (like_q * (1 - w) + like_null * w + 1e-12)
        return float(posterior)

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def compute_q_score(self) -> None:
        """Alias kept for API compatibility; use batch_score_double_tradition."""
        pass

    def batch_score_double_tradition(
        self, corpus: SynopticCorpus
    ) -> ScoringReport:
        """
        Score all double-tradition pericopes in the corpus.

        Parameters
        ----------
        corpus : SynopticCorpus

        Returns
        -------
        ScoringReport
        """
        double = [p for p in corpus.double_tradition if p.matthew and p.luke]
        log.info("Scoring %d double-tradition pericopes …", len(double))
        raw_scores: List[PericopyScore] = [self.score_pericope(p) for p in double]

        # Normalise Q-scores to [0, 1]
        qs = np.array([s.q_score for s in raw_scores])
        q_min, q_max = qs.min(), qs.max()
        span = q_max - q_min if q_max > q_min else 1.0
        for s in raw_scores:
            s.q_score_norm = float((s.q_score - q_min) / span)

        report = ScoringReport(
            scores=raw_scores,
            calibration=self._cal,
            config=self._cfg,
        )
        log.info(
            "Q-score distribution: mean=%.4f std=%.4f min=%.4f max=%.4f",
            qs.mean(), qs.std(), qs.min(), qs.max(),
        )
        return report

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def generate_distribution_report(
        self,
        report: ScoringReport,
        csv_path: Optional[pathlib.Path] = None,
        fig_path: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Save CSV report and histogram figure.

        Parameters
        ----------
        report   : ScoringReport
        csv_path : path for CSV (default outputs/reports/q_score_distribution.csv)
        fig_path : path for PNG (default outputs/figures/q_score_histogram.png)
        """
        csv_path = csv_path or (
            default_config.paths.outputs_reports / "q_score_distribution.csv"
        )
        fig_path = fig_path or (
            default_config.paths.outputs_figures / "q_score_histogram.png"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV
        fieldnames = list(report.scores[0].as_dict().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for s in report.scores:
                writer.writerow(s.as_dict())
        log.info("CSV saved: %s", csv_path)

        # Histogram
        qs = report.q_scores
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("STCM — Q-Score Distribution (Double Tradition)", fontsize=14, fontweight="bold")

        ax0 = axes[0]
        ax0.hist(qs, bins=self._cfg.histogram_bins, color="#4A90D9", edgecolor="white", alpha=0.85)
        ax0.axvline(self._cal.sig_a.mean, color="#E74C3C", lw=2, linestyle="--",
                    label=f"Triple-tradition baseline (μ={self._cal.sig_a.mean:.3f})")
        ax0.axvline(qs.mean(), color="#2ECC71", lw=2, linestyle="-",
                    label=f"Double-tradition mean (μ={qs.mean():.3f})")
        ax0.set_xlabel("Q-Score")
        ax0.set_ylabel("Count")
        ax0.set_title("Q-Score Histogram")
        ax0.legend(fontsize=8)

        ax1 = axes[1]
        sorted_scores = sorted(report.scores, key=lambda s: s.q_score, reverse=True)
        labels_short = [s.label[:25] for s in sorted_scores]
        vals = [s.q_score for s in sorted_scores]
        colors = ["#E74C3C" if v >= qs.mean() else "#95A5A6" for v in vals]
        y_pos = range(len(labels_short))
        ax1.barh(list(y_pos), vals, color=colors, edgecolor="white", alpha=0.85)
        ax1.set_yticks(list(y_pos))
        ax1.set_yticklabels(labels_short, fontsize=7)
        ax1.invert_yaxis()
        ax1.set_xlabel("Q-Score")
        ax1.set_title("Q-Scores by Pericope")
        ax1.axvline(qs.mean(), color="#2ECC71", lw=1.5, linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Figure saved: %s", fig_path)
