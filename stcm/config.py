"""
stcm/config.py
==============
Central configuration for the Synoptic Transform Calibration Model.

All settings can be overridden via environment variables with the prefix STCM_.
Example: STCM_EMBEDDING_MODEL=bert-base-uncased overrides the default model.
"""
from __future__ import annotations

import logging
import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent  # repository root


def _env(key: str, default: str) -> str:
    """Read an STCM_ environment variable with a fallback default."""
    return os.environ.get(f"STCM_{key}", default)


@dataclass
class Paths:
    root: pathlib.Path = _ROOT
    data_raw: pathlib.Path = _ROOT / "data" / "raw"
    data_processed: pathlib.Path = _ROOT / "data" / "processed"
    data_metadata: pathlib.Path = _ROOT / "data" / "metadata"
    outputs_models: pathlib.Path = _ROOT / "outputs" / "models"
    outputs_figures: pathlib.Path = _ROOT / "outputs" / "figures"
    outputs_reports: pathlib.Path = _ROOT / "outputs" / "reports"
    logs: pathlib.Path = _ROOT / "logs"

    def ensure_all(self) -> None:
        """Create all directories if they don't exist."""
        for f in self.__dataclass_fields__:
            p = getattr(self, f)
            if isinstance(p, pathlib.Path) and f != "root":
                p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingConfig:
    # Primary model — Ancient-Greek-BERT from HuggingFace
    model_name: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL", "pranaydeeps/Ancient-Greek-BERT"
        )
    )
    # Layer to extract CLS token / mean pool from (-1 = last hidden state)
    layer_index: int = -1
    # Pooling strategy: "mean" | "cls"
    pooling: str = "mean"
    # Batch size for batch_embed()
    batch_size: int = 16
    # Max token length
    max_length: int = 512
    # Whether to use GPU when available
    use_gpu: bool = True
    # Deterministic inference (sets torch seed & eval mode)
    deterministic: bool = True
    # Cache directory for model weights
    cache_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class CalibrationConfig:
    # Number of bootstrap iterations for CIs
    n_bootstrap: int = 1000
    # Confidence level
    ci_level: float = 0.95
    # Minimum pericopes needed to compute a signature
    min_pericopes: int = 5
    # Random seed for reproducibility
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class ScoringConfig:
    # Toggle Bayesian normalisation (vs. frequentist cosine distance)
    use_bayesian: bool = False
    # Prior weight for Bayesian mode
    prior_weight: float = 0.1
    # Output histogram bin count
    histogram_bins: int = 30


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
@dataclass
class ReconstructionConfig:
    # Maximum iterations for convergence loop
    max_iter: int = 100
    # Convergence tolerance (L2 norm of update step)
    tolerance: float = 1e-6
    # Lambda regularisation for pseudo-inverse
    ridge_lambda: float = 1e-3


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    n_permutations: int = 1000
    n_seeds: int = 5
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class STCMConfig:
    paths: Paths = field(default_factory=Paths)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def setup_logging(self) -> logging.Logger:
        """Configure root logger and return it."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger("stcm")

    def seed_everything(self) -> None:
        """Seed Python, NumPy, and (optionally) PyTorch for reproducibility."""
        seed = self.calibration.random_seed
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


# Singleton default config
default_config = STCMConfig()
