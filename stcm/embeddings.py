"""
stcm/embeddings.py
==================
Embedding pipeline for Ancient Greek text.

Primary model : ABeZet/Koine-Greek-BERT (HuggingFace)
Fallback       : character n-gram hash embedder (offline, deterministic,
                 opt-in via STCM_ALLOW_FALLBACK=1)

The embedder is loaded lazily, on the first cache miss: a fully cached
corpus can be analysed without the model weights (or torch/transformers)
being installed at all.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
import pickle
from typing import List, Optional

import numpy as np

from stcm.config import EmbeddingConfig, default_config
from stcm.utils import normalise_greek, text_hash

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback: character n-gram hash embedder
# ---------------------------------------------------------------------------

class _NgramHashEmbedder:
    """
    Offline, deterministic character-trigram count-hashed embedder.

    Maps a Greek string to a 256-dimensional vector via feature hashing
    of character trigrams.  Deterministic, no external dependencies.
    """

    DIM = 256

    def embed(self, text: str) -> np.ndarray:
        text = normalise_greek(text)
        vec = np.zeros(self.DIM, dtype=np.float32)
        for i in range(len(text) - 2):
            gram = text[i : i + 3]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            idx = h % self.DIM
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# Primary: Koine-Greek-BERT wrapper
# ---------------------------------------------------------------------------

class _TransformerEmbedder:
    """
    Thin wrapper around HuggingFace AutoModel for mean-pooled sentence embeddings.
    """

    def __init__(self, config: EmbeddingConfig, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        log.info("Loading tokenizer: %s", config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        log.info("Loading model: %s", config.model_name)
        self._model = AutoModel.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        self._model.eval()
        self._model.to(device)
        self._device = device
        self._config = config
        log.info("Model loaded on device: %s", device)

    def embed(self, text: str) -> np.ndarray:
        import torch  # type: ignore

        text = normalise_greek(text)
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_length,
            padding=True,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        # Mean pool over token dimension, weighted by attention mask
        hidden = out.last_hidden_state  # (1, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (1, T, 1)
        vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, H)
        vec = vec.squeeze(0).cpu().numpy().astype(np.float32)
        # L2-normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# Public EmbeddingPipeline
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """
    High-level embedding pipeline with caching.

    Usage
    -----
    >>> pipe = EmbeddingPipeline()
    >>> vec = pipe.embed_text("Βίβλος γενέσεως Ἰησοῦ Χριστοῦ")
    >>> vec.shape
    (768,)  # or (256,) with fallback
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        cache_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self._cfg = config or default_config.embedding
        self._cache_dir = cache_dir or default_config.paths.data_processed
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}
        # The embedder is loaded lazily, on the first cache miss.  A fully
        # cached corpus can therefore be analysed without the model weights
        # (or torch/transformers) being available at all.
        self._embedder = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_embedder(self):
        """
        Load the transformer model.

        The offline n-gram fallback is only used when explicitly enabled via
        the STCM_ALLOW_FALLBACK=1 environment variable.  A silent fallback
        would risk producing results with a different embedder from the one
        named in the reports, so by default a failure to load the model is
        a hard error.
        """
        import os

        if self._cfg.use_gpu:
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = "cpu"

        try:
            embedder = _TransformerEmbedder(self._cfg, device)
            log.info("Using transformer embedder: %s", self._cfg.model_name)
            return embedder
        except Exception as exc:
            if os.environ.get("STCM_ALLOW_FALLBACK") == "1":
                log.warning(
                    "Cannot load transformer model (%s). "
                    "STCM_ALLOW_FALLBACK=1 set — using n-gram hash embedder. "
                    "Results will NOT be comparable to published outputs.",
                    exc,
                )
                return _NgramHashEmbedder()
            raise RuntimeError(
                f"Cannot load transformer model '{self._cfg.model_name}' ({exc}). "
                "Set STCM_ALLOW_FALLBACK=1 to use the offline n-gram embedder "
                "for smoke-testing only."
            ) from exc

    def _ensure_embedder(self):
        """Load the embedder on first use (lazy)."""
        if self._embedder is None:
            self._embedder = self._load_embedder()
        return self._embedder

    @property
    def embedder_id(self) -> str:
        """Identifier of the embedder actually in use (for reports and cache keys)."""
        if isinstance(self._embedder, _NgramHashEmbedder):
            return "ngram-fallback-256"
        return self._cfg.model_name

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        if isinstance(self._embedder, _NgramHashEmbedder):
            return _NgramHashEmbedder.DIM
        # Peek at a dummy embedding (loads the embedder only if uncached)
        dummy = self.embed_text("τεστ")
        return dummy.shape[0]

    def _cache_path(self, key: str) -> pathlib.Path:
        return self._cache_dir / f"emb_{key}.pkl"

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            return self._cache[key]
        p = self._cache_path(key)
        if p.exists():
            with open(p, "rb") as fh:
                vec = pickle.load(fh)
            self._cache[key] = vec
            return vec
        return None

    def _save_cache(self, key: str, vec: np.ndarray) -> None:
        self._cache[key] = vec
        p = self._cache_path(key)
        # Atomic write (tmp + rename) so an interrupted run can never
        # leave a truncated cache entry behind.
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(vec, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single Greek string, using the disk cache when available.

        Parameters
        ----------
        text : Greek string (raw or normalised)

        Returns
        -------
        np.ndarray, shape (D,), float32, L2-normalised
        """
        # The cache key includes the model identity so that switching
        # models (e.g. Ancient-Greek-BERT -> Koine-Greek-BERT) can never
        # silently reuse vectors produced by a different model.
        key = text_hash(f"{self._cfg.model_name}::{text}")
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        # Backward compatibility: caches written before v0.2 were keyed by
        # the text hash alone.  Accept them only for the default published
        # model, and migrate the entry to the new key on first access.
        if self._cfg.model_name == "ABeZet/Koine-Greek-BERT":
            legacy = self._load_cache(text_hash(text))
            if legacy is not None:
                self._save_cache(key, legacy)
                return legacy
        vec = self._ensure_embedder().embed(text)
        self._save_cache(key, vec)
        return vec

    def embed_pericope(self, sentences: List[str]) -> np.ndarray:
        """
        Embed a list of sentences (a pericope) by mean-pooling individual embeddings.

        Parameters
        ----------
        sentences : list of Greek strings

        Returns
        -------
        np.ndarray, shape (D,), float32, L2-normalised
        """
        vecs = [self.embed_text(s) for s in sentences if s.strip()]
        if not vecs:
            raise ValueError("Cannot embed an empty pericope.")
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        return mean_vec

    def batch_embed(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of strings, returning a (N, D) matrix.

        Parameters
        ----------
        texts         : list of Greek strings
        show_progress : log progress every 10 %

        Returns
        -------
        np.ndarray, shape (N, D), float32
        """
        results: List[np.ndarray] = []
        n = len(texts)
        log.info("Batch-embedding %d texts ...", n)
        for i, t in enumerate(texts):
            results.append(self.embed_text(t))
            if show_progress and n >= 10 and (i + 1) % max(1, n // 10) == 0:
                log.info("  %.0f %% done (%d / %d)", 100 * (i + 1) / n, i + 1, n)
        return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def load_model(config: Optional[EmbeddingConfig] = None) -> EmbeddingPipeline:
    """
    Construct and return an EmbeddingPipeline (convenience function).

    Parameters
    ----------
    config : optional EmbeddingConfig; uses default_config if None

    Returns
    -------
    EmbeddingPipeline
    """
    return EmbeddingPipeline(config=config)
