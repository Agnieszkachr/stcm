"""
stcm/embeddings.py
==================
Embedding pipeline for Ancient Greek text.

Primary model : ABeZet/Koine-Greek-BERT (HuggingFace)
Fallback       : character n-gram hash embedder (offline, deterministic,
                 opt-in via STCM_ALLOW_FALLBACK=1)

Each contiguous verse range is encoded as ONE sequence (whole-text
contextual encoding) and represented by the attention-masked mean of the
hidden states (last layer by default), L2-normalised.  Texts longer than
the model window (512 tokens) are split at sentence boundaries and the
chunk vectors are averaged with token-count weights; nothing is ever
truncated.  Composite units (verses gathered from different places in a
gospel) are encoded range by range and combined the same way, so that no
context is created across a splice that the evangelist never wrote
(``embed_parts``).

Cache entries (``data/processed/emb_<key>.pkl``) store the vector together
with the number of subword tokens, so that token-weighted combination of
cached parts needs no tokenizer.

The embedder is loaded lazily, on the first cache miss: a fully cached
corpus can be analysed without the model weights (or torch/transformers)
being installed at all.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
import pickle
from typing import List, Optional, Tuple

import numpy as np

from stcm.config import EmbeddingConfig, default_config
from stcm.utils import normalise_greek, strip_accents, text_hash

log = logging.getLogger(__name__)

# Cache-key salt for text normalisation and range-wise encoding.
ENCODING_VERSION = "v3"


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
        return vec, max(1, len(text.split()))


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
            revision=config.model_revision,
            cache_dir=config.cache_dir,
        )
        log.info("Loading model: %s (revision %s)", config.model_name, config.model_revision)
        self._model = AutoModel.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            cache_dir=config.cache_dir,
            output_hidden_states=(config.layers != "last"),
        )
        self._model.eval()
        self._model.to(device)
        self._device = device
        self._config = config
        log.info("Model loaded on device: %s", device)

    def n_tokens(self, text: str) -> int:
        """Number of subword tokens (without special tokens)."""
        return len(self._tokenizer(normalise_greek(text), add_special_tokens=False)["input_ids"])

    def _chunks(self, text: str) -> List[str]:
        """
        Split a text into chunks that fit the model's window.

        The whole text is encoded in one pass whenever it fits (this is the
        case for every double-tradition pericope).  Longer texts are split
        at sentence boundaries (. ; ·) into the smallest number of chunks
        that each fit within max_length - 2 tokens; a chunk is never cut
        mid-sentence unless a single sentence alone exceeds the window, in
        which case it is split on whitespace.  Silent truncation never
        occurs.
        """
        limit = self._config.max_length - 2
        if self.n_tokens(text) <= limit:
            return [text]
        import re
        sents = [s.strip() for s in re.split(r"(?<=[.;·])\s+", text) if s.strip()]
        chunks: List[str] = []
        cur = ""
        for s in sents:
            if self.n_tokens(s) > limit:
                # Degenerate case: split a very long sentence on words.
                words = s.split()
                piece = ""
                for w in words:
                    cand = (piece + " " + w).strip()
                    if self.n_tokens(cand) > limit and piece:
                        chunks.append(piece)
                        piece = w
                    else:
                        piece = cand
                s = piece
            cand = (cur + " " + s).strip()
            if self.n_tokens(cand) > limit and cur:
                chunks.append(cur)
                cur = s
            else:
                cur = cand
        if cur:
            chunks.append(cur)
        return chunks

    def _encode(self, text: str) -> Tuple[np.ndarray, int]:
        """Mean-pooled last-hidden-state vector for one chunk, with its token count."""
        import torch  # type: ignore

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        n_tok = int(enc["attention_mask"].sum())
        if n_tok > self._config.max_length:
            raise ValueError(
                f"Chunk of {n_tok} tokens exceeds max_length={self._config.max_length}"
            )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        # Mean pool over the token dimension (special tokens included, as
        # in the attention mask), from the last layer or the mean of the
        # last four layers.
        if self._config.layers == "last4":
            hidden = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)  # (1, T, H)
        elif self._config.layers == "last":
            hidden = out.last_hidden_state  # (1, T, H)
        else:
            raise ValueError(f"Unknown layers setting: {self._config.layers}")
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (1, T, 1)
        vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, H)
        return vec.squeeze(0).cpu().numpy().astype(np.float32), n_tok

    def embed(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Embed a text: one forward pass if it fits the window, otherwise the
        token-count-weighted mean of the chunk vectors (equivalent to mean
        pooling over all tokens of the text), L2-normalised.  Returns the
        vector and the total number of tokens encoded.
        """
        text = normalise_greek(text)
        if self._config.strip_accents:
            text = strip_accents(text)
        chunks = self._chunks(text)
        if len(chunks) > 1:
            log.info("Text of %d tokens split into %d chunks for encoding.",
                     self.n_tokens(text), len(chunks))
        vecs, weights = [], []
        for c in chunks:
            v, n = self._encode(c)
            vecs.append(v)
            weights.append(n)
        vec = np.average(np.array(vecs), axis=0, weights=np.array(weights, dtype=np.float64))
        vec = vec.astype(np.float32)
        # L2-normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec, int(sum(weights))


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
        return f"{self._cfg.model_name}@{self._cfg.model_revision} ({self._cfg.layers})"

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

    def _load_cache(self, key: str) -> Optional[Tuple[np.ndarray, int]]:
        if key in self._cache:
            return self._cache[key]
        p = self._cache_path(key)
        if p.exists():
            with open(p, "rb") as fh:
                entry = pickle.load(fh)
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise ValueError(f"Cache entry {p} is not a (vector, n_tokens) tuple.")
            self._cache[key] = entry
            return entry
        return None

    def _save_cache(self, key: str, entry: Tuple[np.ndarray, int]) -> None:
        self._cache[key] = entry
        p = self._cache_path(key)
        # Atomic write (tmp + rename) so an interrupted run can never
        # leave a truncated cache entry behind.
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(entry, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    def _key(self, text: str) -> str:
        # The key includes the model identity, its pinned revision, the
        # layer setting, the accent handling and the encoding-scheme
        # version, so that vectors produced under any other configuration
        # can never be silently reused.
        cfg = self._cfg
        return text_hash(f"{cfg.model_name}@{cfg.model_revision}::{cfg.layers}::"
                         f"strip={int(cfg.strip_accents)}::{ENCODING_VERSION}::{normalise_greek(text)}")

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
        return self.embed_text_with_count(text)[0]

    def embed_text_with_count(self, text: str) -> Tuple[np.ndarray, int]:
        """Embed a string and return (vector, number of subword tokens encoded)."""
        key = self._key(text)
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        entry = self._ensure_embedder().embed(text)
        self._save_cache(key, entry)
        return entry

    def token_count(self, text: str) -> int:
        """Number of subword tokens the pipeline encoded for this text (cached)."""
        return self.embed_text_with_count(text)[1]

    def embed_parts(self, parts: List[str]) -> np.ndarray:
        """
        Embed a unit given as a list of contiguous verse ranges: each range
        is encoded on its own and the vectors are averaged with token-count
        weights (the same combination used for chunks of an over-long
        text), then L2-normalised.  A single-range unit is identical to
        ``embed_text``.

        Parameters
        ----------
        parts : list of Greek strings, one per contiguous range

        Returns
        -------
        np.ndarray, shape (D,), float32, L2-normalised
        """
        parts = [p for p in parts if p.strip()]
        if not parts:
            raise ValueError("Cannot embed an empty unit.")
        if len(parts) == 1:
            return self.embed_text(parts[0])
        vecs, weights = zip(*(self.embed_text_with_count(p) for p in parts))
        vec = np.average(np.array(vecs, dtype=np.float64), axis=0,
                         weights=np.array(weights, dtype=np.float64)).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_pericope(self, sentences: List[str]) -> np.ndarray:
        """Embed a list of sentences by mean-pooling their individual embeddings (legacy)."""
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
