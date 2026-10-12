"""Text embedders. Every embedder returns a unit-norm vector."""

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class HashingEmbedder:
    """Deterministic bag-of-words embedder. Offline; for tests, not for semantics."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
            v[h % self.dim] += 1.0
        return _unit(v)


class SentenceTransformerEmbedder:
    """Semantic embedder backed by sentence-transformers (install the `st` extra)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed(self, text: str) -> np.ndarray:
        return _unit(np.asarray(self._model.encode(text), dtype=float))
