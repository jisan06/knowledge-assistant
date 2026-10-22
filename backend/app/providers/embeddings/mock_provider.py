import hashlib
import math
from typing import List

from app.providers.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic mock embedding provider for tests and zero-dependency local verification.
    Generates unit-length normalized vectors derived from SHA-256 hash of input text.
    Produces repeatable vectors without downloading models or calling external APIs.
    """

    def __init__(self, model_name: str = "mock-embedding-384", dimension: int = 384):
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        # Hash text to an integer seed for deterministic pseudo-random vector generation
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        raw = []
        for _ in range(self._dimension):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            val = (seed / 0x7FFFFFFF) * 2.0 - 1.0
            raw.append(val)

        # L2 normalize so the vector has unit norm (length = 1.0)
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [round(x / norm, 6) for x in raw]

    async def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")
        return self._generate_vector(text.strip())

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [await self.embed_text(t) for t in texts]
