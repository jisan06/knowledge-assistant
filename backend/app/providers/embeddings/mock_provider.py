import hashlib
import math
import re
from typing import List, Set

from app.providers.embeddings.base import BaseEmbeddingProvider

# Common English stop words to exclude from semantic keyword hashing
STOP_WORDS: Set[str] = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be",
    "been", "by", "can", "could", "do", "does", "for", "from", "had",
    "has", "have", "how", "if", "in", "into", "is", "it", "of", "on",
    "or", "our", "out", "over", "so", "that", "the", "their", "there",
    "this", "to", "up", "was", "were", "what", "when", "where", "which",
    "who", "whom", "why", "will", "with", "would",
}


# Meta-query boilerplate terms to exclude when embedding search queries
QUERY_BOILERPLATE: Set[str] = {
    "document", "documents", "listed", "pdf", "file", "please", "tell",
    "stated", "mentioned", "give", "provide", "show",
}


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic mock embedding provider for tests and zero-dependency local verification.
    Generates unit-length normalized vectors representing semantic token similarity.
    Uses sparse random feature projections (Count-Sketch / Random Indexing) derived from
    word tokens, subword prefixes, and compound terms to support realistic semantic retrieval
    without downloading models or calling external APIs.
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

    def _generate_vector(self, text: str, is_query: bool = False) -> List[float]:
        # Tokenize text into words and compound tokens
        raw_words = re.findall(r"[a-z0-9]+", text.lower())
        ignored_words = STOP_WORDS | QUERY_BOILERPLATE if is_query else STOP_WORDS
        content_words = [w for w in raw_words if w not in ignored_words and len(w) >= 2]

        # Fallback for texts containing only stop words or single characters
        tokens = content_words if content_words else (raw_words if raw_words else [text.strip().lower()])

        # Accumulate sparse random feature projections in a vector
        vector = [0.0] * self._dimension

        def _project_feature(feat: str, weight: float = 1.0) -> None:
            seed = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:16], 16)
            for _ in range(8):
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                idx = seed % self._dimension
                sign = 1.0 if ((seed >> 16) & 1) else -1.0
                vector[idx] += sign * weight

        # 1. Word and subword prefix projections for semantic and morphological matching
        for w in tokens:
            weight = 3.0 if (is_query and len(w) >= 6) else 2.0
            _project_feature(f"w:{w}", weight=weight)
            if len(w) >= 3:
                _project_feature(f"p3:{w[:3]}", weight=0.8)
            if len(w) >= 4:
                _project_feature(f"p4:{w[:4]}", weight=1.0)
            if len(w) >= 5:
                _project_feature(f"p5:{w[:5]}", weight=1.2)

        # 2. Add a subtle text-level hash component for sequence/uniqueness distinction
        text_seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(self._dimension):
            text_seed = (text_seed * 1103515245 + 12345) & 0x7FFFFFFF
            val = ((text_seed / 0x7FFFFFFF) * 2.0 - 1.0) * 0.05
            vector[i] += val

        # 3. L2 normalize to unit length (norm = 1.0)
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [round(x / norm, 6) for x in vector]

    async def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")
        return self._generate_vector(text.strip(), is_query=False)

    async def embed_query(self, query: str) -> List[float]:
        if not query or not query.strip():
            raise ValueError("Cannot generate embedding for empty query.")
        return self._generate_vector(query.strip(), is_query=True)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [await self.embed_text(t) for t in texts]

