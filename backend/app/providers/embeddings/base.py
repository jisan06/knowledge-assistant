from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Abstract interface for document text embedding providers.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the identifier/name of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of vectors produced by this model."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string into a float vector.
        Should handle empty/whitespace checks before invoking model.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of strings into a list of float vectors.
        """
        pass

    async def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string for semantic retrieval.
        Default implementation delegates to embed_text.
        """
        return await self.embed_text(query)

