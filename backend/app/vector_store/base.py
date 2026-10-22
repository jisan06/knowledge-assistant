from abc import ABC, abstractmethod
from typing import List, Optional

from app.schemas.document import DocumentChunk
from app.schemas.search import SearchResultItem


class BaseVectorStore(ABC):
    """
    Abstract interface for document vector storage and similarity retrieval.
    """

    @property
    @abstractmethod
    def store_type(self) -> str:
        """Name of the vector store backend implementation."""
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Name of the active vector collection."""
        pass

    @abstractmethod
    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Stores or updates chunk embeddings with their text and metadata.
        Preserves stable chunk IDs and avoids duplicates on re-indexing.
        Returns the count of upserted items.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        filter_filename: Optional[str] = None,
    ) -> List[SearchResultItem]:
        """
        Performs vector similarity search against indexed chunks.
        Returns top_k most similar chunks with similarity scores.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of indexed chunk vectors in the collection."""
        pass

    @abstractmethod
    def delete_by_filename(self, filename: str) -> int:
        """Deletes all chunks belonging to the specified document filename."""
        pass

    @abstractmethod
    def get_by_filename(self, filename: str) -> List[dict]:
        """Returns copies of all records belonging to the specified document filename."""
        pass

    @abstractmethod
    def restore_records(self, records: List[dict]) -> None:
        """Restores previously backed up chunk records."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all vectors from the collection."""
        pass

