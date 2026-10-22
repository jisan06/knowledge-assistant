import logging
from typing import List, Optional

from app.config import settings
from app.schemas.document import DocumentChunk, DocumentPage
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    IndexDocumentResponse,
    CollectionStatsResponse,
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.vector_store.base import BaseVectorStore
from app.vector_store.factory import get_vector_store

logger = logging.getLogger(__name__)


class SearchService:
    """
    Orchestration service for semantic search and document vector indexing.

    Responsibilities:
    - Embedding query text using the same model as document chunks.
    - Retrieving top-k relevant chunks by vector cosine similarity.
    - Indexing and updating document chunks with deduplication via stable chunk IDs.
    - Handling empty queries, empty collections, and error states gracefully.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or get_vector_store()

    async def search(
        self,
        query: str,
        top_k: int = 4,
        filter_filename: Optional[str] = None,
    ) -> SearchResponse:
        """
        Performs semantic vector search for a user text query.
        1. Validates non-emptiness of query.
        2. Embeds the query into a float vector using the configured embedding model.
        3. Retrieves the top_k most similar document chunks.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("Search query cannot be empty.")

        k = max(1, min(top_k, 20)) if top_k else settings.SEARCH_TOP_K

        # Check if collection is empty upfront
        if self.vector_store.count() == 0:
            return SearchResponse(
                query=clean_query,
                total_results=0,
                results=[],
                model=self.embedding_service.model_name,
            )

        # Generate query vector with the identical embedding model used for indexing
        query_vector = await self.embedding_service.embed_text(clean_query)

        results: List[SearchResultItem] = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            filter_filename=filter_filename,
        )

        return SearchResponse(
            query=clean_query,
            total_results=len(results),
            results=results,
            model=self.embedding_service.model_name,
        )

    async def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Embeds and stores document chunks in the vector database.
        Preserves stable chunk IDs, preventing duplicate entries on re-indexing.
        """
        if not chunks:
            return 0

        # Filter out empty or whitespace-only chunks
        valid_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not valid_chunks:
            return 0

        # Embed all chunks in batch if they don't already have embeddings
        needs_embedding = [c for c in valid_chunks if not c.embedding]
        if needs_embedding:
            await self.embedding_service.embed_chunks(needs_embedding)

        embeddings = [c.embedding for c in valid_chunks if c.embedding is not None]

        if len(embeddings) != len(valid_chunks):
            raise RuntimeError("Mismatch between chunks and generated embeddings.")

        return self.vector_store.upsert_chunks(
            chunks=valid_chunks,
            embeddings=embeddings,
        )

    async def index_document_pages(
        self,
        pages: List[DocumentPage],
        filename: str,
    ) -> IndexDocumentResponse:
        """
        Chunks and indexes extracted pages of a document into the vector store.
        """
        # Chunk pages independently
        chunks = ChunkingService.chunk_document_pages(pages=pages, filename=filename)
        indexed_count = await self.index_chunks(chunks)

        return IndexDocumentResponse(
            filename=filename,
            total_pages=len(pages),
            total_chunks=len(chunks),
            indexed_chunks=indexed_count,
            message=(
                f"Successfully indexed {indexed_count} chunk(s) from "
                f"'{filename}' across {len(pages)} page(s)."
            ),
        )

    def get_stats(self) -> CollectionStatsResponse:
        """Returns statistics about the vector store collection."""
        return CollectionStatsResponse(
            collection_name=self.vector_store.collection_name,
            total_chunks=self.vector_store.count(),
            store_type=self.vector_store.store_type,
        )

    def delete_document(self, filename: str) -> int:
        """Deletes all chunks belonging to a document filename."""
        return self.vector_store.delete_by_filename(filename)

    def clear(self) -> None:
        """Clears all indexed chunks from the vector store."""
        self.vector_store.clear()

