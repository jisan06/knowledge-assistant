import logging
from typing import List, Optional

from app.providers.embeddings.base import BaseEmbeddingProvider
from app.providers.embeddings.factory import get_embedding_provider
from app.schemas.document import DocumentChunk, ChunkEmbeddingResult

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Base domain exception for embedding errors."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmptyChunkError(EmbeddingServiceError):
    """Raised when an embedding is requested on an empty or whitespace-only text."""

    def __init__(self, message: str = "Cannot generate embedding for empty chunk text."):
        super().__init__(message, status_code=400)


class EmbeddingModelLoadError(EmbeddingServiceError):
    """Raised when the embedding model cannot be loaded or dependencies are missing."""

    def __init__(self, message: str):
        super().__init__(message, status_code=503)


class EmbeddingService:
    """
    Service responsible for converting document chunks into numerical vectors.

    Features:
    - Reuses existing chunking output and preserves chunk_id, filename, and page_number.
    - Generates one embedding vector per non-empty chunk.
    - Handles empty text, model loading failures, and generation errors gracefully.
    - Decoupled from specific ML frameworks via BaseEmbeddingProvider.
    """

    def __init__(self, provider: Optional[BaseEmbeddingProvider] = None):
        self.provider = provider or get_embedding_provider()

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def get_model_info(self) -> dict:
        return {
            "model": self.model_name,
            "dimension": self.dimension,
        }

    async def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single raw text string.
        Raises EmptyChunkError if text is empty or only whitespace.
        """
        if not text or not text.strip():
            raise EmptyChunkError("Cannot generate embedding for empty or whitespace-only text.")

        try:
            return await self.provider.embed_text(text.strip())
        except ValueError as e:
            raise EmptyChunkError(str(e)) from e
        except RuntimeError as e:
            raise EmbeddingModelLoadError(str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error while generating text embedding")
            raise EmbeddingServiceError(f"Embedding generation failed: {e}") from e

    async def embed_query(self, query: str) -> List[float]:
        """
        Embeds a user search query, applying model-specific query instructions or prefixes.
        Raises EmptyChunkError if query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise EmptyChunkError("Cannot generate embedding for empty or whitespace-only query.")

        try:
            return await self.provider.embed_query(query.strip())
        except ValueError as e:
            raise EmptyChunkError(str(e)) from e
        except RuntimeError as e:
            raise EmbeddingModelLoadError(str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error while generating query embedding")
            raise EmbeddingServiceError(f"Query embedding generation failed: {e}") from e


    async def embed_chunk(self, chunk: DocumentChunk) -> ChunkEmbeddingResult:
        """
        Generates an embedding vector for a single DocumentChunk.
        Preserves all source chunk metadata (chunk_id, filename, page_number, chunk_index).
        """
        if not chunk.text or not chunk.text.strip():
            raise EmptyChunkError(
                f"Chunk '{chunk.chunk_id}' on page {chunk.page_number} contains no text."
            )

        vector = await self.embed_text(chunk.text)
        chunk.embedding = vector

        return ChunkEmbeddingResult(
            chunk_id=chunk.chunk_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            vector=vector,
            dimension=len(vector),
            model=self.model_name,
        )

    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[ChunkEmbeddingResult]:
        """
        Generates embedding vectors for a list of DocumentChunks in a batch.
        Filters out empty chunks safely, preserving metadata on all valid chunks.
        """
        if not chunks:
            return []

        # Only process non-empty chunks
        valid_chunks: List[DocumentChunk] = [
            c for c in chunks if c.text and c.text.strip()
        ]

        if not valid_chunks:
            return []

        texts = [c.text.strip() for c in valid_chunks]

        try:
            vectors = await self.provider.embed_batch(texts)
        except RuntimeError as e:
            raise EmbeddingModelLoadError(str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error during batch chunk embedding")
            raise EmbeddingServiceError(f"Batch embedding generation failed: {e}") from e

        results: List[ChunkEmbeddingResult] = []
        for chunk, vector in zip(valid_chunks, vectors):
            chunk.embedding = vector
            results.append(
                ChunkEmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    vector=vector,
                    dimension=len(vector),
                    model=self.model_name,
                )
            )

        return results
