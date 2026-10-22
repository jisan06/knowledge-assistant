import logging
import math
import re
from typing import List, Optional, Set

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

# Stopwords and meta-query boilerplate ignored during lexical relevance weighting
SEARCH_STOPWORDS: Set[str] = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be",
    "been", "by", "can", "could", "do", "does", "for", "from", "had",
    "has", "have", "how", "if", "in", "into", "is", "it", "of", "on",
    "or", "our", "out", "over", "so", "that", "the", "their", "there",
    "this", "to", "up", "was", "were", "what", "when", "where", "which",
    "who", "whom", "why", "will", "with", "would",
    "document", "documents", "pdf", "page", "pages", "file", "listed",
    "tell", "please", "stated", "mentioned", "give", "show", "provide",
}


class SearchService:
    """
    Orchestration service for semantic search and document vector indexing.

    Responsibilities:
    - Embedding query text using query-specific instruction/model.
    - Retrieving candidates and applying hybrid dense-lexical scoring.
    - Preserving stable chunk IDs and deduplicating entries.
    - Handling empty queries, empty collections, and error states gracefully.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or get_vector_store()

    @staticmethod
    def _apply_hybrid_reranking(
        query: str,
        candidates: List[SearchResultItem],
        top_k: int,
    ) -> List[SearchResultItem]:
        """
        Combines dense vector similarity with BM25-like lexical relevance scores.
        Ensures high-discriminative query terms (e.g. 'reference') match their exact chunk
        even when embeddings are diluted by long lists or URLs.
        """
        if not candidates or top_k <= 0:
            return []

        # Extract content words from query
        raw_words = re.findall(r"[a-z0-9]+", query.lower())
        query_terms = [w for w in raw_words if w not in SEARCH_STOPWORDS and len(w) >= 2]

        if not query_terms or len(candidates) <= 1:
            return candidates[:top_k]

        total_candidates = len(candidates)

        # 1. Calculate Document Frequency (DF) and Inverse Document Frequency (IDF) for each query term
        term_df = {}
        for t in query_terms:
            df = sum(1 for c in candidates if t in c.text.lower())
            term_df[t] = df

        term_idf = {}
        for t, df in term_df.items():
            if df > 0:
                # Standard smoothed BM25 IDF formulation
                term_idf[t] = math.log(1.0 + (total_candidates - df + 0.5) / (df + 0.5))
            else:
                term_idf[t] = 0.0

        # 2. Score each candidate by term match frequency weighted by IDF
        raw_scores = {}
        for c in candidates:
            text_lower = c.text.lower()
            cand_score = 0.0
            for t in query_terms:
                idf = term_idf.get(t, 0.0)
                if idf <= 0.0:
                    continue
                # Whole-word with common inflection boundary matching
                term_count = len(re.findall(r"\b" + re.escape(t) + r"(?:s|es|ed|ing)?\b", text_lower))
                if term_count > 0:
                    cand_score += idf * (1.0 + math.log(term_count))
            raw_scores[c.chunk_id] = cand_score

        max_score = max(raw_scores.values()) if raw_scores else 0.0

        scored_candidates = []
        if max_score > 0.0:
            # 3. Linearly blend dense vector similarity (65%) and normalized lexical score (35%)
            for c in candidates:
                vec_sim = c.similarity_score
                norm_lex = raw_scores.get(c.chunk_id, 0.0) / max_score
                hybrid_sim = round(0.65 * vec_sim + 0.35 * norm_lex, 4)
                hybrid_dist = max(0.0, round(1.0 - hybrid_sim, 4))

                updated_item = SearchResultItem(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    filename=c.filename,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                    similarity_score=hybrid_sim,
                    distance=hybrid_dist,
                )
                scored_candidates.append(updated_item)
        else:
            # When no candidate matches any lexical term (pure dense query or unanswerable query),
            # preserve original vector similarity without artificial attenuation penalty
            scored_candidates = list(candidates)

        # Sort descending by hybrid similarity score
        scored_candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored_candidates[:top_k]

    async def search(
        self,
        query: str,
        top_k: int = 4,
        filter_filename: Optional[str] = None,
    ) -> SearchResponse:
        """
        Performs hybrid semantic search for a user text query.
        1. Validates non-emptiness of query.
        2. Embeds the query into a float vector using query-specific instruction/model.
        3. Retrieves top candidates from vector store and applies hybrid dense-lexical scoring.
        4. Returns the top_k most relevant document chunks.
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

        # Generate query vector with query-specific representation
        query_vector = await self.embedding_service.embed_query(clean_query)

        # Retrieve a broader candidate pool to enable hybrid reranking
        candidate_k = max(k * 3, 12)
        candidates: List[SearchResultItem] = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
            filter_filename=filter_filename,
        )

        # Apply hybrid dense-lexical reranking to accurately prioritize specific terms
        results = self._apply_hybrid_reranking(clean_query, candidates, top_k=k)

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

        # Step 2: Store chunks in vector database with partial-state cleanup on failure
        filename = valid_chunks[0].filename
        try:
            return self.vector_store.upsert_chunks(
                chunks=valid_chunks,
                embeddings=embeddings,
            )
        except Exception as e:
            logger.error(
                f"Failed to upsert chunks into vector store for '{filename}': {e}. "
                "Cleaning up any partial vectors to maintain index consistency."
            )
            try:
                self.vector_store.delete_by_filename(filename)
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up partial chunks for '{filename}': {cleanup_err}")
            raise

    async def replace_document_chunks(
        self,
        filename: str,
        chunks: List[DocumentChunk],
    ) -> int:
        """
        Replaces all chunks for a document filename cleanly and atomically.
        1. Embeds new chunks in memory first.
        2. Backs up existing chunk records for filename in case replacement fails.
        3. Deletes existing chunks for filename from the vector store so no stale chunks remain.
        4. Upserts new chunks.
        5. If upsert fails, restores the backed-up records to preserve a working previous version.
        """
        if not chunks:
            return 0

        valid_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not valid_chunks:
            return 0

        # Step 1: Embed new chunks BEFORE modifying vector store
        needs_embedding = [c for c in valid_chunks if not c.embedding]
        if needs_embedding:
            await self.embedding_service.embed_chunks(needs_embedding)

        embeddings = [c.embedding for c in valid_chunks if c.embedding is not None]
        if len(embeddings) != len(valid_chunks):
            raise RuntimeError("Mismatch between chunks and generated embeddings.")

        # Step 2: Backup existing records for this filename
        backup_records = self.vector_store.get_by_filename(filename)

        # Step 3: Remove old chunks from vector store to prevent stale data
        self.vector_store.delete_by_filename(filename)

        # Step 4: Upsert new chunks, restoring backup if failure occurs
        try:
            return self.vector_store.upsert_chunks(
                chunks=valid_chunks,
                embeddings=embeddings,
            )
        except Exception as e:
            logger.error(
                f"Failed to upsert new chunks during document update for '{filename}': {e}. "
                f"Restoring {len(backup_records)} backup record(s)."
            )
            if backup_records:
                try:
                    self.vector_store.restore_records(backup_records)
                except Exception as restore_err:
                    logger.error(f"Failed to restore backup chunks for '{filename}': {restore_err}")
            raise


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

