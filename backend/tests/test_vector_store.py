import io
import math
import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import settings
from app.providers.embeddings.mock_provider import MockEmbeddingProvider
from app.schemas.document import DocumentChunk, DocumentPage
from app.schemas.search import SearchRequest
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.vector_store.base import BaseVectorStore
from app.vector_store.chroma_store import ChromaVectorStore
from app.vector_store.memory_store import InMemoryVectorStore, cosine_similarity
from app.vector_store.factory import get_vector_store

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests: Cosine Similarity and InMemoryVectorStore
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:
    """Verifies the core vector storage, metadata retention, and math in InMemoryVectorStore."""

    @pytest.fixture
    def empty_store(self):
        return InMemoryVectorStore(collection_name="test_collection")

    def test_empty_collection_returns_empty_results(self, empty_store):
        results = empty_store.search(query_vector=[0.1, 0.2, 0.3], top_k=4)
        assert results == []
        assert empty_store.count() == 0

    def test_upsert_and_retrieve_chunks_preserves_metadata(self, empty_store):
        chunks = [
            DocumentChunk(
                chunk_id="resume_p1_c1",
                filename="resume.pdf",
                page_number=1,
                chunk_index=1,
                text="Senior software engineer with 5 years experience in Python and FastAPI.",
                char_count=73,
            ),
            DocumentChunk(
                chunk_id="resume_p1_c2",
                filename="resume.pdf",
                page_number=1,
                chunk_index=2,
                text="Designed scalable microservices and PostgreSQL database schemas.",
                char_count=65,
            ),
        ]
        # Vectors normalized to length 1
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        count = empty_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        assert count == 2
        assert empty_store.count() == 2

        # Search with vector close to chunk 1
        results = empty_store.search(query_vector=[0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2

        top_match = results[0]
        assert top_match.chunk_id == "resume_p1_c1"
        assert top_match.filename == "resume.pdf"
        assert top_match.page_number == 1
        assert top_match.chunk_index == 1
        assert "Senior software engineer" in top_match.text
        assert top_match.similarity_score > 0.8
        assert top_match.distance is not None

    def test_duplicate_chunk_id_updates_without_increasing_count(self, empty_store):
        """Re-indexing or re-uploading the same document chunk must not produce duplicates."""
        initial_chunk = DocumentChunk(
            chunk_id="doc_p1_c1",
            filename="doc.pdf",
            page_number=1,
            chunk_index=1,
            text="Initial text content.",
            char_count=21,
        )
        empty_store.upsert_chunks(chunks=[initial_chunk], embeddings=[[1.0, 0.0]])
        assert empty_store.count() == 1

        # Re-upsert with updated text and vector
        updated_chunk = DocumentChunk(
            chunk_id="doc_p1_c1",
            filename="doc.pdf",
            page_number=1,
            chunk_index=1,
            text="Updated revised text content.",
            char_count=29,
        )
        empty_store.upsert_chunks(chunks=[updated_chunk], embeddings=[[0.0, 1.0]])

        # Count must remain exactly 1
        assert empty_store.count() == 1

        results = empty_store.search(query_vector=[0.0, 1.0], top_k=1)
        assert len(results) == 1
        assert results[0].text == "Updated revised text content."
        assert abs(results[0].similarity_score - 1.0) < 1e-4

    def test_delete_by_filename(self, empty_store):
        chunks = [
            DocumentChunk(
                chunk_id="docA_p1_c1",
                filename="docA.pdf",
                page_number=1,
                chunk_index=1,
                text="Doc A content",
                char_count=13,
            ),
            DocumentChunk(
                chunk_id="docB_p1_c1",
                filename="docB.pdf",
                page_number=1,
                chunk_index=1,
                text="Doc B content",
                char_count=13,
            ),
        ]
        empty_store.upsert_chunks(chunks=chunks, embeddings=[[1.0, 0.0], [0.0, 1.0]])
        assert empty_store.count() == 2

        deleted = empty_store.delete_by_filename("docA.pdf")
        assert deleted == 1
        assert empty_store.count() == 1

        remaining = empty_store.search(query_vector=[0.0, 1.0], top_k=2)
        assert len(remaining) == 1
        assert remaining[0].filename == "docB.pdf"

    def test_filename_filter(self, empty_store):
        chunks = [
            DocumentChunk(
                chunk_id="report_p1_c1",
                filename="report.pdf",
                page_number=1,
                chunk_index=1,
                text="Financial figures for Q1",
                char_count=24,
            ),
            DocumentChunk(
                chunk_id="memo_p1_c1",
                filename="memo.pdf",
                page_number=1,
                chunk_index=1,
                text="Team meeting notes",
                char_count=18,
            ),
        ]
        empty_store.upsert_chunks(chunks=chunks, embeddings=[[1.0, 0.0], [0.95, 0.05]])

        # Filter strictly by 'memo.pdf'
        results = empty_store.search(query_vector=[1.0, 0.0], top_k=5, filter_filename="memo.pdf")
        assert len(results) == 1
        assert results[0].filename == "memo.pdf"


# ---------------------------------------------------------------------------
# Unit Tests: SearchService with Mock Embeddings (Dimension Compatibility)
# ---------------------------------------------------------------------------

class TestSearchService:
    """Verifies end-to-end vector search orchestration and dimension compatibility."""

    @pytest.fixture
    def search_service(self):
        # 384 dimensions matching BAAI/bge-small-en-v1.5
        mock_provider = MockEmbeddingProvider(dimension=384)
        embedding_service = EmbeddingService(provider=mock_provider)
        memory_store = InMemoryVectorStore(collection_name="test_search_service")
        return SearchService(embedding_service=embedding_service, vector_store=memory_store)

    @pytest.mark.asyncio
    async def test_search_empty_collection_returns_zero_results(self, search_service):
        response = await search_service.search(query="Python asynchronous programming")
        assert response.total_results == 0
        assert response.results == []
        assert response.query == "Python asynchronous programming"

    @pytest.mark.asyncio
    async def test_search_empty_query_raises_value_error(self, search_service):
        with pytest.raises(ValueError, match="cannot be empty"):
            await search_service.search(query="")

        with pytest.raises(ValueError, match="cannot be empty"):
            await search_service.search(query="   \t\n  ")

    @pytest.mark.asyncio
    async def test_dimension_compatibility_and_metadata_retrieval(self, search_service):
        """Verifies that query vector and chunk vectors share identical dimensions (384)."""
        pages = [
            DocumentPage(
                page_number=1,
                character_count=100,
                text=(
                    "Antigravity is an advanced agentic coding assistant with rich developer tools. "
                    "It supports FastAPI backend architecture and semantic vector indexing."
                ),
            ),
            DocumentPage(
                page_number=2,
                character_count=90,
                text=(
                    "Marine biology studies oceanic organisms, coral reef preservation, and deep sea life. "
                    "Whales and dolphins navigate using echolocation."
                ),
            ),
        ]

        index_result = await search_service.index_document_pages(pages=pages, filename="knowledge.pdf")
        assert index_result.indexed_chunks >= 2
        assert search_service.vector_store.count() >= 2

        # Query relevant to page 1
        coding_response = await search_service.search(query="agentic coding assistant tools", top_k=3)
        assert coding_response.total_results >= 1
        top_match = coding_response.results[0]
        assert top_match.page_number == 1
        assert "Antigravity" in top_match.text
        assert top_match.chunk_id.startswith("knowledge_pdf_p1_c")
        assert top_match.similarity_score > 0.0

        # Query relevant to page 2
        biology_response = await search_service.search(query="ocean coral reef whales", top_k=3)
        assert biology_response.total_results >= 1
        bio_match = biology_response.results[0]
        assert bio_match.page_number == 2
        assert "Marine biology" in bio_match.text
        assert bio_match.chunk_id.startswith("knowledge_pdf_p2_c")


# ---------------------------------------------------------------------------
# API Integration Tests: Search and Indexing Endpoints
# ---------------------------------------------------------------------------

class TestSearchAPIEndpoints:
    """Tests the FastAPI routes for semantic search and document indexing."""

    @pytest.fixture(autouse=True)
    def clean_collection(self):
        # Clear vector collection before each API test
        client.delete("/api/documents/collection")
        yield
        client.delete("/api/documents/collection")

    def test_collection_stats_initial_empty(self):
        resp = client.get("/api/documents/collection-stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "collection_name" in data
        assert data["total_chunks"] == 0
        assert "store_type" in data

    def test_search_empty_collection(self):
        resp = client.post("/api/documents/search", json={"query": "test query", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test query"
        assert data["total_results"] == 0
        assert data["results"] == []

    def test_search_empty_query_rejected(self):
        resp = client.post("/api/documents/search", json={"query": "   ", "top_k": 3})
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_index_chunks_and_search_get_and_post(self):
        # Index two sample chunks via API
        chunks_payload = {
            "chunks": [
                {
                    "chunk_id": "testdoc_p1_c1",
                    "filename": "testdoc.pdf",
                    "page_number": 1,
                    "chunk_index": 1,
                    "text": "Machine learning algorithms optimize mathematical loss functions over training data.",
                    "char_count": 82,
                },
                {
                    "chunk_id": "testdoc_p2_c1",
                    "filename": "testdoc.pdf",
                    "page_number": 2,
                    "chunk_index": 1,
                    "text": "Gardening techniques include soil composting, drip irrigation, and seasonal pruning.",
                    "char_count": 83,
                },
            ]
        }

        index_resp = client.post("/api/documents/index-chunks", json=chunks_payload)
        assert index_resp.status_code == 200
        assert index_resp.json()["indexed_chunks"] == 2

        # Verify collection count
        stats = client.get("/api/documents/collection-stats").json()
        assert stats["total_chunks"] == 2

        # Test POST search
        post_search = client.post(
            "/api/documents/search",
            json={"query": "loss function optimization", "top_k": 2},
        )
        assert post_search.status_code == 200
        post_data = post_search.json()
        assert post_data["total_results"] == 2
        # Most relevant match should be chunk 1 (Machine learning)
        assert post_data["results"][0]["chunk_id"] == "testdoc_p1_c1"
        assert post_data["results"][0]["page_number"] == 1
        assert "Machine learning" in post_data["results"][0]["text"]

        # Test GET search (browser-friendly endpoint)
        get_search = client.get("/api/documents/search?q=gardening%20pruning&top_k=2")
        assert get_search.status_code == 200
        get_data = get_search.json()
        assert get_data["total_results"] == 2
        assert get_data["results"][0]["chunk_id"] == "testdoc_p2_c1"
        assert get_data["results"][0]["page_number"] == 2
        assert "Gardening" in get_data["results"][0]["text"]

    def test_duplicate_id_preservation_via_index_chunks(self):
        """Upserting chunks with identical IDs must not increase total collection count."""
        payload = {
            "chunks": [
                {
                    "chunk_id": "dedup_p1_c1",
                    "filename": "dedup.pdf",
                    "page_number": 1,
                    "chunk_index": 1,
                    "text": "Original text version.",
                    "char_count": 22,
                }
            ]
        }
        resp1 = client.post("/api/documents/index-chunks", json=payload)
        assert resp1.status_code == 200
        assert client.get("/api/documents/collection-stats").json()["total_chunks"] == 1

        # Re-index with updated text
        payload["chunks"][0]["text"] = "Revised text version."
        resp2 = client.post("/api/documents/index-chunks", json=payload)
        assert resp2.status_code == 200
        # Count must remain 1
        assert client.get("/api/documents/collection-stats").json()["total_chunks"] == 1

        # Search should return revised content
        search_res = client.get("/api/documents/search?q=Revised&top_k=1").json()
        assert search_res["total_results"] == 1
        assert search_res["results"][0]["text"] == "Revised text version."


# ---------------------------------------------------------------------------
# Unit Tests: ChromaVectorStore Mocked Execution
# ---------------------------------------------------------------------------

class TestChromaVectorStore:
    """Verifies ChromaVectorStore mapping, upsert semantics, and distance conversion."""

    def test_chroma_store_upsert_and_search_mapping(self, monkeypatch):
        class MockChromaCollection:
            def __init__(self):
                self._stored = {}

            def upsert(self, ids, embeddings, documents, metadatas):
                for i, cid in enumerate(ids):
                    self._stored[cid] = {
                        "embedding": embeddings[i],
                        "document": documents[i],
                        "metadata": metadatas[i],
                    }

            def count(self):
                return len(self._stored)

            def query(self, query_embeddings, n_results, include, where=None):
                results_ids = list(self._stored.keys())[:n_results]
                docs = [self._stored[cid]["document"] for cid in results_ids]
                metas = [self._stored[cid]["metadata"] for cid in results_ids]
                # Cosine distance = 0.15 -> similarity = 0.85
                dists = [0.15 for _ in results_ids]
                return {
                    "ids": [results_ids],
                    "documents": [docs],
                    "metadatas": [metas],
                    "distances": [dists],
                }

            def delete(self, where=None):
                if where and "filename" in where:
                    target = where["filename"]
                    to_del = [cid for cid, v in self._stored.items() if v["metadata"].get("filename") == target]
                    for cid in to_del:
                        del self._stored[cid]

        mock_coll = MockChromaCollection()
        store = ChromaVectorStore(persist_directory="test_data/chroma", collection_name="test_col")
        monkeypatch.setattr(store, "_get_collection", lambda: mock_coll)

        chunks = [
            DocumentChunk(
                chunk_id="test_p1_c1",
                filename="test.pdf",
                page_number=1,
                chunk_index=1,
                text="ChromaDB persistent vector indexing text.",
                char_count=42,
            )
        ]
        upserted = store.upsert_chunks(chunks=chunks, embeddings=[[0.1] * 384])
        assert upserted == 1
        assert store.count() == 1

        results = store.search(query_vector=[0.1] * 384, top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "test_p1_c1"
        assert results[0].filename == "test.pdf"
        assert results[0].page_number == 1
        assert results[0].chunk_index == 1
        assert results[0].text == "ChromaDB persistent vector indexing text."
        assert results[0].similarity_score == 0.85
        assert results[0].distance == 0.15

    def test_chroma_store_empty_collection(self, monkeypatch):
        class EmptyChromaCollection:
            def count(self):
                return 0

            def query(self, **kwargs):
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        store = ChromaVectorStore(persist_directory="test_data/chroma", collection_name="empty_col")
        monkeypatch.setattr(store, "_get_collection", lambda: EmptyChromaCollection())

        assert store.count() == 0
        results = store.search(query_vector=[0.5, 0.5], top_k=3)
        assert results == []

