import math
import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import settings
from app.providers.embeddings.mock_provider import MockEmbeddingProvider
from app.schemas.document import DocumentChunk
from app.services.embedding_service import (
    EmbeddingService,
    EmptyChunkError,
    EmbeddingModelLoadError,
)

client = TestClient(app)


class TestMockEmbeddingProvider:
    """Verifies the mock embedding provider logic and math."""

    @pytest.mark.asyncio
    async def test_dimension_and_l2_normalization(self):
        provider = MockEmbeddingProvider(dimension=384)
        vector = await provider.embed_text("Artificial intelligence and knowledge systems.")

        assert len(vector) == 384
        assert provider.dimension == 384
        # L2 norm check: sqrt(sum(v_i^2)) should be ~1.0
        norm = math.sqrt(sum(x * x for x in vector))
        assert abs(norm - 1.0) < 1e-4

    @pytest.mark.asyncio
    async def test_determinism(self):
        provider = MockEmbeddingProvider()
        text = "Consistent query text for vector verification."
        vec1 = await provider.embed_text(text)
        vec2 = await provider.embed_text(text)
        assert vec1 == vec2

    @pytest.mark.asyncio
    async def test_distinct_texts_produce_different_vectors(self):
        provider = MockEmbeddingProvider()
        vec_a = await provider.embed_text("Deep learning neural networks.")
        vec_b = await provider.embed_text("Marine biology ocean ecosystems.")
        assert vec_a != vec_b

    @pytest.mark.asyncio
    async def test_empty_text_rejection(self):
        provider = MockEmbeddingProvider()
        with pytest.raises(ValueError, match="empty text"):
            await provider.embed_text("")

        with pytest.raises(ValueError, match="empty text"):
            await provider.embed_text("   \n\t  ")

    @pytest.mark.asyncio
    async def test_semantic_similarity_ranking(self):
        """Verifies that semantically related texts produce higher cosine similarity than unrelated texts."""
        provider = MockEmbeddingProvider()
        query = await provider.embed_text("vacation days and paid time off accrual")
        related = await provider.embed_text("Full-time employees accrue 20 vacation days of paid time off.")
        unrelated = await provider.embed_text("Hardware laptops are Apple MacBook Pro 16-inch workstations.")

        def dot_product(v1, v2):
            return sum(a * b for a, b in zip(v1, v2))

        sim_related = dot_product(query, related)
        sim_unrelated = dot_product(query, unrelated)
        assert sim_related > sim_unrelated
        assert sim_related > 0.3


class TestEmbeddingService:
    """Unit tests for the EmbeddingService layer."""

    @pytest.fixture
    def mock_service(self):
        return EmbeddingService(provider=MockEmbeddingProvider(dimension=384))

    @pytest.mark.asyncio
    async def test_embed_single_chunk_preserves_metadata(self, mock_service):
        chunk = DocumentChunk(
            chunk_id="resume_pdf_p1_c1",
            filename="resume.pdf",
            page_number=1,
            chunk_index=1,
            text="Experienced senior software engineer with Python and distributed systems expertise.",
            char_count=82,
            start_char=0,
            end_char=82,
        )

        result = await mock_service.embed_chunk(chunk)

        assert result.chunk_id == "resume_pdf_p1_c1"
        assert result.filename == "resume.pdf"
        assert result.page_number == 1
        assert result.chunk_index == 1
        assert result.text == chunk.text
        assert len(result.vector) == 384
        assert result.dimension == 384
        assert chunk.embedding == result.vector

    @pytest.mark.asyncio
    async def test_embed_chunks_batch(self, mock_service):
        chunks = [
            DocumentChunk(
                chunk_id="doc_p1_c1",
                filename="doc.pdf",
                page_number=1,
                chunk_index=1,
                text="Chunk one content about database indexing.",
                char_count=42,
            ),
            DocumentChunk(
                chunk_id="doc_p1_c2",
                filename="doc.pdf",
                page_number=1,
                chunk_index=2,
                text="Chunk two content about distributed consensus.",
                char_count=46,
            ),
            DocumentChunk(
                chunk_id="doc_p2_c1",
                filename="doc.pdf",
                page_number=2,
                chunk_index=1,
                text="Chunk three content on page two.",
                char_count=33,
            ),
        ]

        results = await mock_service.embed_chunks(chunks)

        assert len(results) == 3
        for chunk, result in zip(chunks, results):
            assert result.chunk_id == chunk.chunk_id
            assert result.page_number == chunk.page_number
            assert len(result.vector) == 384
            assert chunk.embedding == result.vector

    @pytest.mark.asyncio
    async def test_empty_chunks_safely_handled(self, mock_service):
        # Empty chunk raises EmptyChunkError on single embed
        empty_chunk = DocumentChunk(
            chunk_id="empty_c1",
            filename="empty.pdf",
            page_number=1,
            chunk_index=1,
            text="   ",
            char_count=0,
        )
        with pytest.raises(EmptyChunkError):
            await mock_service.embed_chunk(empty_chunk)

        # Batch embed filters out empty chunks safely
        mixed_chunks = [
            DocumentChunk(
                chunk_id="valid_c1",
                filename="doc.pdf",
                page_number=1,
                chunk_index=1,
                text="Valid chunk content.",
                char_count=20,
            ),
            DocumentChunk(
                chunk_id="empty_c2",
                filename="doc.pdf",
                page_number=1,
                chunk_index=2,
                text="",
                char_count=0,
            ),
        ]
        results = await mock_service.embed_chunks(mixed_chunks)
        assert len(results) == 1
        assert results[0].chunk_id == "valid_c1"

    @pytest.mark.asyncio
    async def test_model_info_reporting(self, mock_service):
        info = mock_service.get_model_info()
        assert info["dimension"] == 384
        assert "mock" in info["model"].lower()


class TestEmbeddingEndpoints:
    """API endpoint tests using mock embedding provider."""

    @pytest.fixture(autouse=True)
    def use_mock_provider(self, monkeypatch):
        # Force mock provider during test client requests
        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")

    def test_embed_preview_endpoint(self):
        response = client.post(
            "/api/documents/embed-preview",
            json={
                "text": "Machine learning systems and natural language processing.",
                "filename": "guide.pdf",
                "page_number": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["filename"] == "guide.pdf"
        assert data["page_number"] == 2
        assert data["dimension"] == 384
        assert data["full_vector_length"] == 384
        assert len(data["vector_preview"]) == 5
        assert data["chunk_id"] == "guide_pdf_p2_c1"

    def test_embed_preview_endpoint_empty_text_rejected(self):
        response = client.post(
            "/api/documents/embed-preview",
            json={
                "text": "   ",
                "filename": "empty.pdf",
                "page_number": 1,
            },
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_embed_chunks_endpoint(self):
        chunks_payload = [
            {
                "chunk_id": "test_p1_c1",
                "filename": "test.pdf",
                "page_number": 1,
                "chunk_index": 1,
                "text": "First chunk text about vector similarity.",
                "char_count": 40,
                "start_char": 0,
                "end_char": 40,
            },
            {
                "chunk_id": "test_p1_c2",
                "filename": "test.pdf",
                "page_number": 1,
                "chunk_index": 2,
                "text": "Second chunk text about cosine distance.",
                "char_count": 40,
                "start_char": 30,
                "end_char": 70,
            },
        ]

        response = client.post(
            "/api/documents/embed-chunks",
            json={"chunks": chunks_payload},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["total_chunks"] == 2
        assert data["dimension"] == 384
        assert len(data["embeddings"]) == 2
        assert data["embeddings"][0]["chunk_id"] == "test_p1_c1"
        assert data["embeddings"][1]["chunk_id"] == "test_p1_c2"
        assert len(data["embeddings"][0]["vector"]) == 384
