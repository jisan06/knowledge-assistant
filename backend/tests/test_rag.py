import io
import json
import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import settings
from app.providers.base import AIProvider
from app.providers.embeddings.mock_provider import MockEmbeddingProvider
from app.schemas.document import DocumentChunk
from app.schemas.search import SearchResultItem
from app.services.ai_service import AIService
from app.services.embedding_service import EmbeddingService
from app.services.rag_service import RAGService
from app.services.search_service import SearchService
from app.vector_store.memory_store import InMemoryVectorStore

client = TestClient(app)


class MockGroundedAIProvider(AIProvider):
    """
    Mock AI Provider simulating Gemini's response following the RAG grounding prompt instructions.
    """

    async def generate(self, message: str) -> str:
        # Check if prompt contains the RAG instructions and document context
        if "=== RETRIEVED DOCUMENT CONTEXT ===" in message:
            # Check for specific user questions in the prompt
            if "Python experience" in message:
                return (
                    "Alice has 7 years of experience in Python and building scalable FastAPI microservices "
                    "[Source: resume.pdf, Page 1]."
                )
            elif "sourdough bread" in message or "chocolate cake" in message:
                # Following instruction 3: admit when context doesn't contain the answer
                return "The uploaded documents do not provide enough information to answer this question."
            else:
                return "The answer is synthesized from the excerpts [Source: doc.pdf, Page 1]."
        # Ordinary chat fallback
        return f"Ordinary AI assistant reply to: {message}"

    async def generate_stream(self, message: str):
        full_text = await self.generate(message)
        # Yield words as stream chunks
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


# ---------------------------------------------------------------------------
# Unit Tests: RAGService Prompt Construction & Logic
# ---------------------------------------------------------------------------

class TestRAGServiceUnit:
    """Verifies RAG prompt construction, citations preservation, and context retrieval."""

    def test_prompt_formatting_preserves_metadata(self):
        chunks = [
            SearchResultItem(
                chunk_id="handbook_pdf_p3_c1",
                text="Employees are entitled to 20 days of annual leave accrued monthly.",
                filename="handbook.pdf",
                page_number=3,
                chunk_index=1,
                similarity_score=0.92,
            ),
            SearchResultItem(
                chunk_id="handbook_pdf_p5_c2",
                text="Remote work policy requires core team availability between 10am and 4pm EST.",
                filename="handbook.pdf",
                page_number=5,
                chunk_index=2,
                similarity_score=0.88,
            ),
        ]

        prompt = RAGService.format_rag_prompt(
            query="What is the remote work core hours policy?",
            chunks=chunks,
        )

        # 1. Verify prompt contains source filenames, page numbers, and chunk IDs
        assert "handbook.pdf" in prompt
        assert "Page 3" in prompt
        assert "Page 5" in prompt
        assert "handbook_pdf_p3_c1" in prompt
        assert "handbook_pdf_p5_c2" in prompt

        # 2. Verify prompt contains strict anti-hallucination instructions
        assert "MANDATORY INSTRUCTIONS" in prompt
        assert "cite" in prompt.lower()
        assert "do not provide enough information" in prompt

    @pytest.mark.asyncio
    async def test_empty_collection_returns_clear_notice_without_ai_call(self):
        empty_store = InMemoryVectorStore(collection_name="test_empty_rag")
        mock_embedding = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        search_service = SearchService(embedding_service=mock_embedding, vector_store=empty_store)

        rag_service = RAGService(search_service=search_service)

        # 1. Non-streaming call
        answer = await rag_service.chat("What is the refund policy?")
        assert "no indexed documents" in answer.lower()
        assert "upload a pdf" in answer.lower()

        # 2. Streaming call
        chunks = []
        async for chunk in rag_service.chat_stream("What is the refund policy?"):
            chunks.append(chunk)
        streamed = "".join(chunks)
        assert "no indexed documents" in streamed.lower()

    @pytest.mark.asyncio
    async def test_grounded_answer_with_explicit_citations(self):
        """Verifies grounded answer generation with mock AI provider."""
        memory_store = InMemoryVectorStore(collection_name="test_grounded_rag")
        mock_embedding = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        search_service = SearchService(embedding_service=mock_embedding, vector_store=memory_store)

        # Store a chunk
        sample_chunk = DocumentChunk(
            chunk_id="resume_pdf_p1_c1",
            filename="resume.pdf",
            page_number=1,
            chunk_index=1,
            text="Alice has 7 years of experience in Python and building scalable FastAPI microservices.",
            char_count=87,
        )
        await search_service.index_chunks([sample_chunk])

        # Instantiate RAGService with MockGroundedAIProvider
        ai_service = AIService()
        ai_service.provider = MockGroundedAIProvider()
        rag_service = RAGService(search_service=search_service, ai_service=ai_service)

        # Ask question explicitly present in document
        answer = await rag_service.chat("How many years of Python experience does Alice have?")
        assert "7 years" in answer
        assert "[Source: resume.pdf, Page 1]" in answer

    @pytest.mark.asyncio
    async def test_unrelated_question_rejects_hallucinations(self):
        """Verifies that questions unrelated to the document are acknowledged as unsupported."""
        memory_store = InMemoryVectorStore(collection_name="test_unrelated_rag")
        mock_embedding = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        search_service = SearchService(embedding_service=mock_embedding, vector_store=memory_store)

        # Store a tech resume chunk
        sample_chunk = DocumentChunk(
            chunk_id="resume_pdf_p1_c1",
            filename="resume.pdf",
            page_number=1,
            chunk_index=1,
            text="Alice is a cloud infrastructure engineer specializing in Kubernetes and Terraform.",
            char_count=85,
        )
        await search_service.index_chunks([sample_chunk])

        ai_service = AIService()
        ai_service.provider = MockGroundedAIProvider()
        rag_service = RAGService(search_service=search_service, ai_service=ai_service)

        # Ask unrelated question
        answer = await rag_service.chat("What is the recipe for baking sourdough bread?")
        assert "do not provide enough information" in answer.lower()
        # Verify no fabricated citations
        assert "Page 1" not in answer


# ---------------------------------------------------------------------------
# Integration Tests: Chat Endpoints with RAG Mode Enabled and Disabled
# ---------------------------------------------------------------------------

class TestChatRAGIntegration:
    """Verifies conversations streaming and non-streaming with use_rag toggle."""

    @pytest.fixture(autouse=True)
    def clean_slate(self):
        client.delete("/api/documents/collection")
        yield
        client.delete("/api/documents/collection")

    def test_rag_disabled_preserves_ordinary_chat_behavior(self, monkeypatch):
        # Create a conversation
        conv_resp = client.post("/api/conversations", json={"title": "Ordinary Chat"})
        conv_id = conv_resp.json()["id"]

        # Mock AIService provider
        from app.routes import conversations as conv_module
        monkeypatch.setattr(conv_module.ai_service, "provider", MockGroundedAIProvider())

        # Send with use_rag=False (ordinary mode)
        resp = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"message": "Hello AI assistant!", "use_rag": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Ordinary AI assistant reply" in data["assistant_message"]["content"]

    def test_rag_enabled_empty_documents_notice(self, monkeypatch):
        # Empty collection
        conv_resp = client.post("/api/conversations", json={"title": "RAG Test"})
        conv_id = conv_resp.json()["id"]

        resp = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"message": "What is in the document?", "use_rag": True},
        )
        assert resp.status_code == 200
        content = resp.json()["assistant_message"]["content"]
        assert "no indexed documents" in content.lower()

    def test_rag_streaming_with_citations(self, monkeypatch):
        # 1. Index sample document chunks
        chunks_payload = {
            "chunks": [
                {
                    "chunk_id": "resume_p1_c1",
                    "filename": "resume.pdf",
                    "page_number": 1,
                    "chunk_index": 1,
                    "text": "Alice has 7 years of experience in Python and FastAPI.",
                    "char_count": 55,
                }
            ]
        }
        idx_res = client.post("/api/documents/index-chunks", json=chunks_payload)
        assert idx_res.status_code == 200

        # 2. Mock providers
        from app.routes import conversations as conv_module
        monkeypatch.setattr(conv_module.rag_service.ai_service, "provider", MockGroundedAIProvider())

        # 3. Create conversation and stream message with use_rag=True
        conv_resp = client.post("/api/conversations", json={"title": "RAG Stream Test"})
        conv_id = conv_resp.json()["id"]

        with client.stream(
            "POST",
            f"/api/conversations/{conv_id}/messages/stream",
            json={"message": "How much Python experience does Alice have?", "use_rag": True},
        ) as response:
            assert response.status_code == 200
            events = response.iter_lines()
            full_reply = ""
            for line in events:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "chunk" in data:
                            full_reply += data["chunk"]
                    except json.JSONDecodeError:
                        pass

            assert "7 years" in full_reply
            assert "[Source: resume.pdf, Page 1]" in full_reply

        # 4. Verify conversation history in SQLite was persisted
        history = client.get(f"/api/conversations/{conv_id}/messages").json()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert "7 years" in history[1]["content"]

    def test_direct_ask_endpoint(self, monkeypatch):
        # 1. Index document
        chunks_payload = {
            "chunks": [
                {
                    "chunk_id": "spec_p2_c1",
                    "filename": "spec.pdf",
                    "page_number": 2,
                    "chunk_index": 1,
                    "text": "The knowledge assistant uses ChromaDB and BGE embeddings.",
                    "char_count": 57,
                }
            ]
        }
        client.post("/api/documents/index-chunks", json=chunks_payload)

        # 2. Test /api/documents/ask
        from app.routes import documents as docs_module
        from app.services.rag_service import RAGService

        # Mock AIService inside RAG
        mock_rag = RAGService()
        mock_rag.ai_service.provider = MockGroundedAIProvider()
        monkeypatch.setattr(docs_module, "RAGService", lambda: mock_rag)

        ask_res = client.post(
            "/api/documents/ask",
            json={"question": "What database does the knowledge assistant use?"},
        )
        assert ask_res.status_code == 200
        res_data = ask_res.json()
        assert "question" in res_data
        assert "answer" in res_data
        assert "sources" in res_data
        assert len(res_data["sources"]) >= 1
        assert res_data["sources"][0]["filename"] == "spec.pdf"
        assert res_data["sources"][0]["page_number"] == 2
