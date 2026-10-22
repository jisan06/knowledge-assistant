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
            elif "reference" in message.lower() and "person" in message.lower():
                # Grounded answer: strictly verify whether reference chunk was retrieved in context
                if "Ismail Jabiullah" in message:
                    return (
                        "Professor Dr. Md. Ismail Jabiullah is listed as a reference in the document "
                        "[Source: Jesan_resume_with_cover_letter.pdf, Page 4]."
                    )
                return (
                    "The uploaded documents do not provide enough information to answer this question. "
                    "Please provide a specific question related to the provided document context."
                )
            elif any(q in message.lower() for q in ["sourdough bread", "chocolate cake", "favorite pet", "dog name", "cat name"]):
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


# 8 extracted chunks from Jesan_resume_with_cover_letter.pdf
JESAN_RESUME_CHUNKS = [
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p1_c1",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=1,
        chunk_index=1,
        text=(
            "November 20, 2024 \nHiring Manager \nBdtask Limited \nSubject: Senior PHP Developer \n"
            "Hi There, \nWith over 5.6 years of development experience and a diverse skill set, "
            "I am well -prepared to make a significant contribution to your company's operations. "
            "My expertise encompasses handling real -time, large, complex projects in both client-based "
            "and product-based development. \nI excel at managing intricate projects and am equally "
            "comfortable assuming leadership responsibilities within a team."
        ),
        char_count=521,
    ),
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p1_c2",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=1,
        chunk_index=2,
        text=(
            "learning opportunities, and knowledge sharing within your esteemed company. \n"
            "Currently, I serve as a Senior Software Engineer at WPXPO, where I have experience in both "
            "backend and frontend development. \nThank you for taking the time to review my credentials "
            "and accomplishments. I look forward to speaking with you about this opportunity. \n"
            "Best Regards, \nJesan Ahmad"
        ),
        char_count=398,
    ),
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p2_c1",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=2,
        chunk_index=1,
        text=(
            "JESAN AHMAD \nAddress: Garden Street, Shyamoli, Dhaka \nMobile No: 01832967276 \n"
            "E-mail: jisanahmed06@gmail.com \nCareer Objective \nAn innovative individual trying to achieve "
            "top-level skills, inventing creative ideas, and new feature development. \n"
            "Employment History \nTotal Time of Experience: 5.6 Years \n1. Sr. Software Engineer \n"
            "Duration : January 2022 – Present \nDepartment : IT \nCompany Name : WPXPO \n"
            "Company Location : House: #1/C, Road -1, Shyamoli, Mohammadpur, Dhaka"
        ),
        char_count=493,
    ),
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p2_c2",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=2,
        chunk_index=2,
        text=(
            "Responsibilities : Lead the product team, Requirement analysis, Planning, Architecture design. \n"
            "2. Software Engineer \nDuration : December 1, 2020 – December 30, 2021 \nDepartment : IT \n"
            "Company Name : Perky Rabbit \nCompany Location : House: #19, Road -16, DIT Project, Merul Badda, Dhaka \n"
            "3. Assistant Programmer \nDuration : March 2, 2019 – November 30, 2020 \nDepartment : IT \n"
            "Company Name : Techno Park Bangladesh"
        ),
        char_count=442,
    ),
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p2_c3",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=2,
        chunk_index=3,
        text=(
            "Assistant Programmer \nDuration : March 2, 2019 – November 30, 2020 \nDepartment : IT \n"
            "Company Name : Techno Park Bangladesh \nCompany Location : House: #23, Road -09, DIT Project, Merul Badda, Dhaka \n"
            "Responsibilities : Customize import management, Create e-commerce website, Simple website"
        ),
        char_count=297,
    ),
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p3_c1",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=3,
        chunk_index=1,
        text=(
            "Skills \nLanguages: PHP, HTML, CSS, JavaScript, SQL \nDatabases: MySQL, MongoDB, MariaDB \n"
            "Frameworks: Laravel, Codeigniter, CakePHP, Node.js, Vue.js, React.js, WordPress \n"
            "Education \nBachelor of Science (BSc) in Computer Science & Engineering \nPassing Year : 2018 \n"
            "Result : 3.48 (out of 4.00) \nInstitute : Hamdard University Bangladesh"
        ),
        char_count=361,
    ),
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p3_c2",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=3,
        chunk_index=2,
        text=(
            "Training Summary \nTitle : Certified Web Development Specialist \n"
            "Topics : HTML5, CSS3, Bootstrap, JavaScript, JQuery, Ajax, PHP, MySQL \n"
            "Duration : 6 Month \nInstitute : BITM \nYear : 2017"
        ),
        char_count=208,
    ),
    DocumentChunk(
        chunk_id="Jesan_resume_with_cover_letter_pdf_p4_c1",
        filename="Jesan_resume_with_cover_letter.pdf",
        page_number=4,
        chunk_index=1,
        text=(
            "Reference \nProfessor Dr. Md. Ismail Jabiullah \n"
            "    Professor, Department of Computer Science and Engineering \n"
            "    Daffodil International University \n \n"
            "Personal Information \nDate of Birth : 06 December 1995 \n"
            "Permanent address : Ghasirchar, Ananda Bazar, Matlab Uttar, Chandpur-3640 \n"
            "Marital Status : Unmarried \nNationality : Bangladeshi \n \n"
            "Some of My Work \nhttps://www.knowwear.shop/en/home \nhttps://wordpress.org/plugins/ultimate-post/ \n"
            "https://wordpress.org/plugins/product-blocks/ \nhttps://enoc.fireservice.gov.bd/login \n"
            "https://fsc.fireservice.gov.bd/ \nhttps://elicense.fireservice.gov.bd/ \n"
            "http://fscws.viewdemo.xyz/login  \nhttps://asgardstore.com/en \nhttp://banglacar.net/"
        ),
        char_count=716,
    ),
]



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

    @pytest.mark.asyncio
    async def test_reference_person_semantic_retrieval_and_logging(self):
        """
        Task 3: Tests semantic retrieval for 'Which person's name is listed as a reference in the document?'.
        Logs the retrieved chunk text, filename, page number, chunk ID, and similarity/distance score.
        Verifies that page 4 reference chunk is successfully retrieved in top results.
        """
        memory_store = InMemoryVectorStore(collection_name="test_resume_retrieval")
        mock_embedding = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        search_service = SearchService(embedding_service=mock_embedding, vector_store=memory_store)

        # Index all 8 chunks of the user's resume
        await search_service.index_chunks(JESAN_RESUME_CHUNKS)

        query = "Which person's name is listed as a reference in the document?"
        search_resp = await search_service.search(query=query, top_k=4)

        print("\n--- [Semantic Retrieval Results for Reference Question] ---")
        retrieved_ids = []
        for rank, item in enumerate(search_resp.results, start=1):
            retrieved_ids.append(item.chunk_id)
            print(
                f"Rank {rank}: Chunk ID: {item.chunk_id} | "
                f"File: {item.filename} | Page: {item.page_number} | "
                f"Similarity: {item.similarity_score} | Distance: {item.distance}\n"
                f"Text: {item.text[:100]}...\n"
            )

        # Confirm all 8 chunks are in the store
        assert search_service.vector_store.count() == 8

        # Confirm page 4 reference chunk is retrieved in the top results
        assert "Jesan_resume_with_cover_letter_pdf_p4_c1" in retrieved_ids
        p4_item = next(
            item for item in search_resp.results
            if item.chunk_id == "Jesan_resume_with_cover_letter_pdf_p4_c1"
        )
        assert p4_item.page_number == 4
        assert "Professor Dr. Md. Ismail Jabiullah" in p4_item.text
        assert "Reference" in p4_item.text

    @pytest.mark.asyncio
    async def test_reference_person_query_retrieval_and_grounded_answer(self):
        """
        Task 7 Regression Test: Verifies that 'Which person's name is listed as a reference in the document?'
        returns 'Professor Dr. Md. Ismail Jabiullah' with an exact citation to page 4.
        """
        memory_store = InMemoryVectorStore(collection_name="test_resume_rag")
        mock_embedding = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        search_service = SearchService(embedding_service=mock_embedding, vector_store=memory_store)

        await search_service.index_chunks(JESAN_RESUME_CHUNKS)

        ai_service = AIService()
        ai_service.provider = MockGroundedAIProvider()
        rag_service = RAGService(search_service=search_service, ai_service=ai_service)

        query = "Which person's name is listed as a reference in the document?"
        answer = await rag_service.chat(query)

        # Must return the expected reference person
        assert "Professor Dr. Md. Ismail Jabiullah" in answer
        # Must cite page 4
        assert "Page 4" in answer or "p. 4" in answer
        assert "Jesan_resume_with_cover_letter.pdf" in answer

    @pytest.mark.asyncio
    async def test_genuinely_unanswerable_question_on_resume_refuses(self):
        """
        Task 8: Verifies that genuinely unanswerable questions on the resume
        strictly trigger refusal and do not fabricate facts or citations.
        """
        memory_store = InMemoryVectorStore(collection_name="test_resume_unanswerable")
        mock_embedding = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        search_service = SearchService(embedding_service=mock_embedding, vector_store=memory_store)

        await search_service.index_chunks(JESAN_RESUME_CHUNKS)

        ai_service = AIService()
        ai_service.provider = MockGroundedAIProvider()
        rag_service = RAGService(search_service=search_service, ai_service=ai_service)

        answer = await rag_service.chat("What is Jesan's favorite pet or dog name?")
        assert "do not provide enough information" in answer.lower()
        # Must not fabricate page 4 citation for unanswerable question
        assert "Page 4" not in answer

    @pytest.mark.asyncio
    async def test_missing_documents_notice_for_reference_query(self):
        """
        Task 8: Verifies that asking the reference query with zero indexed documents
        returns the clear missing documents guidance message upfront without calling LLM.
        """
        empty_store = InMemoryVectorStore(collection_name="test_empty_resume_store")
        mock_embedding = EmbeddingService(provider=MockEmbeddingProvider(dimension=384))
        search_service = SearchService(embedding_service=mock_embedding, vector_store=empty_store)

        rag_service = RAGService(search_service=search_service)

        answer = await rag_service.chat("Which person's name is listed as a reference in the document?")
        assert "no indexed documents" in answer.lower()
        assert "upload a pdf" in answer.lower()



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
        from app.vector_store.factory import get_vector_store
        get_vector_store().clear()

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

    def test_rag_streaming_resume_reference_query(self, monkeypatch):
        """
        Task 7 & 8: End-to-end streaming test for 'Which person's name is listed as a reference in the document?'
        Verifies that streaming yields Professor Dr. Md. Ismail Jabiullah with source citation to page 4.
        """
        # 1. Index all 8 chunks of Jesan's resume
        chunks_payload = {
            "chunks": [c.model_dump() for c in JESAN_RESUME_CHUNKS]
        }
        idx_res = client.post("/api/documents/index-chunks", json=chunks_payload)
        assert idx_res.status_code == 200

        # 2. Mock AIService provider with MockGroundedAIProvider
        from app.routes import conversations as conv_module
        monkeypatch.setattr(conv_module.rag_service.ai_service, "provider", MockGroundedAIProvider())

        # 3. Create conversation and stream message with use_rag=True
        conv_resp = client.post("/api/conversations", json={"title": "Resume Reference Stream Test"})
        conv_id = conv_resp.json()["id"]

        with client.stream(
            "POST",
            f"/api/conversations/{conv_id}/messages/stream",
            json={
                "message": "Which person's name is listed as a reference in the document?",
                "use_rag": True,
            },
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

            assert "Professor Dr. Md. Ismail Jabiullah" in full_reply
            assert "[Source: Jesan_resume_with_cover_letter.pdf, Page 4]" in full_reply

        # 4. Verify conversation history was persisted
        history = client.get(f"/api/conversations/{conv_id}/messages").json()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert "Professor Dr. Md. Ismail Jabiullah" in history[1]["content"]
        assert "Page 4" in history[1]["content"]

