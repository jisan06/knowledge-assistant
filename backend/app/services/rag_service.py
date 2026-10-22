import logging
from typing import AsyncGenerator, List, Optional, Tuple

from app.config import settings
from app.schemas.search import SearchResultItem
from app.services.ai_service import AIService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service integrating document vector retrieval with the AI chat model (RAG).
    
    Responsibilities:
    - Retrieving the top matching chunks for user questions using the existing SearchService.
    - Formatting retrieved chunk text alongside source filename, original page number, and chunk ID.
    - Prompting the AI model with strict grounding rules: cite sources, avoid hallucinations,
      and admit when information is missing.
    - Providing both streaming and non-streaming responses.
    """

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        ai_service: Optional[AIService] = None,
    ):
        self.search_service = search_service or SearchService()
        self.ai_service = ai_service or AIService()

    @staticmethod
    def format_rag_prompt(query: str, chunks: List[SearchResultItem]) -> str:
        """
        Builds a grounded prompt containing retrieved excerpts and strict citation instructions.
        """
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            context_blocks.append(
                f"--- [Source {i}: {chunk.filename} | Page {chunk.page_number} | Chunk ID: {chunk.chunk_id}] ---\n"
                f"{chunk.text.strip()}"
            )
        formatted_context = "\n\n".join(context_blocks)

        return (
            "You are a trustworthy AI knowledge assistant. Your task is to answer the user's question "
            "using ONLY the retrieved document excerpts provided below.\n\n"
            "=== RETRIEVED DOCUMENT CONTEXT ===\n"
            f"{formatted_context}\n"
            "=== END OF CONTEXT ===\n\n"
            "=== MANDATORY INSTRUCTIONS ===\n"
            "1. Base your answer strictly on the facts present in the RETRIEVED DOCUMENT CONTEXT above.\n"
            "2. For every claim or key fact you state, cite your source with an exact citation citing the filename and page number, "
            "for example: [Source: filename.pdf, Page 1] or (filename.pdf, p. 1).\n"
            "3. If the provided context does NOT contain enough information to answer the question, or if the question is "
            "unrelated to the context, clearly state: 'The uploaded documents do not provide enough information to answer this question.' "
            "Do NOT attempt to guess, extrapolate, or fabricate any information or citations.\n"
            "4. Never invent page numbers, filenames, or facts not present in the retrieved context.\n"
            "5. Distinguish clearly between what the sources explicitly state and what is missing or unsupported.\n"
            "6. Keep your answer factual, clear, and direct.\n\n"
            f"User Question: {query}\n\n"
            "Grounded Answer:"
        )

    async def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[SearchResultItem]:
        """Retrieves top-k relevant chunks for the user query."""
        k = top_k or settings.SEARCH_TOP_K
        search_resp = await self.search_service.search(query=query, top_k=k)
        return search_resp.results

    async def chat(self, message: str, top_k: Optional[int] = None) -> str:
        """Non-streaming RAG question answering."""
        clean_query = (message or "").strip()
        if not clean_query:
            return "Please provide a valid question."

        if self.search_service.vector_store.count() == 0:
            return (
                "The knowledge base currently has no indexed documents. "
                "Please upload a PDF in the Documents section to enable document-grounded answers."
            )

        try:
            chunks = await self.retrieve_context(clean_query, top_k=top_k)
        except Exception as e:
            logger.exception("Error retrieving document chunks for RAG")
            return "An error occurred while retrieving document context. Please try again."

        if not chunks:
            return (
                "The uploaded documents do not contain any relevant information to answer this question."
            )

        rag_prompt = self.format_rag_prompt(clean_query, chunks)
        return await self.ai_service.chat(rag_prompt)

    async def chat_stream(
        self,
        message: str,
        top_k: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming RAG question answering via generator chunks."""
        clean_query = (message or "").strip()
        if not clean_query:
            yield "Please provide a valid question."
            return

        if self.search_service.vector_store.count() == 0:
            yield (
                "The knowledge base currently has no indexed documents. "
                "Please upload a PDF in the Documents section to enable document-grounded answers."
            )
            return

        try:
            chunks = await self.retrieve_context(clean_query, top_k=top_k)
        except Exception as e:
            logger.exception("Error retrieving document chunks for RAG")
            yield "An error occurred while retrieving document context. Please try again."
            return

        if not chunks:
            yield (
                "The uploaded documents do not contain any relevant information to answer this question."
            )
            return

        rag_prompt = self.format_rag_prompt(clean_query, chunks)
        async for chunk in self.ai_service.chat_stream(rag_prompt):
            yield chunk

    async def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Tuple[str, List[SearchResultItem]]:
        """Answers a question and returns both the answer text and the retrieved sources."""
        clean_query = (question or "").strip()
        if not clean_query:
            return "Please provide a valid question.", []

        if self.search_service.vector_store.count() == 0:
            return (
                "The knowledge base currently has no indexed documents. "
                "Please upload a PDF in the Documents section to enable document-grounded answers.",
                [],
            )

        chunks = await self.retrieve_context(clean_query, top_k=top_k)
        if not chunks:
            return (
                "The uploaded documents do not contain any relevant information to answer this question.",
                [],
            )

        rag_prompt = self.format_rag_prompt(clean_query, chunks)
        answer = await self.ai_service.chat(rag_prompt)
        return answer, chunks
