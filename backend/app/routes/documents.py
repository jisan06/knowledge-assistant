import logging
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.config import settings
from app.schemas.document import (
    DocumentUploadResponse,
    ChunkPreviewRequest,
    ChunkPreviewResponse,
    EmbedPreviewRequest,
    EmbedPreviewResponse,
    EmbedChunksRequest,
    EmbedChunksResponse,
)
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    IndexDocumentResponse,
    CollectionStatsResponse,
    AskQuestionRequest,
    AskQuestionResponse,
)
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
)
from app.services.pdf_service import (
    PDFService,
    PDFServiceError,
    FileTooLargeError,
    EmptyFileError,
    InvalidPDFError,
    EncryptedPDFError,
)
from app.services.search_service import SearchService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF and extract text page by page",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to parse and extract"),
    embed: bool = False,
    index: bool = False,
):
    """
    Accepts a single PDF file, validates format and size limits, and extracts
    text page-by-page in memory. If embed=True, generates embeddings for all chunks.
    """
    # Quick client-provided metadata validation
    filename = file.filename or ""
    content_type = (file.content_type or "").lower()

    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported.",
        )

    # Pre-check upload size if provided by the client header before reading into RAM
    if file.size is not None and file.size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum limit of {max_mb} MB.",
        )

    # Read uploaded bytes with size enforcement
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded file stream.",
        )
    finally:
        await file.close()

    # Process PDF strictly in memory
    try:
        result = PDFService.extract_text(
            file_bytes=content,
            raw_filename=filename,
            max_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        )

        if index and result.has_extractable_text and result.chunks:
            search_service = SearchService()
            await search_service.index_chunks(result.chunks)
        elif embed and result.has_extractable_text and result.chunks:
            embedding_service = EmbeddingService()
            await embedding_service.embed_chunks(result.chunks)

        return result

    except PDFServiceError as e:
        # Expected domain validation errors
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
        )

    except EmbeddingServiceError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
        )

    except Exception as e:
        # Catch-all to never expose internal exceptions or stack traces
        logger.exception("Unexpected error occurred while processing PDF document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the PDF file. Please try again.",
        )


@router.post(
    "/chunk-preview",
    response_model=ChunkPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview text chunking for arbitrary text without uploading a PDF",
)
def preview_chunks(payload: ChunkPreviewRequest):
    """
    Splits arbitrary text into chunks of ~800 chars with ~150 chars overlap.
    Useful for inspecting chunking behavior on short, long, or empty text.
    """
    chunks = ChunkingService.chunk_page(
        text=payload.text,
        filename=payload.filename,
        page_number=payload.page_number,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    return ChunkPreviewResponse(
        filename=payload.filename,
        page_number=payload.page_number,
        total_characters=len(payload.text.strip()) if payload.text else 0,
        total_chunks=len(chunks),
        chunks=chunks,
    )


@router.post(
    "/embed-preview",
    response_model=EmbedPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Test embedding generation and inspect vector dimensions",
)
async def preview_embedding(payload: EmbedPreviewRequest):
    """
    Generates an embedding vector for arbitrary text to inspect model name,
    vector dimensionality, and preview the first few float components.
    """
    embedding_service = EmbeddingService()
    try:
        vector = await embedding_service.embed_text(payload.text)
        clean_name = ChunkingService.clean_filename_for_id(payload.filename)
        chunk_id = f"{clean_name}_p{payload.page_number}_c1"
        return EmbedPreviewResponse(
            chunk_id=chunk_id,
            filename=payload.filename,
            page_number=payload.page_number,
            model=embedding_service.model_name,
            dimension=len(vector),
            vector_preview=vector[:5],
            full_vector_length=len(vector),
        )
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in embed preview")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/embed-chunks",
    response_model=EmbedChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Convert a list of DocumentChunk items into numerical embeddings",
)
async def embed_chunks_endpoint(payload: EmbedChunksRequest):
    """
    Generates embedding vectors for a list of DocumentChunks in batch,
    preserving each chunk's ID, filename, and page number.
    """
    embedding_service = EmbeddingService()
    try:
        results = await embedding_service.embed_chunks(payload.chunks)
        return EmbedChunksResponse(
            model=embedding_service.model_name,
            dimension=embedding_service.dimension,
            total_chunks=len(results),
            embeddings=results,
        )
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error in batch embed chunks")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic search across indexed document chunks using vector similarity",
)
async def semantic_search(payload: SearchRequest):
    """
    Accepts a user text query, generates an embedding using the configured model,
    and returns top-k most similar document chunks with similarity/distance metrics.
    """
    clean_query = (payload.query or "").strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace only.",
        )

    search_service = SearchService()
    try:
        return await search_service.search(
            query=clean_query,
            top_k=payload.top_k,
            filter_filename=payload.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error during semantic search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )


@router.get(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Convenience GET endpoint for semantic search via browser or query params",
)
async def semantic_search_get(
    q: str = Query(..., min_length=1, description="Search query string"),
    top_k: int = Query(default=4, ge=1, le=20, description="Number of results (3-5 typical)"),
    filename: Optional[str] = Query(default=None, description="Optional document filename filter"),
):
    """
    GET version of semantic search allowing testing directly from the browser URL bar.
    """
    clean_query = (q or "").strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace only.",
        )

    search_service = SearchService()
    try:
        return await search_service.search(
            query=clean_query,
            top_k=top_k,
            filter_filename=filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error during semantic search GET")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )


@router.post(
    "/index",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a PDF and index its chunks into the vector database",
)
async def index_pdf_document(
    file: UploadFile = File(..., description="PDF document to parse, chunk, embed, and store in vector index"),
):
    """
    End-to-end endpoint: validates PDF, extracts text page-by-page, generates
    chunks with stable IDs, embeds them, and upserts them into vector storage.
    """
    filename = file.filename or ""
    content_type = (file.content_type or "").lower()

    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported.",
        )

    if file.size is not None and file.size > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum limit of {max_mb} MB.",
        )

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded file stream.",
        )
    finally:
        await file.close()

    try:
        result = PDFService.extract_text(
            file_bytes=content,
            raw_filename=filename,
            max_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        )

        if not result.has_extractable_text or not result.chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF contains no extractable text to index.",
            )

        search_service = SearchService()
        indexed_count = await search_service.index_chunks(result.chunks)

        return IndexDocumentResponse(
            filename=result.filename,
            total_pages=result.total_pages,
            total_chunks=len(result.chunks),
            indexed_chunks=indexed_count,
            message=(
                f"Successfully indexed {indexed_count} chunks from '{result.filename}' "
                f"across {result.total_pages} page(s)."
            ),
        )

    except (PDFServiceError, EmbeddingServiceError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error while indexing PDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


@router.post(
    "/index-chunks",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Index an existing list of DocumentChunk items directly into the vector store",
)
async def index_chunks_endpoint(payload: EmbedChunksRequest):
    """
    Embeds and upserts provided DocumentChunk items into the vector store.
    """
    if not payload.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunk list cannot be empty.",
        )

    search_service = SearchService()
    try:
        indexed_count = await search_service.index_chunks(payload.chunks)
        filename = payload.chunks[0].filename if payload.chunks else "unknown"
        return IndexDocumentResponse(
            filename=filename,
            total_pages=len(set(c.page_number for c in payload.chunks)),
            total_chunks=len(payload.chunks),
            indexed_chunks=indexed_count,
            message=f"Successfully indexed {indexed_count} chunk(s).",
        )
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error while indexing chunks")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing chunks failed: {str(e)}",
        )


@router.get(
    "/collection-stats",
    response_model=CollectionStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current vector store collection statistics",
)
def get_collection_stats():
    """
    Returns the collection name, total indexed chunks, and active vector store backend.
    """
    search_service = SearchService()
    return search_service.get_stats()


@router.delete(
    "/collection",
    status_code=status.HTTP_200_OK,
    summary="Clear all indexed chunks from the vector store collection",
)
def clear_collection():
    """
    Clears all chunk embeddings from the vector store. Useful for testing and resets.
    """
    search_service = SearchService()
    search_service.clear()
    return {"status": "success", "message": "Vector collection cleared successfully."}


@router.post(
    "/ask",
    response_model=AskQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a document-grounded question with retrieved citations using RAG",
)
async def ask_documents(payload: AskQuestionRequest):
    """
    RAG Question-Answering endpoint:
    Retrieves supporting document chunks, prompts the AI model with strict grounding rules,
    and returns a cited answer alongside the preserved source chunks.
    """
    clean_q = (payload.question or "").strip()
    if not clean_q:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    rag_service = RAGService()
    try:
        answer, sources = await rag_service.ask(clean_q, top_k=payload.top_k)
        return AskQuestionResponse(
            question=clean_q,
            answer=answer,
            sources=sources,
        )
    except Exception as e:
        logger.exception("Error during RAG question answering")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question using RAG: {str(e)}",
        )


