import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import settings
from app.schemas.document import (
    DocumentUploadResponse,
    ChunkPreviewRequest,
    ChunkPreviewResponse,
)
from app.services.chunking_service import ChunkingService
from app.services.pdf_service import (
    PDFService,
    PDFServiceError,
    FileTooLargeError,
    EmptyFileError,
    InvalidPDFError,
    EncryptedPDFError,
)

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
):
    """
    Accepts a single PDF file, validates format and size limits, and extracts
    text page-by-page in memory.
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
        return result

    except PDFServiceError as e:
        # Expected domain validation errors
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
