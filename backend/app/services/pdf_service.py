import io
import os
import re
from typing import Optional
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

from app.config import settings
from app.schemas.document import DocumentPage, DocumentUploadResponse
from app.services.chunking_service import ChunkingService


class PDFServiceError(Exception):
    """Base exception for PDF processing errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmptyFileError(PDFServiceError):
    def __init__(self, message: str = "Uploaded file is empty."):
        super().__init__(message, status_code=400)


class FileTooLargeError(PDFServiceError):
    def __init__(self, max_bytes: int):
        max_mb = max_bytes // (1024 * 1024)
        super().__init__(
            f"File size exceeds the maximum limit of {max_mb} MB.",
            status_code=413,
        )


class InvalidPDFError(PDFServiceError):
    def __init__(self, message: str = "The uploaded file is not a valid PDF document or is corrupted."):
        super().__init__(message, status_code=422)


class EncryptedPDFError(PDFServiceError):
    def __init__(self, message: str = "Password-protected or encrypted PDFs are not supported."):
        super().__init__(message, status_code=400)


class PDFService:
    """
    Dedicated service for safe, in-memory PDF inspection and text extraction.
    Ensures uploaded files are never executed, written to disk, or trusted blindly.
    """

    @staticmethod
    def sanitize_filename(filename: Optional[str]) -> str:
        """
        Sanitize user-provided filename to prevent path traversal or injection.
        Only keeps the base name and cleans dangerous characters.
        """
        if not filename:
            return "document.pdf"
        # Extract base name to prevent directory traversal
        base_name = os.path.basename(filename)
        # Strip non-whitelisted characters, keeping alphanumeric, dots, underscores, dashes, and spaces
        cleaned = re.sub(r"[^\w\s.-]", "", base_name).strip()
        return cleaned or "document.pdf"

    @classmethod
    def validate_file_bytes(cls, content: bytes, max_bytes: Optional[int] = None) -> None:
        """
        Validates raw file bytes for size, non-emptiness, and PDF header magic bytes.
        """
        limit = max_bytes if max_bytes is not None else settings.MAX_UPLOAD_SIZE_BYTES

        if not content or len(content) == 0:
            raise EmptyFileError("Uploaded file is empty. Please select a valid PDF.")

        if len(content) > limit:
            raise FileTooLargeError(limit)

        # Validate PDF magic bytes: standard PDFs begin with %PDF
        # Stripping leading whitespace/BOM just in case
        stripped_prefix = content.lstrip()[:5]
        if not stripped_prefix.startswith(b"%PDF"):
            raise InvalidPDFError(
                "Invalid file format. The file header does not match a standard PDF document."
            )

    @classmethod
    def extract_text(
        cls,
        file_bytes: bytes,
        raw_filename: Optional[str],
        max_bytes: Optional[int] = None,
    ) -> DocumentUploadResponse:
        """
        Extracts text page-by-page from raw PDF bytes strictly in memory.
        Preserves 1-based page numbers and returns extraction status.
        """
        cls.validate_file_bytes(file_bytes, max_bytes=max_bytes)
        safe_filename = cls.sanitize_filename(raw_filename)

        stream = io.BytesIO(file_bytes)
        try:
            reader = PdfReader(stream, strict=False)
        except (PdfReadError, PyPdfError, ValueError, Exception):
            raise InvalidPDFError(
                "Unable to parse PDF document. The file may be corrupted or malformed."
            )

        # Check for encryption
        try:
            if reader.is_encrypted:
                raise EncryptedPDFError("Encrypted or password-protected PDFs are not supported.")
        except EncryptedPDFError:
            raise
        except Exception:
            raise InvalidPDFError("Failed to inspect PDF security attributes.")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise InvalidPDFError("The uploaded PDF contains no pages.")

        pages = []
        total_characters = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            char_count = len(page_text.strip())
            total_characters += char_count

            pages.append(
                DocumentPage(
                    page_number=page_idx,
                    text=page_text,
                    char_count=char_count,
                )
            )

        has_extractable_text = total_characters > 0

        # Generate page-isolated text chunks
        chunks = ChunkingService.chunk_document_pages(
            pages=pages,
            filename=safe_filename,
        )
        total_chunks = len(chunks)

        if has_extractable_text:
            message = (
                f"Successfully extracted text from {total_pages} page(s) "
                f"({total_characters} characters, {total_chunks} chunks)."
            )
        else:
            message = (
                "Document parsed successfully, but contains no extractable text. "
                "This document may be an image-only or scanned PDF."
            )

        return DocumentUploadResponse(
            filename=safe_filename,
            total_pages=total_pages,
            total_characters=total_characters,
            has_extractable_text=has_extractable_text,
            pages=pages,
            total_chunks=total_chunks,
            chunks=chunks,
            message=message,
        )
