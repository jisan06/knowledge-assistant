import io
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from main import app
from app.services.pdf_service import PDFService, FileTooLargeError, EmptyFileError, InvalidPDFError


client = TestClient(app)


def generate_text_pdf(text: str = "Knowledge Assistant Test Document") -> bytes:
    """Generate minimal valid PDF bytes containing text."""
    content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET"
    content_len = len(content)
    raw = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        f"4 0 obj << /Length {content_len} >> stream\n{content}\nendstream\nendobj\n".encode("latin-1")
        + b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000300 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n377\n%%EOF"
    )
    return raw


def generate_blank_pdf() -> bytes:
    """Generate valid PDF without extractable text (simulating image-only/scanned PDF)."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# --- Unit Tests for PDFService ---

def test_sanitize_filename():
    assert PDFService.sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
    assert PDFService.sanitize_filename("C:\\Windows\\System32\\test.pdf") == "test.pdf"
    assert PDFService.sanitize_filename("valid_report-v1.2.pdf") == "valid_report-v1.2.pdf"
    assert PDFService.sanitize_filename(None) == "document.pdf"
    assert PDFService.sanitize_filename("") == "document.pdf"


def test_pdf_service_empty_bytes():
    with pytest.raises(EmptyFileError):
        PDFService.extract_text(b"", "empty.pdf")


def test_pdf_service_file_too_large():
    with pytest.raises(FileTooLargeError):
        # Test with 10 bytes limit
        PDFService.extract_text(b"%PDF-1.4 123456789012345", "big.pdf", max_bytes=10)


def test_pdf_service_invalid_magic_bytes():
    with pytest.raises(InvalidPDFError):
        PDFService.extract_text(b"This is a text file", "not_pdf.pdf")


# --- Endpoint Tests: POST /api/documents/upload ---

def test_upload_valid_text_pdf():
    pdf_bytes = generate_text_pdf("Knowledge Assistant Stage 1")
    response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "sample.pdf"
    assert data["total_pages"] == 1
    assert data["has_extractable_text"] is True
    assert data["total_characters"] > 0
    assert len(data["pages"]) == 1

    page = data["pages"][0]
    assert page["page_number"] == 1
    assert "Knowledge Assistant Stage 1" in page["text"]
    assert page["char_count"] > 0


def test_upload_blank_or_image_pdf():
    pdf_bytes = generate_blank_pdf()
    response = client.post(
        "/api/documents/upload",
        files={"file": ("blank.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "blank.pdf"
    assert data["total_pages"] == 1
    assert data["has_extractable_text"] is False
    assert data["total_characters"] == 0
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1
    assert data["pages"][0]["text"] == ""
    assert "no extractable text" in data["message"].lower()


def test_upload_non_pdf_extension_rejected():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("data.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]


def test_upload_empty_pdf_file():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_corrupt_pdf():
    corrupted_bytes = b"%PDF-1.4 corrupt content that cannot be parsed by pypdf"
    response = client.post(
        "/api/documents/upload",
        files={"file": ("corrupt.pdf", corrupted_bytes, "application/pdf")},
    )
    assert response.status_code == 422
    assert "corrupted" in response.json()["detail"].lower() or "unable to parse" in response.json()["detail"].lower()
