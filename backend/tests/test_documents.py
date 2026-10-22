import io
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from main import app
from app.config import settings
from app.services.pdf_service import PDFService, FileTooLargeError, EmptyFileError, InvalidPDFError


client = TestClient(app)


@pytest.fixture(autouse=True)
def use_mock_provider(monkeypatch):
    """Ensure fast, deterministic vector store tests without downloading external weights."""
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    from app.vector_store.factory import get_vector_store
    from app.db.session import SessionLocal, init_db
    from app.db.models import UploadedDocument

    init_db()
    db = SessionLocal()
    try:
        db.query(UploadedDocument).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

    store = get_vector_store()
    yield
    store.clear()

    db = SessionLocal()
    try:
        db.query(UploadedDocument).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def generate_text_pdf(text: str = "Knowledge Assistant Test Document") -> bytes:
    """Generate minimal valid PDF bytes containing text."""
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = f"BT /F1 12 Tf 72 712 Td ({safe_text}) Tj ET\n".encode("latin-1")
    stream_len = len(stream_content)

    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    obj4 = (
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin-1")
        + stream_content
        + b"endstream\nendobj\n"
    )
    obj5 = b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    header = b"%PDF-1.4\n"
    body_objs = [obj1, obj2, obj3, obj4, obj5]

    offsets = []
    current_offset = len(header)
    for obj in body_objs:
        offsets.append(current_offset)
        current_offset += len(obj)

    xref_offset = current_offset
    total_objs = len(body_objs) + 1  # includes object 0

    xref_lines = [f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin-1")]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n".encode("latin-1"))
    xref = b"".join(xref_lines)

    trailer = (
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")

    return header + b"".join(body_objs) + xref + trailer


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
    assert data["total_chunks"] == 1
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "sample_pdf_p1_c1"
    assert data["chunks"][0]["page_number"] == 1
    assert data["chunks"][0]["filename"] == "sample.pdf"
    assert data["indexed_chunks"] == 1
    assert "indexed" in data["message"].lower()

    page = data["pages"][0]
    assert page["page_number"] == 1
    assert "Knowledge Assistant Stage 1" in page["text"]
    assert page["char_count"] > 0
    assert page["chunk_count"] == 1
    assert len(page["chunks"]) == 1


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
    assert data["indexed_chunks"] == 0
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1
    assert data["pages"][0]["text"] == ""
    assert "no extractable text" in data["message"].lower()


def test_upload_with_explicit_index_false():
    """Verifies that passing index=false extracts text without indexing into the vector store."""
    pdf_bytes = generate_text_pdf("Extraction only content without vector storage")
    response = client.post(
        "/api/documents/upload?index=false",
        files={"file": ("extract_only.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_extractable_text"] is True
    assert data["indexed_chunks"] == 0
    assert "indexed" not in data["message"].lower()


@pytest.mark.asyncio
async def test_full_flow_upload_pdf_index_and_rag_retrieval():
    """
    Verifies the complete flow required by the knowledge assistant:
    1. Upload PDF to /api/documents/upload.
    2. Verify chunks are extracted AND automatically stored in the vector store.
    3. Retrieve the chunk through the identical RAGService used by the chat endpoint.
    4. Confirm grounding text, source filename, and page number match.
    """
    from app.services.rag_service import RAGService
    from app.vector_store.factory import get_vector_store

    resume_text = (
        "Jesan is an accomplished Lead Software Architect with extensive experience in "
        "Python, FastAPI microservices, and React Next.js distributed architectures."
    )
    pdf_bytes = generate_text_pdf(resume_text)

    response = client.post(
        "/api/documents/upload",
        files={"file": ("Jesan_resume_with_cover_letter.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "Jesan_resume_with_cover_letter.pdf"
    assert data["has_extractable_text"] is True
    assert data["indexed_chunks"] >= 1
    assert "indexed" in data["message"].lower()

    # Verify vector store holds the newly indexed chunks
    store = get_vector_store()
    assert store.count() >= 1

    # Verify that the same RAGService used by chat retrieves the indexed document
    rag_service = RAGService()
    assert rag_service.search_service.vector_store.count() >= 1

    retrieved = await rag_service.retrieve_context("What is Jesan's architecture experience?", top_k=2)
    assert len(retrieved) >= 1
    top_chunk = retrieved[0]
    assert top_chunk.filename == "Jesan_resume_with_cover_letter.pdf"
    assert top_chunk.page_number == 1
    assert "FastAPI" in top_chunk.text
    assert top_chunk.similarity_score > 0.0


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


# --- Endpoint Tests: POST /api/documents/chunk-preview ---

def test_chunk_preview_endpoint_short_text():
    response = client.post(
        "/api/documents/chunk-preview",
        json={
            "text": "Short sample text that should easily fit into one chunk.",
            "filename": "sample.pdf",
            "page_number": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.pdf"
    assert data["page_number"] == 1
    assert data["total_chunks"] == 1
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "sample_pdf_p1_c1"
    assert data["chunks"][0]["char_count"] > 0


def test_chunk_preview_endpoint_long_text():
    # ~2400 characters of text
    long_text = "This is a comprehensive paragraph about artificial intelligence systems. " * 30
    response = client.post(
        "/api/documents/chunk-preview",
        json={
            "text": long_text,
            "filename": "guide.pdf",
            "page_number": 3,
            "chunk_size": 800,
            "chunk_overlap": 150,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] >= 3
    assert len(data["chunks"]) >= 3
    for chunk in data["chunks"]:
        assert chunk["filename"] == "guide.pdf"
        assert chunk["page_number"] == 3
        assert chunk["char_count"] > 0


def test_chunk_preview_endpoint_empty_text():
    response = client.post(
        "/api/documents/chunk-preview",
        json={
            "text": "   \n\t  ",
            "filename": "blank.pdf",
            "page_number": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] == 0
    assert data["chunks"] == []


# --- Tests for Duplicate PDF Upload Handling ---

def test_duplicate_content_upload_same_filename():
    """
    Verifies that uploading the exact same PDF content twice under the same filename:
    1. Indexes on the first upload (is_duplicate=False).
    2. Detects the duplicate on the second upload (is_duplicate=True).
    3. Does not re-extract or insert duplicate vectors into the vector store.
    """
    from app.vector_store.factory import get_vector_store
    store = get_vector_store()

    pdf_bytes = generate_text_pdf("Unique policy guidelines for Acme employees in 2026.")
    filename = "company_guidelines.pdf"

    # First upload: initial extraction and indexing
    resp1 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["is_duplicate"] is False
    assert data1["is_updated"] is False
    assert data1["indexed_chunks"] >= 1
    assert data1["content_hash"] is not None
    assert "successfully extracted" in data1["message"].lower()

    count_after_first = store.count()
    assert count_after_first >= 1

    # Second upload: identical PDF content uploaded again
    resp2 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Must be marked as duplicate, not updated
    assert data2["is_duplicate"] is True
    assert data2["is_updated"] is False
    assert data2["content_hash"] == data1["content_hash"]
    assert "already been uploaded" in data2["message"].lower() or "skipping" in data2["message"].lower()

    # Vector store count must NOT increase (no duplicate vectors stored)
    assert store.count() == count_after_first


def test_duplicate_content_upload_different_filename():
    """
    Verifies that duplicate detection uses SHA-256 file content hash,
    not merely the filename: identical bytes uploaded with a different filename are caught.
    """
    from app.vector_store.factory import get_vector_store
    store = get_vector_store()

    pdf_bytes = generate_text_pdf("Content duplicated across departments under different names.")

    # Upload under filename A
    resp1 = client.post(
        "/api/documents/upload",
        files={"file": ("department_policy_finance.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["is_duplicate"] is False
    assert data1["is_updated"] is False
    hash1 = data1["content_hash"]

    count_after_first = store.count()

    # Upload identical bytes under filename B
    resp2 = client.post(
        "/api/documents/upload",
        files={"file": ("department_policy_hr.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Caught as duplicate via content hash
    assert data2["is_duplicate"] is True
    assert data2["is_updated"] is False
    assert data2["content_hash"] == hash1
    assert "already been uploaded" in data2["message"].lower() or "skipping" in data2["message"].lower()

    # Vector store count unchanged
    assert store.count() == count_after_first


def test_different_content_upload_same_filename():
    """
    Verifies that different PDF content uploaded under the same filename:
    1. Is treated as an updated document (is_updated=True, is_duplicate=False).
    2. Has a different SHA-256 hash.
    3. Replaces the database record with the new content hash and metadata.
    """
    from app.db.session import SessionLocal
    from app.db.models import UploadedDocument

    filename = "monthly_report.pdf"
    pdf_v1 = generate_text_pdf("Version 1 content: Q1 revenue was 5 million dollars.")
    pdf_v2 = generate_text_pdf("Version 2 updated content: Q2 revenue reached 8 million dollars.")

    # First upload: Version 1
    resp1 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_v1, "application/pdf")},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["is_duplicate"] is False
    assert data1["is_updated"] is False
    hash1 = data1["content_hash"]

    # Second upload: Version 2 (same filename, different content)
    resp2 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_v2, "application/pdf")},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Must be marked as updated
    assert data2["is_duplicate"] is False
    assert data2["is_updated"] is True
    assert data2["content_hash"] != hash1
    assert "updated" in data2["message"].lower()

    # Check database: exactly 1 record for this filename with the new content hash
    db = SessionLocal()
    try:
        docs = db.query(UploadedDocument).filter(UploadedDocument.filename == filename).all()
        assert len(docs) == 1
        assert docs[0].content_hash == data2["content_hash"]
    finally:
        db.close()


def test_update_document_removes_old_chunks_from_vector_store():
    """
    Verifies that updating a document removes stale chunks/vectors from the vector store:
    1. Chunks that only existed in Version 1 are no longer returned in search.
    2. Chunks from Version 2 are searchable.
    """
    from app.vector_store.factory import get_vector_store
    store = get_vector_store()

    filename = "system_architecture.pdf"
    v1_text = "Architecture V1: Monolith subsystem ZEPHYR_ALPHA_CORE with direct database queries."
    v2_text = "Architecture V2: Microservices subsystem BOREAS_BETA_GRID with event streams."

    pdf_v1 = generate_text_pdf(v1_text)
    pdf_v2 = generate_text_pdf(v2_text)

    # Upload Version 1
    resp1 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_v1, "application/pdf")},
    )
    assert resp1.status_code == 200
    assert store.count() == 1

    # Search for V1-specific term
    search_v1 = client.post(
        "/api/documents/search",
        json={"query": "ZEPHYR_ALPHA_CORE", "top_k": 3},
    )
    assert search_v1.status_code == 200
    assert search_v1.json()["total_results"] > 0
    assert "ZEPHYR_ALPHA_CORE" in search_v1.json()["results"][0]["text"]

    # Upload Version 2 (update)
    resp2 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_v2, "application/pdf")},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["is_updated"] is True
    assert store.count() == 1

    # Search for V1-specific term again: must NO LONGER be returned!
    search_old = client.post(
        "/api/documents/search",
        json={"query": "ZEPHYR_ALPHA_CORE", "top_k": 3},
    )
    assert search_old.status_code == 200
    results_old = search_old.json()["results"]
    for item in results_old:
        assert "ZEPHYR_ALPHA_CORE" not in item["text"]

    # Search for V2-specific term: must be returned!
    search_new = client.post(
        "/api/documents/search",
        json={"query": "BOREAS_BETA_GRID", "top_k": 3},
    )
    assert search_new.status_code == 200
    assert search_new.json()["total_results"] > 0
    assert "BOREAS_BETA_GRID" in search_new.json()["results"][0]["text"]


def test_update_failure_preserves_previous_usable_version(monkeypatch):
    """
    Verifies that when an update fails:
    1. Validation failure (e.g. corrupt PDF) preserves the usable previous version.
    2. Blank PDF without extractable text preserves the previous version.
    3. Storage/upsert failure rolls back and restores previous working vectors.
    """
    from app.vector_store.factory import get_vector_store
    from app.db.session import SessionLocal
    from app.db.models import UploadedDocument
    store = get_vector_store()

    filename = "critical_guide.pdf"
    v1_text = "Guide V1: All production instances require authenticated mTLS certificates."
    pdf_v1 = generate_text_pdf(v1_text)

    # Initial upload of working Version 1
    resp1 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_v1, "application/pdf")},
    )
    assert resp1.status_code == 200
    v1_hash = resp1.json()["content_hash"]
    assert store.count() == 1

    # 1. Failure Mode A: Upload corrupt PDF under same filename
    corrupted_bytes = b"%PDF-1.4 completely corrupt bytes that cannot be parsed"
    resp_fail1 = client.post(
        "/api/documents/upload",
        files={"file": (filename, corrupted_bytes, "application/pdf")},
    )
    assert resp_fail1.status_code == 422

    # Verify Version 1 is still intact in vector store and database
    assert store.count() == 1
    search_check1 = client.post("/api/documents/search", json={"query": "mTLS certificates", "top_k": 2})
    assert search_check1.status_code == 200
    assert "mTLS certificates" in search_check1.json()["results"][0]["text"]

    db = SessionLocal()
    try:
        doc = db.query(UploadedDocument).filter(UploadedDocument.filename == filename).first()
        assert doc is not None
        assert doc.content_hash == v1_hash
    finally:
        db.close()

    # 2. Failure Mode B: Upload blank PDF under same filename
    blank_bytes = generate_blank_pdf()
    resp_fail2 = client.post(
        "/api/documents/upload",
        files={"file": (filename, blank_bytes, "application/pdf")},
    )
    assert resp_fail2.status_code == 400

    # Verify Version 1 is still intact
    assert store.count() == 1
    search_check2 = client.post("/api/documents/search", json={"query": "mTLS certificates", "top_k": 2})
    assert search_check2.status_code == 200
    assert "mTLS certificates" in search_check2.json()["results"][0]["text"]

    # 3. Failure Mode C: Upsert failure during update triggers rollback & restore
    def failing_upsert(*args, **kwargs):
        raise RuntimeError("Simulated vector storage write error during update")

    monkeypatch.setattr(store, "upsert_chunks", failing_upsert)

    v2_text = "Guide V2: Zero trust network access with wireguard tunnels."
    pdf_v2 = generate_text_pdf(v2_text)

    resp_fail3 = client.post(
        "/api/documents/upload",
        files={"file": (filename, pdf_v2, "application/pdf")},
    )
    assert resp_fail3.status_code == 500

    # Restore the original upsert_chunks method for subsequent assertions
    monkeypatch.undo()

    # Verify Version 1 was restored to vector store and remains queryable!
    assert store.count() == 1
    search_check3 = client.post("/api/documents/search", json={"query": "mTLS certificates", "top_k": 2})
    assert search_check3.status_code == 200
    assert len(search_check3.json()["results"]) > 0
    assert "mTLS certificates" in search_check3.json()["results"][0]["text"]

    # Database still points to working V1
    db = SessionLocal()
    try:
        doc = db.query(UploadedDocument).filter(UploadedDocument.filename == filename).first()
        assert doc is not None
        assert doc.content_hash == v1_hash
    finally:
        db.close()


def test_duplicate_pdf_index_endpoint():
    """
    Verifies duplicate detection on the /api/documents/index endpoint.
    """
    pdf_bytes = generate_text_pdf("Testing /api/documents/index duplicate detection behavior.")
    filename = "index_test.pdf"

    resp1 = client.post(
        "/api/documents/index",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["is_duplicate"] is False
    assert data1["is_updated"] is False

    # Second index of same content
    resp2 = client.post(
        "/api/documents/index",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["is_duplicate"] is True
    assert data2["is_updated"] is False
    assert "already exists" in data2["message"].lower() or "skipping" in data2["message"].lower()
    assert data2["content_hash"] == data1["content_hash"]


def test_update_pdf_index_endpoint():
    """
    Verifies updating an existing document on the /api/documents/index endpoint.
    """
    from app.vector_store.factory import get_vector_store
    store = get_vector_store()

    filename = "endpoint_update.pdf"
    pdf_v1 = generate_text_pdf("Index endpoint version 1 content.")
    pdf_v2 = generate_text_pdf("Index endpoint version 2 updated content.")

    resp1 = client.post(
        "/api/documents/index",
        files={"file": (filename, pdf_v1, "application/pdf")},
    )
    assert resp1.status_code == 200
    assert resp1.json()["is_duplicate"] is False
    assert resp1.json()["is_updated"] is False
    assert store.count() == 1

    resp2 = client.post(
        "/api/documents/index",
        files={"file": (filename, pdf_v2, "application/pdf")},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["is_duplicate"] is False
    assert data2["is_updated"] is True
    assert "updated" in data2["message"].lower()
    assert store.count() == 1


# --- Tests for Document Management (Listing & Deletion) ---

def test_list_documents_endpoint():
    """
    Verifies GET /api/documents returns empty list when no documents,
    and populated list with document ID and metadata after upload.
    """
    # 1. Empty initially
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json() == []

    # 2. Upload a document
    pdf_bytes = generate_text_pdf("Document listing verification content.")
    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("list_test.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json().get("id")
    assert doc_id is not None

    # 3. List contains the document
    list_resp = client.get("/api/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert len(docs) == 1
    item = docs[0]
    assert item["id"] == doc_id
    assert item["filename"] == "list_test.pdf"
    assert item["total_pages"] == 1
    assert item["indexed_chunks"] >= 1
    assert item["has_extractable_text"] is True
    assert "created_at" in item


def test_delete_document_by_id_success():
    """
    Verifies DELETE /api/documents/{document_id}:
    1. Removes the document from the database.
    2. Removes all associated chunks from the vector store.
    3. Leaves the document list empty.
    """
    from app.vector_store.factory import get_vector_store
    store = get_vector_store()

    pdf_bytes = generate_text_pdf("Content to be deleted cleanly.")
    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("to_delete.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    assert store.count() >= 1

    # Delete the document by ID
    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["status"] == "success"
    assert del_data["deleted_id"] == doc_id
    assert del_data["deleted_chunks"] >= 1

    # Document list must now be empty
    list_resp = client.get("/api/documents")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 0

    # Vector store count must be 0
    assert store.count() == 0


def test_delete_nonexistent_document_returns_404():
    """
    Verifies DELETE /api/documents/{document_id} returns 404 for missing document IDs.
    """
    resp = client.delete("/api/documents/nonexistent-id-00000000")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_deleted_document_chunks_no_longer_searchable():
    """
    Verifies that deleted document content can no longer appear in retrieval results.
    """
    unique_term = "PROJECT_CYGNUS_NINE_SPEC"
    pdf_bytes = generate_text_pdf(f"Secret operational directives for {unique_term}.")

    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("cygnus.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    # Verify search finds it before deletion
    search_before = client.post("/api/documents/search", json={"query": unique_term, "top_k": 3})
    assert search_before.status_code == 200
    assert search_before.json()["total_results"] > 0
    assert unique_term in search_before.json()["results"][0]["text"]

    # Delete the document
    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200

    # Verify search returns 0 results / no matching chunks after deletion
    search_after = client.post("/api/documents/search", json={"query": unique_term, "top_k": 3})
    assert search_after.status_code == 200
    results = search_after.json()["results"]
    for item in results:
        assert unique_term not in item["text"]


def test_delete_failure_rolls_back_and_preserves_consistency(monkeypatch):
    """
    Verifies that if database deletion fails:
    1. Vector store chunks are restored from backup.
    2. Document remains in database and searchable.
    """
    from app.vector_store.factory import get_vector_store
    store = get_vector_store()

    pdf_bytes = generate_text_pdf("Mission critical document with rollback protection.")
    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("rollback_doc.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    assert store.count() == 1

    # Simulate database commit failure during delete
    from sqlalchemy.orm import Session
    def failing_commit(self):
        raise RuntimeError("Simulated database failure during document deletion")

    monkeypatch.setattr(Session, "commit", failing_commit)

    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 500

    monkeypatch.undo()

    # Verify vector store chunks were restored!
    assert store.count() == 1
    search_resp = client.post("/api/documents/search", json={"query": "rollback protection", "top_k": 2})
    assert search_resp.status_code == 200
    assert len(search_resp.json()["results"]) > 0
    assert "rollback protection" in search_resp.json()["results"][0]["text"]



