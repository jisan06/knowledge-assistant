import pytest

from app.schemas.document import DocumentPage
from app.services.chunking_service import ChunkingService


class TestChunkingService:
    """Comprehensive test suite for ChunkingService."""

    def test_short_page_produces_one_chunk(self):
        """A page shorter than chunk_size (800 chars) must produce exactly one chunk."""
        text = "This is a short document with only a couple sentences. It should fit easily in one chunk."
        chunks = ChunkingService.chunk_page(
            text=text,
            filename="quick_note.pdf",
            page_number=1,
            chunk_size=800,
            chunk_overlap=150,
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_id == "quick_note_pdf_p1_c1"
        assert chunk.filename == "quick_note.pdf"
        assert chunk.page_number == 1
        assert chunk.chunk_index == 1
        assert chunk.text == text
        assert chunk.char_count == len(text)
        assert chunk.start_char == 0
        assert chunk.end_char == len(text)

    def test_long_page_produces_multiple_chunks(self):
        """A page longer than chunk_size (800 chars) must produce multiple chunks."""
        # Generate ~2000 characters of readable text
        paragraph = "Knowledge Assistant is an AI powered system designed to assist users with document processing. " * 20
        assert len(paragraph) > 1500

        chunks = ChunkingService.chunk_page(
            text=paragraph,
            filename="report.pdf",
            page_number=1,
            chunk_size=800,
            chunk_overlap=150,
        )

        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks, start=1):
            assert chunk.filename == "report.pdf"
            assert chunk.page_number == 1
            assert chunk.chunk_index == i
            assert chunk.chunk_id == f"report_pdf_p1_c{i}"
            assert chunk.char_count > 0
            assert chunk.char_count <= 850  # Around 800 chars

    def test_consecutive_chunks_overlap_as_expected(self):
        """Consecutive chunks from the same page must overlap by approximately 150 characters."""
        # Sentence structure text of ~1800 characters
        sentences = [f"Section {i}: This is sentence number {i} providing detailed knowledge." for i in range(1, 40)]
        text = " ".join(sentences)
        assert len(text) > 1600

        chunks = ChunkingService.chunk_page(
            text=text,
            filename="manual.pdf",
            page_number=2,
            chunk_size=800,
            chunk_overlap=150,
        )

        assert len(chunks) >= 2

        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # In terms of character offsets, next chunk should start before current ends
            assert next_chunk.start_char < curr_chunk.end_char
            overlap_length = curr_chunk.end_char - next_chunk.start_char
            # Overlap should be around 150 characters (e.g., between 80 and 200 depending on word boundary snapping)
            assert 80 <= overlap_length <= 220, f"Overlap length was {overlap_length}"

            # Text content check: the suffix of current chunk must appear in the prefix of next chunk
            overlap_text_from_curr = curr_chunk.text[-50:]
            assert overlap_text_from_curr in next_chunk.text

    def test_chunk_retains_correct_page_number_and_filename(self):
        """Every chunk must retain its source filename and 1-based page number."""
        text = "Content from page seven of the document. " * 30
        chunks = ChunkingService.chunk_page(
            text=text,
            filename="annual_review_2026.pdf",
            page_number=7,
        )

        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.filename == "annual_review_2026.pdf"
            assert chunk.page_number == 7
            assert chunk.chunk_id.startswith("annual_review_2026_pdf_p7_c")

    def test_pages_never_merged(self):
        """Text from different pages must NEVER be combined into a single chunk."""
        pages = [
            DocumentPage(
                page_number=1,
                text="Page one text content about astrophysics.",
                char_count=41,
            ),
            DocumentPage(
                page_number=2,
                text="Page two text content about marine biology.",
                char_count=43,
            ),
            DocumentPage(
                page_number=3,
                text="Page three text content about classical literature.",
                char_count=52,
            ),
        ]

        chunks = ChunkingService.chunk_document_pages(pages=pages, filename="multi_subject.pdf")

        assert len(chunks) == 3
        assert chunks[0].page_number == 1
        assert "astrophysics" in chunks[0].text
        assert "marine biology" not in chunks[0].text

        assert chunks[1].page_number == 2
        assert "marine biology" in chunks[1].text
        assert "classical literature" not in chunks[1].text

        assert chunks[2].page_number == 3
        assert "classical literature" in chunks[2].text
        assert "astrophysics" not in chunks[2].text

        # Verify pages updated with their own chunks
        assert pages[0].chunk_count == 1
        assert pages[1].chunk_count == 1
        assert pages[2].chunk_count == 1

    def test_empty_and_whitespace_pages_produce_no_chunks(self):
        """Empty or whitespace-only pages must not produce meaningless chunks."""
        empty_chunks = ChunkingService.chunk_page(
            text="",
            filename="test.pdf",
            page_number=1,
        )
        assert empty_chunks == []

        none_chunks = ChunkingService.chunk_page(
            text=None,
            filename="test.pdf",
            page_number=1,
        )
        assert none_chunks == []

        whitespace_chunks = ChunkingService.chunk_page(
            text="   \n\n\t   \n  ",
            filename="test.pdf",
            page_number=1,
        )
        assert whitespace_chunks == []

    def test_mixed_pages_including_empty_pages(self):
        """Document with blank pages (e.g. image-only pages) handled safely."""
        pages = [
            DocumentPage(page_number=1, text="First page has valid text.", char_count=27),
            DocumentPage(page_number=2, text="   ", char_count=0),  # Blank page
            DocumentPage(page_number=3, text="Third page has valid text.", char_count=27),
        ]

        chunks = ChunkingService.chunk_document_pages(pages=pages, filename="scanned_mix.pdf")

        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 3
        assert pages[1].chunk_count == 0
        assert pages[1].chunks == []

    def test_infinite_loop_prevention_monotonically_increases(self):
        """
        Even with unusual inputs (e.g., continuous repetitive characters without spaces,
        or overlap configuration edge cases), chunking must make forward progress.
        """
        continuous_text = "A" * 3000  # No spaces or punctuation
        chunks = ChunkingService.chunk_page(
            text=continuous_text,
            filename="continuous.pdf",
            page_number=1,
            chunk_size=800,
            chunk_overlap=150,
        )

        assert len(chunks) > 1
        last_start = -1
        for chunk in chunks:
            assert chunk.start_char > last_start, "Chunk start must strictly increase"
            last_start = chunk.start_char

    def test_overlap_larger_than_size_handled_gracefully(self):
        """If overlap is configured >= chunk_size, it should adjust safely and terminate."""
        text = "Hello world. " * 100
        chunks = ChunkingService.chunk_page(
            text=text,
            filename="test.pdf",
            page_number=1,
            chunk_size=100,
            chunk_overlap=100,  # Invalid config: overlap == size
        )
        assert len(chunks) > 0

    def test_two_page_document_realistic_chunk_distribution(self):
        """
        Simulates a realistic two-page document (e.g. ~5,450 characters total,
        ~2,700 chars on page 1 and ~2,750 chars on page 2).
        Verifies that it yields ~9 chunks total with strict page isolation.
        """
        sample_sentence = "Software engineer experienced in backend architecture, API design, and distributed systems. "
        page1_text = (sample_sentence * 30)[:2700]
        page2_text = (sample_sentence * 30)[:2749]

        pages = [
            DocumentPage(page_number=1, text=page1_text, char_count=len(page1_text)),
            DocumentPage(page_number=2, text=page2_text, char_count=len(page2_text)),
        ]

        chunks = ChunkingService.chunk_document_pages(pages=pages, filename="resume.pdf")

        # 2700 / 650 step is ~4-5 chunks per page; two pages produce 8-10 chunks (typically 9)
        assert len(chunks) == 9
        assert pages[0].chunk_count + pages[1].chunk_count == 9

        # Ensure all page 1 chunks have page_number == 1 and IDs matching resume_pdf_p1_c*
        for chunk in pages[0].chunks:
            assert chunk.page_number == 1
            assert chunk.chunk_id.startswith("resume_pdf_p1_c")
            assert chunk.filename == "resume.pdf"

        # Ensure all page 2 chunks have page_number == 2 and IDs matching resume_pdf_p2_c*
        for chunk in pages[1].chunks:
            assert chunk.page_number == 2
            assert chunk.chunk_id.startswith("resume_pdf_p2_c")
            assert chunk.filename == "resume.pdf"
