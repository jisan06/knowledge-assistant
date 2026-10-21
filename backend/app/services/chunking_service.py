import re
from typing import List, Optional

from app.config import settings
from app.schemas.document import DocumentChunk, DocumentPage


class ChunkingService:
    """
    Dedicated, reusable service for page-isolated text chunking.

    Guarantees:
    - Target chunk size: ~800 characters.
    - Overlap: ~150 characters between consecutive chunks from the same page.
    - Page boundary isolation: text from different pages is NEVER merged.
    - Metadata preservation: preserves source filename and original 1-based page number.
    - Stable chunk IDs: '{clean_filename}_p{page_number}_c{chunk_index}'.
    - Robustness: safely handles empty pages, whitespace-only, short text, and long text.
    - Infinite loop prevention: guaranteed strictly monotonic forward progress on every step.
    """

    @staticmethod
    def clean_filename_for_id(filename: Optional[str]) -> str:
        """Sanitizes filename into a clean, predictable prefix for stable chunk IDs."""
        if not filename:
            return "document"
        clean = re.sub(r"[^\w-]", "_", filename.strip())
        return clean.strip("_") or "document"

    @classmethod
    def chunk_page(
        cls,
        text: Optional[str],
        filename: str,
        page_number: int,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """
        Splits a single page's text into overlapping chunks.
        Never combines text across pages.
        Returns an empty list for empty or whitespace-only pages.
        """
        if not text:
            return []

        raw_text = text.strip()
        if not raw_text:
            return []

        size = chunk_size if chunk_size is not None and chunk_size > 0 else settings.CHUNK_SIZE
        overlap = chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else settings.CHUNK_OVERLAP

        # Overlap must be strictly smaller than chunk size to ensure forward progress
        if overlap >= size:
            overlap = max(0, size // 4)

        doc_prefix = cls.clean_filename_for_id(filename)
        total_len = len(raw_text)

        # Fast path: Short text fits entirely in a single chunk
        if total_len <= size:
            return [
                DocumentChunk(
                    chunk_id=f"{doc_prefix}_p{page_number}_c1",
                    filename=filename,
                    page_number=page_number,
                    chunk_index=1,
                    text=raw_text,
                    char_count=total_len,
                    start_char=0,
                    end_char=total_len,
                )
            ]

        chunks: List[DocumentChunk] = []
        start = 0
        chunk_index = 1

        while start < total_len:
            target_end = start + size

            if target_end >= total_len:
                end = total_len
            else:
                # Search for a natural boundary near target_end
                # Look backwards in a search window to avoid cutting sentences or words in half
                search_window = min(160, max(50, size // 4))
                window_start = max(start + 1, target_end - search_window)
                sub = raw_text[window_start:target_end]

                split_point = -1

                # Priority 1: Paragraph break (\n\n)
                p_idx = sub.rfind("\n\n")
                if p_idx != -1:
                    split_point = window_start + p_idx + 2
                else:
                    # Priority 2: Line break (\n)
                    n_idx = sub.rfind("\n")
                    if n_idx != -1:
                        split_point = window_start + n_idx + 1
                    else:
                        # Priority 3: Sentence ending punctuation followed by whitespace (. ! ?)
                        sent_matches = list(re.finditer(r"[.!?]\s+", sub))
                        if sent_matches:
                            split_point = window_start + sent_matches[-1].end()
                        else:
                            # Priority 4: Word boundary (space)
                            s_idx = sub.rfind(" ")
                            if s_idx != -1:
                                split_point = window_start + s_idx + 1

                if split_point > start:
                    end = split_point
                else:
                    # Fallback: hard cut at target_end
                    end = target_end

            chunk_content = raw_text[start:end].strip()
            if chunk_content:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_prefix}_p{page_number}_c{chunk_index}",
                        filename=filename,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=chunk_content,
                        char_count=len(chunk_content),
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

            if end >= total_len:
                break

            # Calculate next chunk start position based on overlap
            provisional_next = end - overlap

            # Try to snap provisional_next forward to the next word boundary within a small window
            if provisional_next < total_len:
                snap_window = raw_text[provisional_next : min(provisional_next + 30, total_len)]
                space_rel = snap_window.find(" ")
                if space_rel != -1 and space_rel < 25:
                    provisional_next = provisional_next + space_rel + 1

            # Strict forward progress guarantee: next_start MUST be greater than current start
            min_next = start + max(1, size - overlap)
            next_start = max(provisional_next, min_next)

            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks

    @classmethod
    def chunk_document_pages(
        cls,
        pages: List[DocumentPage],
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """
        Chunks multiple document pages, processing each page independently.
        Never combines text across page boundaries.
        Attaches chunks and chunk_count to each DocumentPage and returns the flattened list.
        """
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = cls.chunk_page(
                text=page.text,
                filename=filename,
                page_number=page.page_number,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            page.chunks = page_chunks
            page.chunk_count = len(page_chunks)
            all_chunks.extend(page_chunks)

        return all_chunks
