from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Stable unique chunk identifier (e.g., sample_pdf_p1_c1)")
    filename: str = Field(..., description="Original filename of the document")
    page_number: int = Field(..., description="1-based source page number")
    chunk_index: int = Field(..., description="1-based index of this chunk on the page")
    text: str = Field(..., description="Extracted text chunk content")
    char_count: int = Field(..., description="Character count of this chunk")
    start_char: int = Field(default=0, description="Start character offset on the source page")
    end_char: int = Field(default=0, description="End character offset on the source page")


class DocumentPage(BaseModel):
    page_number: int = Field(..., description="1-based page number")
    text: str = Field(..., description="Extracted text from this page")
    char_count: int = Field(..., description="Number of characters extracted from this page")
    chunk_count: int = Field(default=0, description="Number of chunks generated from this page")
    chunks: List[DocumentChunk] = Field(
        default_factory=list,
        description="List of text chunks generated for this page",
    )


class DocumentUploadResponse(BaseModel):
    filename: str = Field(..., description="Original uploaded filename")
    total_pages: int = Field(..., description="Total number of pages parsed")
    total_characters: int = Field(..., description="Total characters extracted across all pages")
    has_extractable_text: bool = Field(
        ...,
        description="True if any text was extracted, False if document has no extractable text",
    )
    pages: List[DocumentPage] = Field(
        default_factory=list,
        description="List of extracted pages with 1-based page numbers, text, and chunks",
    )
    total_chunks: int = Field(default=0, description="Total number of text chunks generated across all pages")
    chunks: List[DocumentChunk] = Field(
        default_factory=list,
        description="Flattened list of all chunks across all pages",
    )
    message: str = Field(..., description="Informative status message about extraction results")


class ChunkPreviewRequest(BaseModel):
    text: str = Field(..., description="Raw text to split into chunks")
    filename: str = Field(default="preview.txt", description="Source filename for metadata")
    page_number: int = Field(default=1, description="Source page number for metadata", ge=1)
    chunk_size: Optional[int] = Field(default=None, description="Optional target chunk size (default 800)")
    chunk_overlap: Optional[int] = Field(default=None, description="Optional overlap size (default 150)")


class ChunkPreviewResponse(BaseModel):
    filename: str = Field(..., description="Filename used for chunk metadata")
    page_number: int = Field(..., description="Page number used for chunk metadata")
    total_characters: int = Field(..., description="Total characters in input text")
    total_chunks: int = Field(..., description="Total number of chunks generated")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of generated chunks")
