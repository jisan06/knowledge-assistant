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
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional numerical embedding vector for this chunk",
    )


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
    indexed_chunks: int = Field(default=0, description="Number of chunks successfully indexed into the vector store")
    message: str = Field(..., description="Informative status message about extraction results")
    is_duplicate: bool = Field(default=False, description="True if the uploaded document content already exists in the system")
    is_updated: bool = Field(default=False, description="True if an existing document with the same filename was updated")
    content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of the uploaded document content")


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


# --- Embedding Request & Response Schemas ---

class ChunkEmbeddingResult(BaseModel):
    chunk_id: str = Field(..., description="Unique ID matching the source DocumentChunk")
    filename: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="1-based page number")
    chunk_index: int = Field(..., description="1-based chunk index on the page")
    text: str = Field(..., description="Chunk text content")
    vector: List[float] = Field(..., description="Numerical embedding vector")
    dimension: int = Field(..., description="Length of the embedding vector")
    model: str = Field(..., description="Name of the embedding model used")


class EmbedPreviewRequest(BaseModel):
    text: str = Field(..., description="Sample text or chunk to embed")
    filename: str = Field(default="preview.pdf", description="Optional source filename")
    page_number: int = Field(default=1, description="Optional page number", ge=1)


class EmbedPreviewResponse(BaseModel):
    chunk_id: str = Field(..., description="Stable chunk identifier")
    filename: str = Field(..., description="Source filename")
    page_number: int = Field(..., description="1-based page number")
    model: str = Field(..., description="Name of embedding model")
    dimension: int = Field(..., description="Vector dimensionality")
    vector_preview: List[float] = Field(
        ...,
        description="First few elements of the vector for inspection (e.g. first 5 floats)",
    )
    full_vector_length: int = Field(..., description="Total length of the vector")


class EmbedChunksRequest(BaseModel):
    chunks: List[DocumentChunk] = Field(..., description="List of DocumentChunk items to embed")


class EmbedChunksResponse(BaseModel):
    model: str = Field(..., description="Embedding model name used")
    dimension: int = Field(..., description="Vector dimensionality")
    total_chunks: int = Field(..., description="Total number of chunks processed")
    embeddings: List[ChunkEmbeddingResult] = Field(
        default_factory=list,
        description="List of embedded chunk results preserving metadata and vectors",
    )
