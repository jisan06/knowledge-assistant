from typing import List, Optional
from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    chunk_id: str = Field(..., description="Stable chunk identifier")
    text: str = Field(..., description="Extracted text content of the retrieved chunk")
    filename: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="1-based page number")
    chunk_index: int = Field(..., description="1-based chunk index on that page")
    similarity_score: float = Field(..., description="Cosine similarity score (higher = more relevant)")
    distance: Optional[float] = Field(default=None, description="Raw distance metric from vector store")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string", min_length=1)
    top_k: int = Field(default=4, ge=1, le=20, description="Number of top relevant chunks to retrieve (3-5 typical)")
    filename: Optional[str] = Field(default=None, description="Optional filter to restrict search to a specific document")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original user search query")
    total_results: int = Field(..., description="Number of matching passages retrieved")
    results: List[SearchResultItem] = Field(default_factory=list, description="Top relevant chunk passages")
    model: str = Field(..., description="Embedding model used for query and chunk vectorization")


class IndexDocumentResponse(BaseModel):
    id: Optional[str] = Field(default=None, description="Unique database identifier for this document")
    filename: str = Field(..., description="Indexed document filename")
    total_pages: int = Field(..., description="Total pages in the document")
    total_chunks: int = Field(..., description="Total chunks extracted")
    indexed_chunks: int = Field(..., description="Number of chunk vectors stored in vector database")
    message: str = Field(..., description="Status summary of the indexing operation")
    is_duplicate: bool = Field(default=False, description="True if the uploaded document content already exists in the system")
    is_updated: bool = Field(default=False, description="True if an existing document with the same filename was updated")
    content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of the uploaded document content")


class CollectionStatsResponse(BaseModel):
    collection_name: str = Field(..., description="Vector collection name")
    total_chunks: int = Field(..., description="Total count of chunk vectors currently indexed")
    store_type: str = Field(..., description="Backend vector store implementation (e.g. chroma, memory)")


class AskQuestionRequest(BaseModel):
    question: str = Field(..., description="User question to answer using RAG", min_length=1)
    top_k: int = Field(default=4, ge=1, le=20, description="Number of supporting document chunks to retrieve")


class AskQuestionResponse(BaseModel):
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded answer synthesized from retrieved document chunks")
    sources: List[SearchResultItem] = Field(default_factory=list, description="Retrieved source chunks used as context")

