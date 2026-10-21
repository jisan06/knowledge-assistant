from typing import List
from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    page_number: int = Field(..., description="1-based page number")
    text: str = Field(..., description="Extracted text from this page")
    char_count: int = Field(..., description="Number of characters extracted from this page")


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
        description="List of extracted pages with 1-based page numbers and text",
    )
    message: str = Field(..., description="Informative status message about extraction results")
