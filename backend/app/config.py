
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    AI_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB default
    CHUNK_SIZE: int = 800  # Default target chunk size in characters
    CHUNK_OVERLAP: int = 150  # Default overlap between consecutive page chunks
    EMBEDDING_PROVIDER: str = "local"  # "local", "mock", etc.
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"  # Compact 384-dim semantic embedding model
    EMBEDDING_DIMENSION: int = 384  # Vector length matching BAAI/bge-small-en-v1.5 & all-MiniLM-L6-v2
    VECTOR_STORE_TYPE: str = "chroma"  # "chroma" or "memory"
    CHROMA_PERSIST_DIRECTORY: str = str(Path(__file__).resolve().parent.parent / "data" / "chroma")
    CHROMA_COLLECTION_NAME: str = "document_chunks"
    SEARCH_TOP_K: int = 4  # Default top 3-5 results for semantic search

    model_config = SettingsConfigDict(
        env_file=(str(ENV_FILE_PATH), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
