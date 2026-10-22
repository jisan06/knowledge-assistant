
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

    model_config = SettingsConfigDict(
        env_file=(str(ENV_FILE_PATH), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
