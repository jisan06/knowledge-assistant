import logging
import os
from typing import Optional

from app.config import settings
from app.vector_store.base import BaseVectorStore
from app.vector_store.chroma_store import ChromaVectorStore
from app.vector_store.memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)

# Global singleton instance for vector store
_vector_store_instance: Optional[BaseVectorStore] = None


def get_vector_store(store_type: Optional[str] = None) -> BaseVectorStore:
    """
    Factory function to retrieve or create the active vector store singleton.

    Order of preference:
    1. 'chroma' (Default): Uses ChromaDB with local persistent storage at backend/data/chroma.
       If chromadb is not installed in the environment, gracefully falls back to persistent InMemoryVectorStore.
    2. 'memory': Pure in-memory vector store, ideal for testing and zero-dependency environments.
    """
    global _vector_store_instance
    if _vector_store_instance is not None and store_type is None:
        return _vector_store_instance

    target_type = (store_type or settings.VECTOR_STORE_TYPE).lower().strip()

    if target_type == "chroma":
        try:
            import chromadb  # Check availability
            store = ChromaVectorStore(
                persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )
            store._get_collection()  # Test that persistent client and collection initialize
            logger.info(f"Initialized ChromaVectorStore at: {settings.CHROMA_PERSIST_DIRECTORY}")
            if store_type is None:
                _vector_store_instance = store
            return store
        except (ImportError, RuntimeError, Exception) as e:
            logger.warning(
                f"ChromaDB unavailable or initialization failed ({e}). Falling back to persistent InMemoryVectorStore."
            )
            fallback_path = os.path.join(
                os.path.dirname(settings.CHROMA_PERSIST_DIRECTORY),
                "vector_store.json",
            )
            store = InMemoryVectorStore(
                persist_path=fallback_path,
                collection_name=settings.CHROMA_COLLECTION_NAME,
            )
            if store_type is None:
                _vector_store_instance = store
            return store

    if target_type in ("memory", "in_memory"):
        store = InMemoryVectorStore(collection_name=settings.CHROMA_COLLECTION_NAME)
        if store_type is None:
            _vector_store_instance = store
        return store

    raise ValueError(f"Unsupported VECTOR_STORE_TYPE: '{target_type}'")
