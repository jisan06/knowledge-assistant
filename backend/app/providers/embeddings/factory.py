from typing import Optional

from app.config import settings
from app.providers.embeddings.base import BaseEmbeddingProvider
from app.providers.embeddings.local_provider import LocalEmbeddingProvider
from app.providers.embeddings.mock_provider import MockEmbeddingProvider


def get_embedding_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> BaseEmbeddingProvider:
    """
    Factory function to instantiate the configured embedding provider.

    Supported providers:
    - 'local' (or 'fastembed'): Local lightweight ONNX / sentence-transformers
    - 'mock': Deterministic mock provider for offline tests and verification
    """
    provider = (provider_name or settings.EMBEDDING_PROVIDER).lower().strip()
    model = model_name or settings.EMBEDDING_MODEL
    dim = dimension or settings.EMBEDDING_DIMENSION

    if provider in ("local", "fastembed", "sentence-transformers"):
        return LocalEmbeddingProvider(model_name=model, dimension=dim)

    if provider == "mock":
        return MockEmbeddingProvider(model_name=model, dimension=dim)

    raise ValueError(
        f"Unsupported embedding provider: '{provider}'. "
        f"Available options are 'local' and 'mock'."
    )
