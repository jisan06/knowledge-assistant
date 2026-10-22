import asyncio
import logging
from typing import List, Optional

from app.providers.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Lightweight local embedding provider for semantic search.
    Default model: 'BAAI/bge-small-en-v1.5' (384 dimensions, MTEB top-performing lightweight model).
    Alternative: 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions).

    Design features:
    - Zero API cost and runs 100% locally on CPU with minimal RAM (~200MB).
    - Lazy loading: the model is not loaded into memory until first use.
    - Gracefully inspects and reports missing dependencies (fastembed or sentence-transformers).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
    ):
        self._model_name = model_name
        self._dimension = dimension
        self._model = None
        self._backend_type = None  # "fastembed" or "sentence_transformers"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self):
        """Loads the local embedding model lazily on first access."""
        if self._model is not None:
            return self._model

        fastembed_import_err: Optional[Exception] = None
        # Attempt 1: FastEmbed (Recommended lightweight ONNX runtime)
        try:
            from fastembed import TextEmbedding
            logger.info(f"Loading FastEmbed model '{self._model_name}'...")
            self._model = TextEmbedding(model_name=self._model_name)
            self._backend_type = "fastembed"
            return self._model
        except ImportError as e:
            fastembed_import_err = e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FastEmbed model '{self._model_name}': {e}"
            ) from e

        sentence_tx_import_err: Optional[Exception] = None
        # Attempt 2: Sentence-Transformers fallback
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self._model_name}'...")
            self._model = SentenceTransformer(self._model_name)
            self._backend_type = "sentence_transformers"
            return self._model
        except ImportError as e:
            sentence_tx_import_err = e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{self._model_name}': {e}"
            ) from e

        # Neither library is available
        diagnostic_lines = [
            "No local embedding library is installed in the current environment.",
            "The project is configured to use 'fastembed' (or 'sentence-transformers').",
            "To install the required dependencies in your active virtual environment, run:",
            "    pip install -r requirements.txt",
            "or:",
            "    pip install fastembed",
        ]
        if fastembed_import_err:
            diagnostic_lines.append(f"\n[Diagnostic] fastembed import error: {fastembed_import_err}")
        if sentence_tx_import_err:
            diagnostic_lines.append(f"[Diagnostic] sentence-transformers import error: {sentence_tx_import_err}")

        raise RuntimeError("\n".join(diagnostic_lines))

    async def embed_text(self, text: str) -> List[float]:
        """Embeds a single text string."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")
        results = await self.embed_batch([text.strip()])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts asynchronously on a threadpool to prevent blocking the event loop."""
        if not texts:
            return []

        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            return []

        def _sync_embed() -> List[List[float]]:
            model = self._load_model()
            if self._backend_type == "fastembed":
                # fastembed returns generator of numpy arrays
                embeddings_gen = model.embed(cleaned)
                return [arr.tolist() for arr in embeddings_gen]
            elif self._backend_type == "sentence_transformers":
                # sentence_transformers returns numpy ndarray
                embeddings_arr = model.encode(cleaned, convert_to_numpy=True)
                return [arr.tolist() for arr in embeddings_arr]
            else:
                raise RuntimeError(f"Unknown embedding backend: {self._backend_type}")

        return await asyncio.to_thread(_sync_embed)

    async def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query text using query-specific instruction/prefix when supported by the backend.
        For BGE models in FastEmbed, calls query_embed().
        For sentence-transformers, prepends 'Represent this sentence for searching relevant passages: '.
        """
        if not query or not query.strip():
            raise ValueError("Cannot generate embedding for empty query.")

        cleaned = query.strip()

        def _sync_embed_query() -> List[float]:
            model = self._load_model()
            if self._backend_type == "fastembed":
                if hasattr(model, "query_embed"):
                    embeddings_gen = model.query_embed([cleaned])
                    return list(embeddings_gen)[0].tolist()
                embeddings_gen = model.embed([cleaned])
                return list(embeddings_gen)[0].tolist()
            elif self._backend_type == "sentence_transformers":
                prefix = "Represent this sentence for searching relevant passages: "
                prompt = (prefix + cleaned) if "bge" in self._model_name.lower() else cleaned
                embeddings_arr = model.encode(prompt, convert_to_numpy=True)
                return embeddings_arr.tolist()
            else:
                raise RuntimeError(f"Unknown embedding backend: {self._backend_type}")

        return await asyncio.to_thread(_sync_embed_query)

