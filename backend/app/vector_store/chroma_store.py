import logging
import os
from typing import List, Optional

from app.config import settings
from app.schemas.document import DocumentChunk
from app.schemas.search import SearchResultItem
from app.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Persistent vector store powered by ChromaDB.
    Persists vectors to a dedicated local directory: backend/data/chroma.
    Keeps vector index completely decoupled from chat SQLite database.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.coll_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._client = None
        self._collection = None

    @property
    def store_type(self) -> str:
        return "chroma"

    @property
    def collection_name(self) -> str:
        return self.coll_name

    def _get_collection(self):
        """Lazy-initializes ChromaDB PersistentClient and collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:
            raise RuntimeError(
                "ChromaDB is not installed in the current environment.\n"
                "Install it using:\n"
                "    pip install chromadb"
            ) from e

        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # Use cosine distance space for normalized semantic embeddings
            self._collection = self._client.get_or_create_collection(
                name=self.coll_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB collection")
            raise RuntimeError(f"ChromaDB initialization error: {e}") from e

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})."
            )

        collection = self._get_collection()

        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []
        vectors: List[List[float]] = []

        for chunk, emb in zip(chunks, embeddings):
            if not chunk.text or not chunk.text.strip():
                continue
            ids.append(chunk.chunk_id)
            docs.append(chunk.text)
            vectors.append(emb)
            metas.append({
                "filename": chunk.filename,
                "page_number": int(chunk.page_number),
                "chunk_index": int(chunk.chunk_index),
                "chunk_id": chunk.chunk_id,
            })

        if not ids:
            return 0

        # Upsert ensures re-indexing the same document updates without duplicating entries
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=docs,
            metadatas=metas,
        )

        return len(ids)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        filter_filename: Optional[str] = None,
    ) -> List[SearchResultItem]:
        collection = self._get_collection()
        total_items = collection.count()

        if total_items == 0:
            return []

        n_results = max(1, min(top_k, total_items))
        where_filter = {"filename": filter_filename} if filter_filename else None

        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        query_output = collection.query(**query_kwargs)

        items: List[SearchResultItem] = []
        ids_raw = query_output.get("ids") or []
        ids_list = ids_raw[0] if ids_raw else []

        docs_raw = query_output.get("documents") or []
        docs_list = docs_raw[0] if docs_raw else []

        metas_raw = query_output.get("metadatas") or []
        metas_list = metas_raw[0] if metas_raw else []

        dists_raw = query_output.get("distances") or []
        dists_list = dists_raw[0] if dists_raw else []

        for chunk_id, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            meta_dict = meta or {}
            # In Chroma with cosine space: distance = 1 - cosine_similarity
            raw_dist = float(dist) if dist is not None else 0.0
            similarity = max(0.0, round(1.0 - raw_dist, 4))

            items.append(
                SearchResultItem(
                    chunk_id=chunk_id,
                    text=doc or "",
                    filename=str(meta_dict.get("filename", "")),
                    page_number=int(meta_dict.get("page_number", 1)),
                    chunk_index=int(meta_dict.get("chunk_index", 1)),
                    similarity_score=similarity,
                    distance=round(raw_dist, 4),
                )
            )

        return items

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()

    def delete_by_filename(self, filename: str) -> int:
        collection = self._get_collection()
        before = collection.count()
        collection.delete(where={"filename": filename})
        after = collection.count()
        return max(0, before - after)

    def clear(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(self.coll_name)
            except Exception:
                pass
            self._collection = None
