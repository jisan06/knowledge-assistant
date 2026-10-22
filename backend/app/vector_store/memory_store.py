import json
import math
import os
from typing import Dict, List, Optional

from app.schemas.document import DocumentChunk
from app.schemas.search import SearchResultItem
from app.vector_store.base import BaseVectorStore


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store with optional JSON file persistence.
    Used for offline testing, local lightweight development, and environments
    where ChromaDB binary dependencies are not installed.
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: str = "document_chunks",
    ):
        self.persist_path = persist_path
        self.coll_name = collection_name
        # Dictionary keyed by chunk_id to prevent duplicates on re-indexing
        self._records: Dict[str, dict] = {}
        if self.persist_path and os.path.exists(self.persist_path):
            self._load_from_disk()

    @property
    def store_type(self) -> str:
        return "memory"

    @property
    def collection_name(self) -> str:
        return self.coll_name

    def _save_to_disk(self):
        if self.persist_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self._records, f)

    def _load_from_disk(self):
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
            except Exception:
                self._records = {}

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

        upserted = 0
        for chunk, emb in zip(chunks, embeddings):
            if not chunk.text or not chunk.text.strip():
                continue

            self._records[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "filename": chunk.filename,
                "page_number": int(chunk.page_number),
                "chunk_index": int(chunk.chunk_index),
                "embedding": emb,
            }
            upserted += 1

        if upserted > 0:
            self._save_to_disk()

        return upserted

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        filter_filename: Optional[str] = None,
    ) -> List[SearchResultItem]:
        if not self._records or not query_vector:
            return []

        scored_items = []
        for record in self._records.values():
            if filter_filename and record.get("filename") != filter_filename:
                continue

            emb = record.get("embedding", [])
            sim = cosine_similarity(query_vector, emb)
            distance = max(0.0, 1.0 - sim)

            scored_items.append((
                sim,
                distance,
                record,
            ))

        # Sort descending by similarity score
        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_items = scored_items[:top_k]

        results = []
        for sim, dist, rec in top_items:
            results.append(
                SearchResultItem(
                    chunk_id=rec["chunk_id"],
                    text=rec["text"],
                    filename=rec["filename"],
                    page_number=rec["page_number"],
                    chunk_index=rec["chunk_index"],
                    similarity_score=round(sim, 4),
                    distance=round(dist, 4),
                )
            )

        return results

    def count(self) -> int:
        return len(self._records)

    def delete_by_filename(self, filename: str) -> int:
        to_delete = [
            cid for cid, rec in self._records.items()
            if rec.get("filename") == filename
        ]
        for cid in to_delete:
            del self._records[cid]

        if to_delete:
            self._save_to_disk()

        return len(to_delete)

    def clear(self) -> None:
        self._records.clear()
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                os.remove(self.persist_path)
            except Exception:
                pass
