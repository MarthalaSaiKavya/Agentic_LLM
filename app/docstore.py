"""In-memory document store with simple cosine similarity retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple
import numpy as np

from .embedding import TransformerEmbedder, embed_text

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentChunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict
    embedding: np.ndarray


@dataclass
class MemoryVectorStore:
    """Simple list-based vector store suitable for interactive RAG demos."""

    embedder: TransformerEmbedder
    chunk_size: int = 400
    chunk_overlap: int = 100
    _chunks: List[DocumentChunk] = field(default_factory=list)

    def add_document(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        metadata = metadata or {}
        logger.info("Adding document %s (chars=%d)", doc_id, len(text))
        before = len(self._chunks)
        for idx, chunk_text in enumerate(self._chunk_text(text)):
            embedding = embed_text(chunk_text, self.embedder)
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_{idx}",
                text=chunk_text,
                metadata=metadata,
                embedding=embedding,
            )
            self._chunks.append(chunk)
            logger.debug("Stored chunk %s with %d tokens", chunk.chunk_id, len(chunk_text.split()))
        logger.info("Document %s added with %d chunk(s).", doc_id, len(self._chunks) - before)

    def clear(self) -> None:
        logger.warning("Clearing %d stored chunk(s).", len(self._chunks))
        self._chunks.clear()

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if not self._chunks:
            logger.info("Similarity search skipped; vector store empty.")
            return []
        query_emb = embed_text(query, self.embedder)
        embeddings = np.stack([chunk.embedding for chunk in self._chunks])
        scores = embeddings @ query_emb
        top_indices = np.argsort(-scores)[:k]
        results = [(self._chunks[i], float(scores[i])) for i in top_indices]
        logger.info(
            "Similarity search for '%s' returned %d result(s) from %d stored chunks.",
            query[:80],
            len(results),
            len(self._chunks),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str) -> Iterable[str]:
        text = text.replace("\r\n", "\n")
        tokens = text.split()
        chunk_size = max(self.chunk_size, 50)
        overlap = max(min(self.chunk_overlap, chunk_size // 2), 0)
        logger.debug(
            "Chunking text into windows of %d tokens with %d overlap (total tokens=%d).",
            chunk_size,
            overlap,
            len(tokens),
        )

        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + chunk_size)
            chunk = " ".join(tokens[start:end])
            yield chunk.strip()
            if end == len(tokens):
                break
            start = end - overlap
