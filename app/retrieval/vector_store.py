from __future__ import annotations

from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import DocumentChunk


class FAISSVectorStore:
    """
    FAISS-backed vector store for document chunks.

    FAISS stores vectors while the original DocumentChunk objects retain
    document/page/chunk metadata for source citations.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 1.0,
    ) -> None:
        if relevance_threshold <= 0:
            raise ValueError("relevance_threshold must be greater than 0.")

        self.model_name = model_name
        self.relevance_threshold = relevance_threshold

        self.embedding_model = SentenceTransformer(model_name)

        self.index = None
        self.chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk]) -> None:
        """Build a FAISS index from document chunks."""

        if not chunks:
            raise ValueError("Cannot build vector store from empty chunks.")

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        self.chunks = list(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        relevance_threshold: float | None = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve relevant chunks using FAISS distance filtering.

        Results above the configured distance threshold are discarded.
        Duplicate chunk text is also removed.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if self.index is None or not self.chunks:
            raise ValueError("Vector store has not been built.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        threshold = (
            self.relevance_threshold
            if relevance_threshold is None
            else relevance_threshold
        )

        if threshold <= 0:
            raise ValueError("relevance_threshold must be greater than 0.")

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        candidate_k = min(max(top_k * 3, top_k), len(self.chunks))

        distances, indices = self.index.search(
            query_embedding,
            candidate_k,
        )

        results: List[Tuple[DocumentChunk, float]] = []
        seen_texts = set()

        for distance, index in zip(distances[0], indices[0]):
            if index < 0:
                continue

            distance = float(distance)

            if distance > threshold:
                continue

            chunk = self.chunks[int(index)]

            normalized_text = " ".join(chunk.text.lower().split())

            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)
            results.append((chunk, distance))

            if len(results) >= top_k:
                break

        return results
