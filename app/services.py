from __future__ import annotations

from typing import Any, Dict

from app.generation.generator import GroundedGenerator
from app.retrieval.vector_store import FAISSVectorStore


class RAGService:
    """Application service combining retrieval and generation."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        generator: GroundedGenerator,
    ) -> None:
        self.vector_store = vector_store
        self.generator = generator

    def ask(
        self,
        query: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Retrieve relevant context and generate a grounded answer."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        retrieved_chunks = self.vector_store.search(
            query=query,
            top_k=top_k,
        )

        result = self.generator.generate_with_sources(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": len(retrieved_chunks),
        }
