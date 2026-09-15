from __future__ import annotations

from typing import List, Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.ingestion.chunker import DocumentChunk


class GroundedGenerator:
    """Generate grounded answers from retrieved document chunks."""

    FALLBACK_MESSAGE = (
        "I couldn't find the answer in the provided document."
    )

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
    ) -> None:
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> str:
        """Build a grounded prompt from retrieved chunks."""

        if not retrieved_chunks:
            return (
                "Question: "
                f"{query}\n\n"
                "Answer: "
                f"{self.FALLBACK_MESSAGE}"
            )

        context_parts = []

        for rank, (chunk, _) in enumerate(
            retrieved_chunks,
            start=1,
        ):
            context_parts.append(
                f"[Source {rank} | "
                f"{chunk.document_name} | "
                f"Page {chunk.page_number}]\n"
                f"{chunk.text}"
            )

        context = "\n\n".join(context_parts)

        return (
            "You are a document question-answering assistant. "
            "Answer using only the provided document context. "
            "Do not use outside knowledge. "
            "If the answer is not present in the context, "
            "say exactly that the information is not available "
            "in the provided document.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        max_new_tokens: int = 150,
    ) -> str:
        """Generate an answer using retrieved context only."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not retrieved_chunks:
            return self.FALLBACK_MESSAGE

        prompt = self.build_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        if not answer:
            return self.FALLBACK_MESSAGE

        return answer

    def build_sources(
        self,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
    ) -> List[dict]:
        """Build source references for the generated answer."""

        sources = []
        seen = set()

        for chunk, distance in retrieved_chunks:
            key = (
                chunk.document_name,
                chunk.page_number,
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "distance": round(distance, 4),
                }
            )

        return sources

    def generate_with_sources(
        self,
        query: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        max_new_tokens: int = 150,
    ) -> dict:
        """Return a grounded answer together with source references."""

        answer = self.generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            max_new_tokens=max_new_tokens,
        )

        sources = self.build_sources(retrieved_chunks)

        return {
            "answer": answer,
            "sources": sources,
        }
