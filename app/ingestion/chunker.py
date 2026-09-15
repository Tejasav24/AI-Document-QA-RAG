from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.ingestion.pdf_processor import PageDocument


@dataclass
class DocumentChunk:
    document_name: str
    page_number: int
    chunk_id: str
    text: str
    source_type: str = "pdf"


def chunk_pages(
    pages: List[PageDocument],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[DocumentChunk]:
    """
    Split page-level documents into overlapping chunks while preserving
    document and page metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[DocumentChunk] = []

    for page in pages:
        text = page.text.strip()

        if not text:
            continue

        step = chunk_size - overlap
        chunk_number = 0

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if not chunk_text:
                continue

            chunk_id = f"{page.page_number}_{chunk_number}"

            chunks.append(
                DocumentChunk(
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_type=page.source_type,
                )
            )

            chunk_number += 1

    return chunks
