from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """Reference to a retrieved document chunk."""

    document_name: str
    page_number: int = Field(ge=1)
    chunk_id: str
    source_type: str = "pdf"


class RetrievedChunk(BaseModel):
    """Structured representation of a retrieved chunk."""

    source: SourceReference
    text: str
    distance: float = Field(ge=0)


class RetrievalResponse(BaseModel):
    """Structured retrieval result returned by the retrieval layer."""

    query: str
    results: List[RetrievedChunk]
