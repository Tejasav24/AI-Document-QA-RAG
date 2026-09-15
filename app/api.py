from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services import RAGService


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


def create_api(rag_service: RAGService) -> FastAPI:
    """Create the FastAPI application for the RAG service."""

    api = FastAPI(
        title="AI Document Q&A API",
        description="Grounded document question-answering API using RAG.",
        version="1.0.0",
    )

    @api.get("/health")
    def health() -> dict:
        return {
            "status": "healthy",
            "service": "ai-document-qa-rag",
        }

    @api.post("/ask")
    def ask(request: AskRequest) -> dict:
        try:
            return rag_service.ask(
                query=request.query,
                top_k=request.top_k,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while processing the request.",
            ) from exc

    return api


__all__ = [
    "AskRequest",
    "create_api",
]
