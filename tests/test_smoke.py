from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import create_api
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_processor import PageDocument
from app.services import RAGService


class FakeVectorStore:
    def search(self, query: str, top_k: int = 3):
        if "machine learning" in query.lower():
            chunk = SimpleNamespace(
                document_name="test.pdf",
                page_number=1,
                chunk_id="1_0",
                text="Machine learning enables computers to learn from data.",
                source_type="pdf",
            )
            return [(chunk, 0.25)]

        return []


class FakeGenerator:
    def generate_with_sources(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find the answer in the provided document.",
                "sources": [],
            }

        chunk, distance = retrieved_chunks[0]

        return {
            "answer": "Machine learning enables computers to learn from data.",
            "sources": [
                {
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "distance": round(distance, 4),
                }
            ],
        }


@pytest.fixture
def client():
    service = RAGService(
        vector_store=FakeVectorStore(),
        generator=FakeGenerator(),
    )

    api = create_api(service)

    return TestClient(api)


def test_chunking_preserves_metadata():
    pages = [
        PageDocument(
            document_name="test.pdf",
            page_number=2,
            text="A" * 700,
        )
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=100,
    )

    assert len(chunks) == 2
    assert chunks[0].document_name == "test.pdf"
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_id == "2_0"
    assert chunks[1].chunk_id == "2_1"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_endpoint(client):
    response = client.post(
        "/ask",
        json={
            "query": "What is machine learning?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is machine learning?"
    assert data["answer"]
    assert data["retrieved_chunks"] == 1
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page_number"] == 1


def test_ask_endpoint_fallback(client):
    response = client.post(
        "/ask",
        json={
            "query": "What is the capital of France?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["answer"]
        == "I couldn't find the answer in the provided document."
    )
    assert data["sources"] == []
    assert data["retrieved_chunks"] == 0


def test_empty_query_rejected(client):
    response = client.post(
        "/ask",
        json={
            "query": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_invalid_top_k_rejected(client):
    response = client.post(
        "/ask",
        json={
            "query": "What is machine learning?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
