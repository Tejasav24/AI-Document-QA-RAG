# AI Document Q&A — Retrieval-Augmented Generation

An end-to-end Retrieval-Augmented Generation (RAG) system for answering questions from PDF documents using semantic retrieval and grounded text generation.

The system extracts PDF content page-by-page, preserves source metadata, creates overlapping chunks, generates semantic embeddings, retrieves relevant context with FAISS, filters low-relevance results, and generates answers using only retrieved document context.

## Features

- PDF text extraction with page-level metadata
- Text cleaning and normalization
- Overlapping document chunking
- Document, page, and chunk metadata preservation
- Sentence Transformer embeddings
- FAISS vector similarity search
- Configurable relevance-distance threshold
- Duplicate chunk filtering
- Grounded answer generation
- Explicit fallback when relevant information is not retrieved
- Source and page references for generated answers
- FastAPI `/health` and `/ask` endpoints
- Pydantic request validation
- Gradio user interface
- Automated smoke and API tests

## Architecture

```text
PDF
 │
 ▼
PDF Extraction
 │
 ▼
Text Cleaning
 │
 ▼
Overlapping Chunking
 │
 ▼
Sentence Transformer Embeddings
 │
 ▼
FAISS Vector Index
 │
 │
 ▼
User Query
 │
 ▼
Query Embedding
 │
 ▼
Similarity Retrieval
 │
 ▼
Relevance Filtering
 │
 ▼
Retrieved Context
 │
 ▼
Grounded Generator
 │
 ▼
Answer + Source/Page References
```

## RAG Pipeline

The pipeline follows these main stages:

1. **Extraction** — PDF content is extracted page-by-page using PyPDF2.
2. **Cleaning** — Empty lines and null characters are normalized without intentionally changing document meaning.
3. **Chunking** — Page text is divided into overlapping character-based chunks.
4. **Metadata** — Each chunk retains the document name, page number, chunk ID, and source type.
5. **Embedding** — `all-MiniLM-L6-v2` converts chunks into dense semantic vectors.
6. **Indexing** — FAISS `IndexFlatL2` stores vectors for similarity search.
7. **Retrieval** — The query is embedded and compared against indexed chunks.
8. **Filtering** — Results above the configured distance threshold are removed.
9. **Deduplication** — Duplicate chunk text is removed from retrieved context.
10. **Generation** — A lightweight text-generation model produces an answer using retrieved context only.
11. **Grounding fallback** — If no relevant chunks are retrieved, the system returns an explicit fallback instead of generating an answer from outside knowledge.
12. **Sources** — Retrieved document and page references are returned with the answer.

## Relevance Filtering and Fallback

The retrieval layer uses FAISS L2 distance together with a configurable relevance threshold.

For the current embedding model and document used during development, relevant questions produced distances below the configured threshold while unrelated questions produced substantially higher distances.

This allows the system to reject low-relevance retrieval results instead of blindly passing unrelated context to the generator.

If no relevant chunks remain after filtering, the generation layer returns an explicit fallback response:

```text
I couldn't find the answer in the provided document.
```

## Project Structure

```text
AI-Document-QA-RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── services.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py
│   │   └── chunker.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py
│   │
│   └── models/
│       ├── __init__.py
│       └── schemas.py
│
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

## Tech Stack

- **Python**
- **PyPDF2** — PDF text extraction
- **Sentence Transformers** — semantic embeddings
- **FAISS** — vector similarity search
- **Transformers** — grounded text generation
- **PyTorch** — model execution
- **FastAPI** — REST API
- **Pydantic** — request/schema validation
- **Gradio** — interactive UI
- **Pytest** — automated testing

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

The core RAG components can then be imported and composed through the application service layer.

The project demonstrates the extraction, chunking, embedding, retrieval, filtering, generation, API, and UI components through the included implementation and tests.

## FastAPI

The API exposes:

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "ai-document-qa-rag"
}
```

### Ask a Question

```http
POST /ask
```

Example request:

```json
{
  "query": "What is machine learning?",
  "top_k": 3
}
```

The response contains the original query, generated answer, retrieved chunk count, source document, page number, chunk ID, and retrieval distance.

## Gradio Interface

The project includes a Gradio interface for asking questions and displaying:

- generated answer
- document source
- page references

Answers are produced from retrieved document context.

## Testing

The project includes automated tests covering:

- chunking and metadata preservation
- FastAPI health endpoint
- question-answer endpoint
- grounded fallback behavior
- empty-query validation
- invalid `top_k` validation

Run:

```bash
pytest -q
```

The final verified regression suite contains **6 tests**.

## Design Decisions

### Page-Level Metadata

PDF content is extracted page-by-page so retrieved context can be traced back to its original page.

### Character-Based Chunking

The current chunker uses a 500-character chunk size with 100-character overlap. This is intentionally simple and reproducible for the portfolio implementation.

### FAISS IndexFlatL2

FAISS `IndexFlatL2` provides straightforward exact L2 similarity search and is appropriate for the current portfolio-scale dataset.

### Relevance Threshold

Retrieval is not treated as successful merely because FAISS returns nearest neighbors. A distance threshold is applied so weak matches can be rejected.

### Grounded Generation

The generator is explicitly instructed to use only retrieved document context. When no relevant context is available, the system returns a fallback response.

## Limitations

- PDF extraction depends on the document containing a usable text layer.
- Scanned/image-only PDFs are not processed with OCR.
- The current implementation uses a lightweight generation model intended for a portfolio/demo system.
- Retrieval quality depends on the embedding model, chunking strategy, and relevance threshold.
- The current implementation is designed for portfolio-scale document processing rather than large production workloads.
- Vector indexes are currently built in memory rather than persisted in a production vector database.
- The current UI focuses on querying an indexed document pipeline rather than providing a complete document-management workflow.

## Future Improvements

Potential future improvements include:

- OCR support for scanned PDFs
- Better chunking strategies
- Hybrid keyword + semantic retrieval
- Reranking models
- Persistent vector storage
- Evaluation datasets and retrieval/generation metrics
- Conversation history
- Multi-document management
- Production deployment and observability

These are intentionally outside the current portfolio scope.

## License

This project is intended as a portfolio and learning project.
