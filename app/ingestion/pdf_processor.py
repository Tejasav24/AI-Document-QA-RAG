from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from PyPDF2 import PdfReader


@dataclass
class PageDocument:
    document_name: str
    page_number: int
    text: str
    source_type: str = "pdf"


def clean_text(text: str) -> str:
    """Normalize extracted PDF text without changing its meaning."""
    if not text:
        return ""

    text = text.replace("\x00", " ")

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def extract_pdf_pages(pdf_path: str | Path) -> List[PageDocument]:
    """
    Extract a PDF page-by-page while preserving document and page metadata.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.suffix}")

    reader = PdfReader(str(pdf_path))

    if not reader.pages:
        raise ValueError(f"PDF contains no pages: {pdf_path.name}")

    documents: List[PageDocument] = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""
        cleaned = clean_text(extracted_text)

        if not cleaned:
            continue

        documents.append(
            PageDocument(
                document_name=pdf_path.name,
                page_number=page_number,
                text=cleaned,
            )
        )

    return documents
