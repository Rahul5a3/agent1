from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import fitz  
def clean_extracted_text(text: str) -> str:
    """
    Light cleanup for extracted PDF text.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract plain text from all pages of a PDF using PyMuPDF.
    """
    all_pages: list[str] = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text("text", sort=True)
            page_text = clean_extracted_text(page_text)
            if page_text:
                all_pages.append(page_text)

    return "\n\n".join(all_pages).strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        # Guard: ensure we always move forward
        # Without this, if end - overlap <= start, the loop never advances
        next_start = max(end - overlap, 0)
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def build_evidence_chunks(downloaded_pdfs: dict[str, Path]) -> list[dict[str, Any]]:
    """
    Build evidence chunks from downloaded PDFs.
    Returns a list of dicts with paper title, source path, chunk id, and chunk text.
    """
    evidence_chunks: list[dict[str, Any]] = []

    for paper_title, pdf_path in downloaded_pdfs.items():
        try:
            full_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(full_text)

            for idx, chunk in enumerate(chunks, start=1):
                evidence_chunks.append(
                    {
                        "paper_title": paper_title,
                        "pdf_path": str(pdf_path),
                        "chunk_id": idx,
                        "chunk_text": chunk,
                    }
                )

        except Exception as exc:
            print(f"Text extraction failed for '{paper_title}': {exc}")

    return evidence_chunks
def build_abstract_chunks(papers: list) -> list[dict]:
    """
    Fallback: build evidence chunks from paper abstracts when
    no PDFs were downloaded. Ensures review mode always has
    something to retrieve from even if all PDFs failed.
    """
    chunks = []
    for paper in papers:
        if not paper.abstract:
            continue
        chunks.append({
            "paper_title": paper.title,
            "pdf_path": "abstract_only",
            "chunk_id": 1,
            "chunk_text": paper.abstract,
        })
    return chunks