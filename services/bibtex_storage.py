from __future__ import annotations

from pathlib import Path
import re

from app.config import BIB_DIR
from app.models import CitationRecord


def sanitize_filename(text: str) -> str:
    """
    Convert a query/title into a safe filename.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:80] if text else "citations"


def deduplicate_citations(citations: list[CitationRecord]) -> list[CitationRecord]:
    """
    Remove duplicate citations using citation key priority,
    falling back to DOI or arXiv ID when useful.
    """
    seen = set()
    unique_citations: list[CitationRecord] = []

    for citation in citations:
        dedup_key = (
            citation.doi
            or citation.arxiv_id
            or citation.semantic_scholar_id
            or citation.citation_key
        )

        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        unique_citations.append(citation)

    return unique_citations


def save_bibtex_file(citations: list[CitationRecord], query: str) -> Path:
    """
    Save citations into a .bib file under storage/bib/.
    """
    BIB_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(query)
    output_path = BIB_DIR / f"{safe_name}.bib"

    bib_entries = [citation.bibtex for citation in citations]
    bib_content = "\n\n".join(bib_entries).strip() + "\n"

    output_path.write_text(bib_content, encoding="utf-8")

    return output_path