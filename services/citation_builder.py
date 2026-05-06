from __future__ import annotations

import re
from typing import List

from app.models import PaperRecord, CitationRecord


def generate_citation_key(paper: PaperRecord, existing_keys: set[str] | None = None) -> str:
    """
    Generate a collision-safe citation key in the format:
        AuthorLastName + Year + ShortTitle[a/b/c...]

    The optional `existing_keys` set allows build_citations() to detect
    collisions and append a suffix (a, b, c...) automatically.

    Examples:
        Smith2024Claim       (first paper)
        Smith2024Claima      (second paper that would have collided)
        Smith2024Claimb      (third)
    """
    if existing_keys is None:
        existing_keys = set()

    # Author last name
    author_part = "Unknown"
    if paper.authors:
        first_author = paper.authors[0]
        author_part = first_author.split()[-1]

    # Year
    year_part = str(paper.year) if paper.year else "UnknownYear"

    # Short title — first meaningful word (> 3 chars)
    title_words = re.findall(r"[A-Za-z]+", paper.title)
    short_title = ""
    for word in title_words:
        if len(word) > 3:
            short_title = word.capitalize()
            break
    if not short_title and title_words:
        short_title = title_words[0].capitalize()

    base_key = f"{author_part}{year_part}{short_title}"

    # Collision check — append a, b, c... until unique
    candidate = base_key
    suffix_index = 0
    while candidate in existing_keys:
        candidate = base_key + chr(ord("a") + suffix_index)
        suffix_index += 1

    return candidate


def generate_bibtex(paper: PaperRecord, citation_key: str) -> str:
    """
    Generate a BibTeX entry for a paper.
    Uses @misc for arXiv papers, @article for everything else.
    """
    authors = " and ".join(paper.authors) if paper.authors else "Unknown"
    entry_type = "misc" if paper.arxiv_id else "article"

    bibtex = f"""@{entry_type}{{{citation_key},
  title={{ {paper.title} }},
  author={{ {authors} }},
  year={{ {paper.year if paper.year else ""} }},
  journal={{ {paper.venue if paper.venue else ""} }},
  url={{ {paper.url if paper.url else ""} }},
  doi={{ {paper.doi if paper.doi else ""} }},
}}"""

    return bibtex


def build_citations(papers: List[PaperRecord]) -> List[CitationRecord]:
    """
    Convert PaperRecord objects into CitationRecord objects.

    Tracks all generated keys within this batch so collisions
    are caught and resolved with a/b/c suffixes before they
    reach the .bib file.
    """
    citations: List[CitationRecord] = []
    used_keys: set[str] = set()

    for paper in papers:
        citation_key = generate_citation_key(paper, existing_keys=used_keys)
        used_keys.add(citation_key)

        bibtex = generate_bibtex(paper, citation_key)

        citation = CitationRecord(
            paper_title=paper.title,
            citation_key=citation_key,
            bibtex=bibtex,
            source=paper.source,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            semantic_scholar_id=paper.semantic_scholar_id,
        )

        citations.append(citation)

    return citations