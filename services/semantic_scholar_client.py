from __future__ import annotations

import os
import time
from typing import Any

import requests

from app.config import MAX_SEARCH_RESULTS_PER_SOURCE
from app.models import PaperRecord

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "authors",
        "year",
        "abstract",
        "externalIds",
        "venue",
        "url",
        "openAccessPdf",
    ]
)

# -------------------------------------------------------------------
# Set your free API key here OR as an environment variable:
#   export SEMANTIC_SCHOLAR_API_KEY="your_key_here"
# Get one free at: https://www.semanticscholar.org/product/api
# Without a key: 1 req/sec limit (very easy to hit 429)
# With a key:   10 req/sec limit
# -------------------------------------------------------------------
_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
def build_semantic_scholar_query(keywords: list[str]) -> str:
    """
    Build a focused search query from the top 2-3 most specific keywords.

    Previously ALL keywords were joined, producing long noisy queries like:
    "claim verification large language models literature review scientific evidence"
    which caused Semantic Scholar to return poor results or time out.

    Now we take only the top 3 keywords (longest = most specific) and
    wrap multi-word phrases in quotes so the API treats them as phrases,
    not individual words.
    """
    if not keywords:
        return ""

    # Sort by length descending — longer phrases are more specific
    ranked = sorted(keywords, key=len, reverse=True)
    top = ranked[:3]

    # Wrap multi-word phrases in quotes for phrase search
    parts = []
    for kw in top:
        if " " in kw:
            parts.append(f'"{kw}"')
        else:
            parts.append(kw)

    return " ".join(parts)


def _build_headers() -> dict[str, str]:
    headers = {"User-Agent": "ResearchAgent/1.0"}
    if _API_KEY:
        headers["x-api-key"] = _API_KEY
    return headers


def search_semantic_scholar(
    query: str,
    limit: int = MAX_SEARCH_RESULTS_PER_SOURCE,
    retries: int = 3,
) -> list[dict[str, Any]]:
    """
    Search Semantic Scholar with retry logic and a 1-second delay between
    attempts to avoid 429 rate-limit errors.
    """
    if not query:
        return []

    params = {
        "query": query,
        "limit": limit,
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }

    headers = _build_headers()

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 429:
                wait = 2 * attempt
                print(
                    f"[Semantic Scholar] Rate limit hit (429). "
                    f"Waiting {wait}s before retry {attempt}/{retries}..."
                )
                if not _API_KEY:
                    print(
                        "  Tip: set SEMANTIC_SCHOLAR_API_KEY env variable for "
                        "a free 10x higher rate limit."
                    )
                time.sleep(wait)
                continue

            if response.status_code == 400:
                print(
                    f"[Semantic Scholar] Bad request (400). Query may be malformed: {query!r}"
                )
                return []

            response.raise_for_status()
            payload = response.json()
            results = payload.get("data", [])
            print(f"[Semantic Scholar] Found {len(results)} results for query: {query!r}")
            return results

        except requests.RequestException as exc:
            print(f"[Semantic Scholar] Request error (attempt {attempt}/{retries}): {exc}")
            if attempt < retries:
                time.sleep(1)

    print("[Semantic Scholar] All retries exhausted. Returning empty results.")
    return []


def normalize_semantic_scholar_paper(
    raw_paper: dict[str, Any], keywords: list[str]
) -> PaperRecord:
    """
    Convert one raw Semantic Scholar result into a PaperRecord.
    """
    authors = [
        author.get("name", "").strip()
        for author in raw_paper.get("authors", [])
        if author.get("name")
    ]

    external_ids = raw_paper.get("externalIds", {}) or {}
    open_access_pdf = raw_paper.get("openAccessPdf") or {}

    return PaperRecord(
        title=raw_paper.get("title", "").strip(),
        authors=authors,
        year=raw_paper.get("year"),
        abstract=raw_paper.get("abstract"),
        doi=external_ids.get("DOI"),
        arxiv_id=external_ids.get("ArXiv"),
        semantic_scholar_id=raw_paper.get("paperId"),
        venue=raw_paper.get("venue"),
        url=raw_paper.get("url"),
        pdf_url=open_access_pdf.get("url"),
        source="semantic_scholar",
        keywords=keywords,
    )


def discover_papers_from_semantic_scholar(keywords: list[str]) -> list[PaperRecord]:
    """
    End-to-end Semantic Scholar discovery:
    keywords -> focused query -> raw results -> PaperRecord list
    """
    query = build_semantic_scholar_query(keywords)
    if not query:
        return []

    # Polite delay before first request if no API key
    if not _API_KEY:
        time.sleep(2)

    raw_results = search_semantic_scholar(query)

    papers: list[PaperRecord] = []
    for raw_paper in raw_results:
        try:
            paper = normalize_semantic_scholar_paper(raw_paper, keywords)
            if paper.title:
                papers.append(paper)
        except Exception:
            continue

    return papers