from __future__ import annotations

from typing import Any
from urllib.parse import urlencode
import re
import requests
import feedparser
import time

from app.config import MAX_SEARCH_RESULTS_PER_SOURCE
from app.models import PaperRecord

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def build_arxiv_query(keywords: list[str]) -> str:
    """
    Build a focused arXiv search query from the top 3 keywords.

    OLD behaviour: joined ALL keywords with OR — so a query like
        all:"claim verification" OR all:"large language models"
    matched anything touching either keyword, returning completely
    off-topic papers like "Personality in LLMs" or "MRI Segmentation".

    NEW behaviour: use AND between the top 2 most specific keywords
    so both must appear somewhere in the paper (title, abstract, body).
    This is stricter and returns much more relevant results.

    If only 1 keyword exists, fall back to a single-term query.
    """
    if not keywords:
        return ""

    # Sort by length — longer phrases are more specific
    ranked = sorted(keywords, key=len, reverse=True)
    top = ranked[:2]

    cleaned = []
    for kw in top:
        kw = kw.replace("-", " ").strip()
        if kw:
            cleaned.append(kw)

    if len(cleaned) == 1:
        return f'all:"{cleaned[0]}"'

    # AND between the two most specific phrases
    return f'all:"{cleaned[0]}" AND all:"{cleaned[1]}"'


def extract_arxiv_id(entry_id: str) -> str | None:
    """
    Extract arXiv ID from entry.id like:
        http://arxiv.org/abs/1706.03762v7
    """
    if not entry_id:
        return None
    match = re.search(r"/abs/([^\/]+)$", entry_id)
    return match.group(1) if match else None


def search_arxiv(query: str, limit: int = MAX_SEARCH_RESULTS_PER_SOURCE) -> list[Any]:
    """
    Search arXiv and return parsed feed entries, with retry on 429.
    """
    if not query:
        return []

    params = {
        "search_query": query,
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API_URL}?{urlencode(params)}"
    headers = {"User-Agent": "ResearchAgent/1.0"}

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 429:
                wait_time = 3 * (attempt + 1)
                print(f"[arXiv] Rate limit hit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            feed = feedparser.parse(response.text)
            print(f"[arXiv] Found {len(feed.entries)} results for query: {query!r}")
            return feed.entries

        except requests.RequestException as exc:
            print(f"[arXiv] Request error: {exc}")
            return []

    print("[arXiv] All retries exhausted.")
    return []


def normalize_arxiv_paper(entry: Any, keywords: list[str]) -> PaperRecord:
    """
    Convert one arXiv feed entry into a PaperRecord.
    """
    authors = []
    for author in getattr(entry, "authors", []):
        name = getattr(author, "name", "").strip()
        if name:
            authors.append(name)

    pdf_url = None
    for link in getattr(entry, "links", []):
        if getattr(link, "type", "") == "application/pdf":
            pdf_url = getattr(link, "href", None)
            break

    entry_id = getattr(entry, "id", "")
    arxiv_id = extract_arxiv_id(entry_id)

    published = getattr(entry, "published", "")
    year = None
    if published and len(published) >= 4:
        try:
            year = int(published[:4])
        except ValueError:
            year = None

    summary = getattr(entry, "summary", None)
    if summary:
        summary = " ".join(summary.split())

    return PaperRecord(
        title=getattr(entry, "title", "").strip(),
        authors=authors,
        year=year,
        abstract=summary,
        doi=getattr(entry, "arxiv_doi", None),
        arxiv_id=arxiv_id,
        semantic_scholar_id=None,
        venue="arXiv",
        url=entry_id or None,
        pdf_url=pdf_url,
        source="arxiv",
        keywords=keywords,
    )


def discover_papers_from_arxiv(keywords: list[str]) -> list[PaperRecord]:
    """
    End-to-end arXiv discovery:
        keywords -> AND query -> feed entries -> PaperRecord list
    """
    query = build_arxiv_query(keywords)
    if not query:
        return []

    print(f"[arXiv] Query: {query}")
    entries = search_arxiv(query)

    papers: list[PaperRecord] = []
    for entry in entries:
        try:
            paper = normalize_arxiv_paper(entry, keywords)
            if paper.title:
                papers.append(paper)
        except Exception:
            continue

    return papers