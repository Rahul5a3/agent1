from __future__ import annotations

from pathlib import Path
import re
import requests

from app.config import PAPERS_DIR
from app.models import PaperRecord


BLOCKED_DOMAINS = {
    "academic.oup.com",
    "link.springer.com",
    "sciencedirect.com",
    "ieeexplore.ieee.org",
    "onlinelibrary.wiley.com",
    "dl.acm.org",
    "nature.com",
}


def sanitize_filename(text: str) -> str:
    """
    Convert paper title into a safe PDF filename.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:100] if text else "paper"


def build_preferred_pdf_url(paper: PaperRecord) -> str | None:
    """
    Choose the best candidate PDF URL for a paper.
    Priority:
    1. arXiv direct PDF link if arxiv_id exists
    2. existing pdf_url if present
    """
    if paper.arxiv_id:
        clean_arxiv_id = paper.arxiv_id.replace("v", "v")  # keep version if present
        return f"https://arxiv.org/pdf/{clean_arxiv_id}.pdf"

    if paper.pdf_url:
        return paper.pdf_url

    return None


def looks_like_blocked_landing_page(url: str) -> bool:
    """
    Detect common publisher landing pages that are unlikely to allow direct PDF download.
    """
    lowered = url.lower()

    if lowered.endswith(".pdf"):
        return False

    return any(domain in lowered for domain in BLOCKED_DOMAINS)


def download_pdf(paper: PaperRecord) -> Path | None:
    """
    Download one paper PDF if a usable PDF source exists.
    Returns the saved file path, or None if download fails.
    """
    pdf_url = build_preferred_pdf_url(paper)
    if not pdf_url:
        print(f"Skipping '{paper.title}': no PDF URL available.")
        return None

    if looks_like_blocked_landing_page(pdf_url):
        print(f"Skipping '{paper.title}': URL looks like a publisher landing page.")
        return None

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename(paper.title)
    filename = f"{safe_title}.pdf"
    output_path = PAPERS_DIR / filename

    if output_path.exists():
        return output_path

    headers = {
        "User-Agent": "Agent1ResearchAssistant/0.1"
    }

    try:
        response = requests.get(pdf_url, headers=headers, timeout=60, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            print(f"Skipping '{paper.title}': response is not a PDF.")
            return None

        output_path.write_bytes(response.content)
        return output_path

    except requests.RequestException as exc:
        print(f"PDF download failed for '{paper.title}': {exc}")
        return None


def download_pdfs(papers: list[PaperRecord]) -> dict[str, Path]:
    """
    Download PDFs for all papers that provide a usable PDF source.
    Returns a mapping:
        paper title -> saved path
    """
    downloaded: dict[str, Path] = {}

    for paper in papers:
        path = download_pdf(paper)
        if path is not None:
            downloaded[paper.title] = path

    return downloaded