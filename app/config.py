from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_DIR = BASE_DIR / "storage"
PAPERS_DIR = STORAGE_DIR / "papers"
METADATA_DIR = STORAGE_DIR / "metadata"
BIB_DIR = STORAGE_DIR / "bib"
INDEX_DIR = STORAGE_DIR / "indexes"

MIN_PAPERS_FOR_REVIEW = 5

# Bumped from 3 -> 15.
# With only 3 results per source and a relevance filter, you could easily
# end up with 0 usable papers. Fetch more, filter aggressively after.
MAX_SEARCH_RESULTS_PER_SOURCE = 15

# Optional: Semantic Scholar free API key
# Get one at https://www.semanticscholar.org/product/api
# Set as environment variable: export SEMANTIC_SCHOLAR_API_KEY="your_key"
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")


def ensure_directories() -> None:
    for path in [STORAGE_DIR, PAPERS_DIR, METADATA_DIR, BIB_DIR, INDEX_DIR]:
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"{path} exists but is not a directory")
        path.mkdir(parents=True, exist_ok=True)