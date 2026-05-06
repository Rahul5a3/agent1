from __future__ import annotations

from app.config import MIN_PAPERS_FOR_REVIEW, INDEX_DIR
from app.models import Mode
from app.state import ResearchState

DISCOVERY_KEYWORDS = {
    "find papers",
    "recommend papers",
    "related work",
    "discover papers",
    "search papers",
    "citations",
    "bibtex",
    "reading list",
    "literature search",
}

REVIEW_KEYWORDS = {
    "literature review",
    "write review",
    "survey",
    "review section",
    "synthesize",
    "summary of papers",
    "what evidence",
    "what do the papers say",
    "evidence exists",
}

def detect_query_intent(user_query: str) -> str:
    """
    Detect whether the query is mainly asking for discovery or review.
    Returns one of:
        - "discovery"
        - "review"
        - "unknown"
    """
    query = user_query.lower().strip()
    # Stron discovery intent should win first
    # Priority 1: explicit discovery actions
    for phrase in DISCOVERY_KEYWORDS:
        if phrase in query:
            return "discovery"
    # Review intent only if no strong discovery phrase is present
    # Priority 2: review-generation actions
    for phrase in REVIEW_KEYWORDS:
        if phrase in query:
            return "review"
    
    return "unknown"


def local_index_exists(index_name: str = "agent1_index") -> bool:
    index_path = INDEX_DIR / f"{index_name}.faiss"
    metadata_path = INDEX_DIR / f"{index_name}_metadata.json"
    return index_path.exists() and metadata_path.exists()


def route_mode(state: ResearchState) -> Mode:
    """
    Decide the system mode using:
    1. document type
    2. query intent
    3. number of available papers
    """
    paper_count = len(state.discovered_papers)
    query_intent = detect_query_intent(state.user_query)
    document_type = state.document_type
    has_local_index = local_index_exists()
    #case 1: proposal-only input should be discovery
    if document_type =="propoal":
        return Mode.DISCOVERY
    
    #case 2: explicit discovery-style query
    if query_intent == "discovery":
        return Mode.DISCOVERY
    
    #case 3: review requested and enough enough papers exist
    if query_intent == "review" and paper_count >= MIN_PAPERS_FOR_REVIEW:
        return Mode.REVIEW
    
    #case 4: review requested but not enough papers, fallback to discovery
    if query_intent == "review" and paper_count < MIN_PAPERS_FOR_REVIEW:
        return Mode.HYBRID
    
    #Fallback rules:
    if paper_count >= MIN_PAPERS_FOR_REVIEW:
        return Mode.REVIEW
    
    return Mode.DISCOVERY
