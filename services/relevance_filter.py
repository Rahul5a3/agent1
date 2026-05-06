from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models import PaperRecord

# -------------------------------------------------------------------
# Semantic relevance filter
#
# Previously: pure string overlap (keyword in title/abstract).
# Problems with that approach:
#   - "LLM" would not match "large language model"
#   - synonyms and paraphrases scored zero
#   - IMPORTANT_TERMS dict was defined but never used (dead code)
#   - paper.keywords (query keywords stored on the record) were ignored
#
# Now: cosine similarity between a sentence embedding of the query
# and the paper's title+abstract embedding. Falls back to keyword
# overlap if the model fails to load.
# -------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer | None:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(_MODEL_NAME)
        except Exception as exc:
            print(f"[RelevanceFilter] Could not load embedding model: {exc}")
            print("[RelevanceFilter] Falling back to keyword overlap scoring.")
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _keyword_overlap_score(paper: PaperRecord, keywords: list[str]) -> float:
    """
    Fallback scorer: keyword overlap against title + abstract.
    Returns a normalised float in [0, 1].
    """
    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()

    hits = 0
    for kw in keywords:
        kw = kw.lower()
        if kw in title:
            hits += 2      # title match is worth more
        elif kw in abstract:
            hits += 1

    max_possible = len(keywords) * 2
    if max_possible == 0:
        return 0.0
    return hits / max_possible


def score_papers_semantic(
    papers: list[PaperRecord],
    keywords: list[str],
) -> list[tuple[float, PaperRecord]]:
    """
    Score all papers at once using batch embedding (fast).

    The query representation is built from the extracted keywords joined
    into a sentence — this is a better embedding target than a raw keyword
    list because the model was trained on natural-language sentences.

    Each paper is represented as:  title + ". " + abstract (first 512 chars)
    This mirrors what the model was trained on and gives it the most signal.
    """
    model = _get_model()

    if model is None:
        # Fallback: keyword overlap
        return [
            (_keyword_overlap_score(p, keywords), p)
            for p in papers
        ]

    # Build query text from keywords
    query_text = " ".join(keywords)

    # Build paper texts — title + abstract
    paper_texts = []
    for p in papers:
        title = (p.title or "").strip()
        abstract = (p.abstract or "")[:512].strip()
        paper_texts.append(f"{title}. {abstract}" if abstract else title)

    # Batch encode everything together (one model pass = fast)
    all_texts = [query_text] + paper_texts
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

    query_emb = embeddings[0]
    paper_embs = embeddings[1:]

    scored = []
    for paper, emb in zip(papers, paper_embs):
        sim = _cosine_similarity(query_emb, emb)
        scored.append((sim, paper))

    return scored


def filter_and_rank_papers(
    papers: list[PaperRecord],
    keywords: list[str],
    min_score: float = 0.25,   # cosine similarity threshold (0–1 scale)
) -> list[PaperRecord]:
    """
    Score, filter, and rank papers by semantic relevance to the query.

    min_score=0.25 means "at least 25% cosine similarity" — this filters
    out completely off-topic papers while keeping loosely related ones.
    Raise to 0.35–0.4 if you want stricter filtering.

    Note: if the embedding model falls back to keyword overlap, the score
    is a normalised fraction (also 0–1), so the threshold still applies.
    """
    if not papers:
        return []

    scored = score_papers_semantic(papers, keywords)

    # Filter by threshold
    filtered = [(score, paper) for score, paper in scored if score >= min_score]

    if not filtered:
        # Threshold too strict — return top 3 regardless
        print(
            f"[RelevanceFilter] No papers passed min_score={min_score}. "
            "Returning top 3 by score anyway."
        )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:3]]

    # Sort descending by score
    filtered.sort(key=lambda x: x[0], reverse=True)

    print(f"[RelevanceFilter] {len(filtered)}/{len(papers)} papers passed threshold {min_score}.")
    for score, paper in filtered[:5]:
        print(f"  {score:.3f}  {paper.title[:70]}")

    return [p for _, p in filtered]