"""
api.py — FastAPI backend for Agent 1: Research Discovery Agent
===============================================================
Place this file in your project root (same level as main.py).

Install extra dependency:
    pip install fastapi uvicorn sse-starlette

Run:
    uvicorn api:app --reload --port 8000

The frontend connects to:
    POST /run        — starts a research job, streams progress via SSE
    GET  /bib/{job}  — download the generated .bib file
    GET  /health     — health check
"""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.config import ensure_directories
from app.models import Mode
from app.router import route_mode
from app.state import ResearchState
from services.arxiv_client import discover_papers_from_arxiv
from services.bibtex_storage import deduplicate_citations, save_bibtex_file
from services.citation_builder import build_citations
from services.index_builder import build_faiss_index, save_faiss_index
from services.keyword_extractor import extract_keywords
from services.local_retriever import search_local_index
from services.pdf_downloader import download_pdfs
from services.pdf_text_extractor import build_evidence_chunks
from services.relevance_filter import filter_and_rank_papers
from services.semantic_scholar_client import discover_papers_from_semantic_scholar

app = FastAPI(title="Research Agent 1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for completed job results (keyed by job_id)
job_results: dict[str, dict] = {}


class RunRequest(BaseModel):
    query: str
    document_type: str | None = None  # "proposal" | "paper" | None


def event(kind: str, **data) -> dict:
    """Helper to build a structured SSE event payload."""
    return {"event": kind, "data": json.dumps({"type": kind, **data})}


async def run_pipeline(req: RunRequest, job_id: str) -> AsyncGenerator[dict, None]:
    """
    Runs the full Agent 1 pipeline and yields SSE events for each step.
    The frontend listens to these events and updates the UI in real time.
    """
    ensure_directories()

    state = ResearchState(
        user_query=req.query,
        document_type=req.document_type,
    )

    bib_path = None
    downloaded_pdfs = {}

    try:
        # ── Step 1: Keywords ──────────────────────────────────────────────
        yield event("step", step=1, message="Extracting keywords from your query...")
        await asyncio.sleep(0)

        state.extracted_keywords = extract_keywords(state.user_query)
        yield event("keywords", keywords=state.extracted_keywords)

        # ── Step 2: Mode routing ──────────────────────────────────────────
        yield event("step", step=2, message="Determining operation mode...")
        await asyncio.sleep(0)

        state.mode = route_mode(state)
        yield event("mode", mode=state.mode.value)

        # ── Step 3: Discovery ─────────────────────────────────────────────
        if state.mode.value in {"discovery", "hybrid"}:
            yield event("step", step=3, message="Searching Semantic Scholar...")
            await asyncio.sleep(0)

            loop = asyncio.get_event_loop()
            state.discovered_papers = await loop.run_in_executor(
                None, discover_papers_from_semantic_scholar, state.extracted_keywords
            )

            if not state.discovered_papers:
                yield event("step", step=3, message="Semantic Scholar empty — trying arXiv...")
                state.discovered_papers = await loop.run_in_executor(
                    None, discover_papers_from_arxiv, state.extracted_keywords
                )

            yield event("step", step=4, message=f"Ranking {len(state.discovered_papers)} papers by relevance...")
            await asyncio.sleep(0)

            state.discovered_papers = await loop.run_in_executor(
                None, filter_and_rank_papers, state.discovered_papers, state.extracted_keywords
            )

            papers_payload = [
                {
                    "title": p.title,
                    "authors": p.authors[:3],
                    "year": p.year,
                    "venue": p.venue,
                    "source": p.source,
                    "url": p.url,
                    "abstract": (p.abstract or "")[:300],
                }
                for p in state.discovered_papers
            ]
            yield event("papers", papers=papers_payload, count=len(state.discovered_papers))

            # ── Step 5: Citations ─────────────────────────────────────────
            yield event("step", step=5, message="Building BibTeX citations...")
            await asyncio.sleep(0)

            state.citation_metadata = build_citations(state.discovered_papers)
            state.citation_metadata = deduplicate_citations(state.citation_metadata)
            bib_path = save_bibtex_file(state.citation_metadata, state.user_query)

            citations_payload = [
                {"key": c.citation_key, "bibtex": c.bibtex}
                for c in state.citation_metadata[:10]
            ]
            yield event("citations", citations=citations_payload, bib_job_id=job_id)

            # Step 6: PDFs
            yield event("step", step=6, message="Downloading open-access PDFs...")
            await asyncio.sleep(0)

            downloaded_pdfs = await loop.run_in_executor(
                None, download_pdfs, state.discovered_papers
            )
            yield event("pdfs", count=len(downloaded_pdfs),
                        titles=list(downloaded_pdfs.keys()))

            # Step 7: Chunks + Index
            yield event("step", step=7, message="Extracting text and building search index...")
            await asyncio.sleep(0)

            state.evidence_chunks = build_evidence_chunks(downloaded_pdfs)

            # Fallback: index abstracts if no PDFs downloaded
            if not state.evidence_chunks:
                yield event("step", step=7, message="No PDFs available — indexing abstracts as fallback...")
                from services.pdf_text_extractor import build_abstract_chunks
                state.evidence_chunks = build_abstract_chunks(state.discovered_papers)

            if state.evidence_chunks:
                index, metadata = await loop.run_in_executor(
                    None, build_faiss_index, state.evidence_chunks
                )
                if index is not None:
                    save_faiss_index(index, metadata)

            yield event("index", chunks=len(state.evidence_chunks))

        # ── Step 8: Review retrieval ──────────────────────────────────────
        if state.mode.value == "review":
            yield event("step", step=8, message="Retrieving evidence from local index...")
            await asyncio.sleep(0)

            loop = asyncio.get_event_loop()
            review_results = await loop.run_in_executor(
                None, lambda: search_local_index(state.user_query, top_k=5)
            )
            yield event("retrieval", results=[
                {
                    "rank": r["rank"],
                    "paper": r["paper_title"],
                    "chunk_id": r["chunk_id"],
                    "distance": round(r["distance"], 4),
                    "text": r["chunk_text"][:300],
                }
                for r in review_results
            ])

        # ── Done ──────────────────────────────────────────────────────────
        job_results[job_id] = {"bib_path": str(bib_path) if bib_path else None}
        yield event("done", job_id=job_id, mode=state.mode.value,
                    paper_count=len(state.discovered_papers),
                    chunk_count=len(state.evidence_chunks))

    except Exception:
        yield event("error", message=traceback.format_exc())


@app.post("/run")
async def run_agent(req: RunRequest):
    """Start a research job and stream progress as Server-Sent Events."""
    if not req.query.strip():
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

    job_id = str(uuid.uuid4())[:8]

    async def stream():
        async for evt in run_pipeline(req, job_id):
            yield evt

    return EventSourceResponse(stream())


@app.get("/bib/{job_id}")
async def download_bib(job_id: str):
    """Download the generated .bib file for a completed job."""
    result = job_results.get(job_id)
    if not result or not result.get("bib_path"):
        return JSONResponse({"error": "BibTeX file not found for this job"}, status_code=404)

    bib_path = Path(result["bib_path"])
    if not bib_path.exists():
        return JSONResponse({"error": "File no longer exists on disk"}, status_code=404)

    return FileResponse(
        path=bib_path,
        media_type="text/plain",
        filename=bib_path.name,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "Agent1-ResearchDiscovery"}
