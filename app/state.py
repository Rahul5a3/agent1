from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from app.models import Mode, PaperRecord, CitationRecord

class ResearchState(BaseModel):
    mode: Optional[Mode] = None
    user_query: str = ""
    document_type: Optional[str] = None
    extracted_keywords: list[str] = Field(default_factory=list)
    discovered_papers: list[PaperRecord] = Field(default_factory=list)
    citation_metadata: list[CitationRecord] = Field(default_factory=list)
    evidence_chunks: list[dict] = Field(default_factory=list)
    retrieved_evidence: list[dict] = Field(default_factory=list)
    writer_draft: str = ""
    reviewer_issues: list[str] = Field(default_factory=list)
    iteration_count: int = 0