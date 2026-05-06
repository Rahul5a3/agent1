from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class Mode(str, Enum):
    DISCOVERY = "discovery"
    REVIEW = "review"
    HYBRID = "hybrid"

class PaperRecord(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    source: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)  

class CitationRecord(BaseModel):
    paper_title: str
    citation_key: str
    bibtex: str
    source: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None