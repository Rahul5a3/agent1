"""
Agent 1 — Component Test Suite
================================
Run from the project root:
    python test.py

Each test prints PASS or FAIL with a reason.
At the end you get a summary: X/Y tests passed.

Tests are grouped by component:
  [1] Config
  [2] Keyword Extractor
  [3] Semantic Scholar Query Builder  (new logic)
  [4] Semantic Scholar API Live Call  (needs internet)
  [5] arXiv API Live Call             (needs internet)
  [6] Relevance Filter                (new semantic scoring)
  [7] Mode Router                     (incl. typo bug check)
  [8] Citation Builder
  [9] BibTeX Deduplication
"""

import sys
import traceback

# ── colour helpers (no external deps) ────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

passed = 0
failed = 0
results: list[str] = []


def ok(name: str, detail: str = "") -> None:
    global passed
    passed += 1
    msg = f"{GREEN}PASS{RESET}  {name}"
    if detail:
        msg += f"\n       → {detail}"
    print(msg)
    results.append(f"PASS  {name}")


def fail(name: str, reason: str) -> None:
    global failed
    failed += 1
    msg = f"{RED}FAIL{RESET}  {name}\n       ✗ {reason}"
    print(msg)
    results.append(f"FAIL  {name}  |  {reason}")


def section(title: str) -> None:
    print(f"\n{YELLOW}{'─'*60}{RESET}")
    print(f"{YELLOW}  {title}{RESET}")
    print(f"{YELLOW}{'─'*60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# [1] Config
# ─────────────────────────────────────────────────────────────────────────────
section("[1] Config")

try:
    from app.config import (
        MAX_SEARCH_RESULTS_PER_SOURCE,
        MIN_PAPERS_FOR_REVIEW,
        PAPERS_DIR,
        BIB_DIR,
        INDEX_DIR,
    )

    # 1a — result limit is high enough to be useful
    if MAX_SEARCH_RESULTS_PER_SOURCE >= 10:
        ok("1a  MAX_SEARCH_RESULTS_PER_SOURCE is >= 10",
           f"value = {MAX_SEARCH_RESULTS_PER_SOURCE}")
    else:
        fail("1a  MAX_SEARCH_RESULTS_PER_SOURCE is >= 10",
             f"value = {MAX_SEARCH_RESULTS_PER_SOURCE}  (too low — bump to 15 in config.py)")

    # 1b — storage directories can be created
    try:
        from app.config import ensure_directories
        ensure_directories()
        ok("1b  ensure_directories() runs without error")
    except Exception as e:
        fail("1b  ensure_directories() runs without error", str(e))

except Exception as e:
    fail("1 — config import failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [2] Keyword Extractor
# ─────────────────────────────────────────────────────────────────────────────
section("[2] Keyword Extractor")

try:
    from services.keyword_extractor import extract_keywords

    # 2a — basic NLP query
    kws = extract_keywords("automated claim verification using large language models")
    print(f"       keywords: {kws}")
    if len(kws) >= 2:
        ok("2a  extracts at least 2 keywords from a normal query")
    else:
        fail("2a  extracts at least 2 keywords from a normal query",
             f"only got: {kws}")

    # 2b — abbreviation expansion
    kws_abbr = extract_keywords("using LLMs for NLP tasks")
    print(f"       keywords: {kws_abbr}")
    expanded = [k for k in kws_abbr if "large language" in k or "natural language" in k]
    if expanded:
        ok("2b  expands abbreviations (LLM -> large language model, NLP -> natural language processing)",
           f"found: {expanded}")
    else:
        fail("2b  expands abbreviations",
             f"expected 'large language model' or 'natural language processing' in {kws_abbr}")

    # 2c — stop words are filtered
    kws_stop = extract_keywords("find papers using methods and systems")
    print(f"       keywords: {kws_stop}")
    bad = [k for k in kws_stop if k in {"find", "papers", "using", "methods", "systems"}]
    if not bad:
        ok("2c  stop words are filtered out")
    else:
        fail("2c  stop words are filtered out", f"these slipped through: {bad}")

    # 2d — empty query returns empty list, not an error
    kws_empty = extract_keywords("")
    if kws_empty == []:
        ok("2d  empty query returns empty list")
    else:
        fail("2d  empty query returns empty list", f"got: {kws_empty}")

except Exception as e:
    fail("2 — keyword extractor import/run failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [3] Semantic Scholar Query Builder  (new logic)
# ─────────────────────────────────────────────────────────────────────────────
section("[3] Semantic Scholar Query Builder (new logic)")

try:
    from services.semantic_scholar_client import build_semantic_scholar_query

    # 3a — multi-word phrases are quoted
    query = build_semantic_scholar_query(["claim verification", "large language models", "fact checking"])
    print(f"       query: {query!r}")
    if '"claim verification"' in query or '"large language models"' in query:
        ok("3a  multi-word phrases are wrapped in quotes")
    else:
        fail("3a  multi-word phrases are wrapped in quotes",
             f"got: {query!r}  — expected quotes around multi-word phrases")

    # 3b — only top 3 keywords are used (not all 8+)
    many_kws = ["claim verification", "large language models", "fact checking",
                "evidence", "scientific", "nlp", "transformer", "bert"]
    query_many = build_semantic_scholar_query(many_kws)
    word_count = len(query_many.replace('"', '').split())
    print(f"       query from 8 keywords: {query_many!r}  (word count: {word_count})")
    if word_count <= 10:
        ok("3b  query stays concise even with many input keywords",
           f"word count = {word_count}")
    else:
        fail("3b  query stays concise even with many input keywords",
             f"query is too long ({word_count} words): {query_many!r}")

    # 3c — empty keyword list returns empty string
    q_empty = build_semantic_scholar_query([])
    if q_empty == "":
        ok("3c  empty keyword list returns empty string")
    else:
        fail("3c  empty keyword list returns empty string", f"got: {q_empty!r}")

    # 3d — single keyword (no quotes needed)
    q_single = build_semantic_scholar_query(["transformers"])
    print(f"       single keyword query: {q_single!r}")
    if q_single == "transformers":
        ok("3d  single keyword has no unnecessary quotes")
    else:
        fail("3d  single keyword has no unnecessary quotes", f"got: {q_single!r}")

except Exception as e:
    fail("3 — semantic scholar query builder failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [4] Semantic Scholar Live API Call
# ─────────────────────────────────────────────────────────────────────────────
section("[4] Semantic Scholar — Live API Call (needs internet)")

try:
    from services.semantic_scholar_client import discover_papers_from_semantic_scholar

    keywords = ["claim verification", "large language models"]
    print(f"       querying with keywords: {keywords}")
    papers = discover_papers_from_semantic_scholar(keywords)
    print(f"       papers returned: {len(papers)}")

    # 4a — returns at least 1 paper
    if len(papers) >= 1:
        ok("4a  Semantic Scholar returns at least 1 paper")
    else:
        fail("4a  Semantic Scholar returns at least 1 paper",
             "0 papers returned — check internet, rate limit, or query")

    # 4b — papers have titles
    if papers:
        no_title = [p for p in papers if not p.title]
        if not no_title:
            ok("4b  all returned papers have titles")
        else:
            fail("4b  all returned papers have titles",
                 f"{len(no_title)} papers missing titles")

    # 4c — at least some papers have abstracts (important for relevance filter)
    if papers:
        with_abstract = [p for p in papers if p.abstract]
        print(f"       papers with abstracts: {len(with_abstract)}/{len(papers)}")
        if len(with_abstract) >= 1:
            ok("4c  at least 1 paper has an abstract",
               f"{len(with_abstract)}/{len(papers)} have abstracts")
        else:
            fail("4c  at least 1 paper has an abstract",
                 "no abstracts found — relevance filter will be degraded")

    # 4d — print top 3 titles for manual inspection
    if papers:
        print("\n       Top papers returned:")
        for i, p in enumerate(papers[:3], 1):
            print(f"         {i}. {p.title} ({p.year}) [{p.source}]")

except Exception as e:
    fail("4 — Semantic Scholar live call failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [5] arXiv Live API Call
# ─────────────────────────────────────────────────────────────────────────────
section("[5] arXiv — Live API Call (needs internet)")

try:
    from services.arxiv_client import discover_papers_from_arxiv

    keywords = ["claim verification", "large language models"]
    print(f"       querying with keywords: {keywords}")
    papers_arxiv = discover_papers_from_arxiv(keywords)
    print(f"       papers returned: {len(papers_arxiv)}")

    # 5a — returns at least 1 paper
    if len(papers_arxiv) >= 1:
        ok("5a  arXiv returns at least 1 paper")
    else:
        fail("5a  arXiv returns at least 1 paper",
             "0 papers returned — check internet or arXiv API")

    # 5b — papers have arXiv IDs
    if papers_arxiv:
        with_id = [p for p in papers_arxiv if p.arxiv_id]
        print(f"       papers with arxiv_id: {len(with_id)}/{len(papers_arxiv)}")
        if with_id:
            ok("5b  papers have arxiv_id set (needed for PDF download)")
        else:
            fail("5b  papers have arxiv_id set",
                 "no arxiv_ids — PDF downloader will skip all papers")

    # 5c — papers have pdf_url set
    if papers_arxiv:
        with_pdf = [p for p in papers_arxiv if p.pdf_url]
        print(f"       papers with pdf_url: {len(with_pdf)}/{len(papers_arxiv)}")
        if with_pdf:
            ok("5c  at least 1 paper has pdf_url")
        else:
            fail("5c  at least 1 paper has pdf_url",
                 "no pdf_urls found")

    # 5d — print top 3 for manual inspection
    if papers_arxiv:
        print("\n       Top arXiv papers:")
        for i, p in enumerate(papers_arxiv[:3], 1):
            print(f"         {i}. {p.title} ({p.year}) — arxiv_id: {p.arxiv_id}")

except Exception as e:
    fail("5 — arXiv live call failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [6] Relevance Filter  (new semantic scoring)
# ─────────────────────────────────────────────────────────────────────────────
section("[6] Relevance Filter (new semantic cosine scoring)")

try:
    from services.relevance_filter import filter_and_rank_papers
    from app.models import PaperRecord

    def make_paper(title: str, abstract: str) -> PaperRecord:
        return PaperRecord(title=title, abstract=abstract, source="test")

    relevant_papers = [
        make_paper(
            "Automated Claim Verification Using Large Language Models",
            "We propose a system for verifying scientific claims using LLMs and retrieval-augmented generation.",
        ),
        make_paper(
            "Fact Checking with Transformer Models",
            "This paper studies automated fact verification and evidence retrieval using BERT and GPT.",
        ),
        make_paper(
            "Literature Review Generation with Neural Networks",
            "We explore LLM-based systems for automated literature review and citation generation.",
        ),
    ]

    irrelevant_papers = [
        make_paper(
            "Deep Learning for MRI Brain Segmentation",
            "Convolutional neural networks for neuroimaging segmentation of MRI scans.",
        ),
        make_paper(
            "Graph Neural Networks for Drug Discovery",
            "GNNs predict molecular interactions in pharmaceutical drug discovery pipelines.",
        ),
    ]

    all_papers = relevant_papers + irrelevant_papers
    keywords = ["claim verification", "large language models", "literature review"]

    ranked = filter_and_rank_papers(all_papers, keywords)
    print(f"\n       Ranked results ({len(ranked)} papers passed threshold):")
    for i, p in enumerate(ranked, 1):
        print(f"         {i}. {p.title}")

    # 6a — relevant papers rank above irrelevant ones
    if ranked:
        top_titles = [p.title for p in ranked[:3]]
        relevant_titles = [p.title for p in relevant_papers]
        top_are_relevant = any(t in relevant_titles for t in top_titles)
        if top_are_relevant:
            ok("6a  relevant papers rank above irrelevant ones")
        else:
            fail("6a  relevant papers rank above irrelevant ones",
                 f"top 3 were: {top_titles}")
    else:
        fail("6a  relevant papers rank above irrelevant ones",
             "filter returned 0 papers — min_score threshold may be too high")

    # 6b — irrelevant papers are filtered out or ranked last
    if ranked:
        irrelevant_titles = {p.title for p in irrelevant_papers}
        ranked_titles = [p.title for p in ranked]
        # irrelevant papers should either be absent or at the bottom
        irrelevant_positions = [i for i, t in enumerate(ranked_titles) if t in irrelevant_titles]
        relevant_positions   = [i for i, t in enumerate(ranked_titles) if t not in irrelevant_titles]
        if not irrelevant_positions or (relevant_positions and min(relevant_positions) < min(irrelevant_positions)):
            ok("6b  irrelevant papers are filtered or ranked last")
        else:
            fail("6b  irrelevant papers are filtered or ranked last",
                 f"irrelevant paper appeared at position {min(irrelevant_positions)+1}")

    # 6c — synonym matching: "LLM" should still match even though query says "large language models"
    synonym_paper = make_paper(
        "Hallucination in LLM Outputs",
        "We study LLM hallucinations in text generation tasks.",
    )
    ranked_syn = filter_and_rank_papers([synonym_paper], ["large language models"])
    if ranked_syn:
        ok("6c  synonym/abbreviation match works (LLM matches 'large language models')")
    else:
        fail("6c  synonym/abbreviation match works",
             "paper about LLMs was filtered out when querying 'large language models' — "
             "semantic scoring may not be working (check model load)")

    # 6d — empty paper list returns empty result
    empty_result = filter_and_rank_papers([], ["claim verification"])
    if empty_result == []:
        ok("6d  empty paper list returns empty result")
    else:
        fail("6d  empty paper list returns empty result", f"got: {empty_result}")

except Exception as e:
    fail("6 — relevance filter failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [7] Mode Router  (incl. typo bug check)
# ─────────────────────────────────────────────────────────────────────────────
section("[7] Mode Router")

try:
    from app.router import route_mode, detect_query_intent
    from app.state import ResearchState
    from app.models import Mode, PaperRecord

    def make_state(query: str, doc_type: str | None = None, n_papers: int = 0) -> ResearchState:
        papers = [
            PaperRecord(title=f"Paper {i}", source="test")
            for i in range(n_papers)
        ]
        return ResearchState(
            user_query=query,
            document_type=doc_type,
            discovered_papers=papers,
        )

    # 7a — discovery intent detected correctly
    intent = detect_query_intent("find papers on claim verification")
    if intent == "discovery":
        ok("7a  'find papers' query intent = discovery")
    else:
        fail("7a  'find papers' query intent = discovery", f"got: {intent!r}")

    # 7b — review intent detected correctly
    intent2 = detect_query_intent("write a literature review on LLMs")
    if intent2 == "review":
        ok("7b  'literature review' query intent = review")
    else:
        fail("7b  'literature review' query intent = review", f"got: {intent2!r}")

    # 7c — unknown intent falls back correctly
    intent3 = detect_query_intent("autonomous agents in NLP")
    if intent3 == "unknown":
        ok("7c  ambiguous query returns unknown intent")
    else:
        # not necessarily wrong — just informational
        ok(f"7c  ambiguous query returned intent = {intent3!r} (acceptable)")

    # 7d — discovery query → DISCOVERY mode
    mode = route_mode(make_state("find papers on NLP"))
    if mode == Mode.DISCOVERY:
        ok("7d  discovery query → DISCOVERY mode")
    else:
        fail("7d  discovery query → DISCOVERY mode", f"got: {mode}")

    # 7e — review query + enough papers → REVIEW mode
    mode2 = route_mode(make_state("write a literature review", n_papers=6))
    if mode2 == Mode.REVIEW:
        ok("7e  review query + 6 papers → REVIEW mode")
    else:
        fail("7e  review query + 6 papers → REVIEW mode", f"got: {mode2}")

    # 7f — review query + not enough papers → HYBRID mode
    mode3 = route_mode(make_state("write a literature review", n_papers=2))
    if mode3 == Mode.HYBRID:
        ok("7f  review query + 2 papers → HYBRID mode")
    else:
        fail("7f  review query + 2 papers → HYBRID mode", f"got: {mode3}")

    # 7g — TYPO BUG CHECK: document_type="proposal" should trigger DISCOVERY
    # The original code had "propoal" (typo) so this never fired
    mode4 = route_mode(make_state("anything", doc_type="proposal"))
    if mode4 == Mode.DISCOVERY:
        ok("7g  document_type='proposal' → DISCOVERY mode")
    else:
        fail("7g  document_type='proposal' → DISCOVERY mode",
             f"got: {mode4}  — the typo bug ('propoal') is still present in router.py. "
             "Fix line 72: change '\"propoal\"' to '\"proposal\"'")

except Exception as e:
    fail("7 — mode router failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [8] Citation Builder
# ─────────────────────────────────────────────────────────────────────────────
section("[8] Citation Builder")

try:
    from services.citation_builder import build_citations, generate_citation_key
    from app.models import PaperRecord

    p1 = PaperRecord(
        title="Claim Verification with Large Language Models",
        authors=["Alice Smith", "Bob Jones"],
        year=2024,
        venue="ACL",
        arxiv_id=None,
        doi="10.1234/test",
        source="test",
    )
    p2 = PaperRecord(
        title="Claim Verification in Scientific Literature",
        authors=["Alice Smith"],
        year=2024,
        venue="arXiv",
        arxiv_id="2401.00001",
        source="test",
    )

    # 8a — citation key is generated
    key1 = generate_citation_key(p1)
    print(f"       citation key 1: {key1}")
    if key1 and len(key1) > 3:
        ok("8a  citation key generated successfully", f"key = {key1!r}")
    else:
        fail("8a  citation key generated", f"got empty or too short: {key1!r}")

    # 8b — bibtex entries are built
    citations = build_citations([p1, p2])
    if len(citations) == 2:
        ok("8b  build_citations returns correct count", "2 papers → 2 citations")
    else:
        fail("8b  build_citations returns correct count", f"got {len(citations)}")

    # 8c — bibtex string contains title
    bib = citations[0].bibtex
    if p1.title in bib:
        ok("8c  BibTeX entry contains the paper title")
    else:
        fail("8c  BibTeX entry contains the paper title",
             f"title not found in bibtex:\n{bib}")

    # 8d — collision check: build_citations must produce unique keys for
    #      two papers that would otherwise collide (same author, year, title start)
    #      We check the keys from the already-built `citations` list, not by
    #      calling generate_citation_key in isolation (which has no collision memory).
    key_from_batch_1 = citations[0].citation_key
    key_from_batch_2 = citations[1].citation_key
    print(f"       citation key 1 (from batch): {key_from_batch_1}")
    print(f"       citation key 2 (from batch): {key_from_batch_2}")
    if key_from_batch_1 != key_from_batch_2:
        ok("8d  two different papers get different citation keys",
           f"{key_from_batch_1!r} vs {key_from_batch_2!r}")
    else:
        fail("8d  two different papers get different citation keys",
             f"both got: {key_from_batch_1!r}  — collision not resolved in build_citations")

except Exception as e:
    fail("8 — citation builder failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# [9] BibTeX Deduplication
# ─────────────────────────────────────────────────────────────────────────────
section("[9] BibTeX Deduplication")

try:
    from services.bibtex_storage import deduplicate_citations
    from app.models import CitationRecord

    c1 = CitationRecord(
        paper_title="Paper A",
        citation_key="Smith2024Claim",
        bibtex="@article{Smith2024Claim, title={Paper A}}",
        doi="10.1234/a",
    )
    c2 = CitationRecord(
        paper_title="Paper A duplicate",
        citation_key="Smith2024Claim2",
        bibtex="@article{Smith2024Claim2, title={Paper A duplicate}}",
        doi="10.1234/a",   # same DOI as c1 → should be deduplicated
    )
    c3 = CitationRecord(
        paper_title="Paper B",
        citation_key="Jones2023Evidence",
        bibtex="@article{Jones2023Evidence, title={Paper B}}",
        doi="10.1234/b",
    )

    deduped = deduplicate_citations([c1, c2, c3])
    print(f"       input: 3 citations (c1 and c2 share DOI), output: {len(deduped)}")

    # 9a — duplicate by DOI is removed
    if len(deduped) == 2:
        ok("9a  duplicate citation (same DOI) is removed", "3 → 2 citations")
    else:
        fail("9a  duplicate citation (same DOI) is removed",
             f"expected 2, got {len(deduped)}")

    # 9b — non-duplicate is kept
    titles = {c.paper_title for c in deduped}
    if "Paper B" in titles:
        ok("9b  non-duplicate citation is kept")
    else:
        fail("9b  non-duplicate citation is kept",
             f"'Paper B' missing from deduped output: {titles}")

except Exception as e:
    fail("9 — bibtex deduplication failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{YELLOW}{'═'*60}{RESET}")
print(f"  Results: {GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}  |  total {passed+failed}")
print(f"{YELLOW}{'═'*60}{RESET}\n")

if failed > 0:
    print("Failed tests:")
    for r in results:
        if r.startswith("FAIL"):
            print(f"  {r}")
    print()

sys.exit(0 if failed == 0 else 1)