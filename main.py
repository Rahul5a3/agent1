from app.config import ensure_directories
from app.router import route_mode
from app.state import ResearchState
from app.models import Mode

from services.keyword_extractor import extract_keywords
from services.semantic_scholar_client import discover_papers_from_semantic_scholar
from services.arxiv_client import discover_papers_from_arxiv
from services.relevance_filter import filter_and_rank_papers
from services.citation_builder import build_citations
from services.bibtex_storage import deduplicate_citations, save_bibtex_file
from services.pdf_downloader import download_pdfs
from services.pdf_text_extractor import build_evidence_chunks
from services.index_builder import build_faiss_index, save_faiss_index
from services.local_retriever import search_local_index


def main() -> None:
    ensure_directories()

    query = input("Enter your research topic or proposal idea: ").strip()
    if not query:
        print("Please enter a valid topic.")
        return

    document_type = input("Enter document type (proposal/paper/none): ").strip().lower()
    if document_type == "none":
        document_type = None

    state = ResearchState(
        user_query=query,
        document_type=document_type,
    )

    bib_path = None
    downloaded_pdfs = {}
    index_path = None
    metadata_path = None
    review_results = []

    # Step 1: extract keywords
    state.extracted_keywords = extract_keywords(state.user_query)

    # Step 2: determine mode
    state.mode = route_mode(state)

    # TEMP TEST ONLY:
    # Uncomment this if you want to force review mode manually
    # state.mode = Mode.REVIEW

    # Step 3: discovery / hybrid behavior
    if state.mode.value in {"discovery", "hybrid"}:
        ss_papers = discover_papers_from_semantic_scholar(state.extracted_keywords)
        arxiv_papers = discover_papers_from_arxiv(state.extracted_keywords)

        # Merge both sources — deduplicate by arXiv ID and title
        seen_arxiv_ids: set[str] = set()
        seen_titles: set[str] = set()
        merged: list = []

        for paper in ss_papers + arxiv_papers:
            # Deduplicate by arXiv ID if available
            if paper.arxiv_id:
                if paper.arxiv_id in seen_arxiv_ids:
                    continue
                seen_arxiv_ids.add(paper.arxiv_id)

            # Deduplicate by normalised title
            norm_title = paper.title.lower().strip()
            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)

            merged.append(paper)

        state.discovered_papers = merged
        print(f"[Discovery] SS: {len(ss_papers)}, arXiv: {len(arxiv_papers)}, merged: {len(merged)}")

        # Step 4: filter and rank results
        state.discovered_papers = filter_and_rank_papers(
            state.discovered_papers,
            state.extracted_keywords,
        )

        # Step 5: build citations
        state.citation_metadata = build_citations(state.discovered_papers)

        # Step 6: deduplicate citations
        state.citation_metadata = deduplicate_citations(state.citation_metadata)

        # Step 7: save BibTeX file
        bib_path = save_bibtex_file(state.citation_metadata, state.user_query)

        # Step 8: download PDFs
        downloaded_pdfs = download_pdfs(state.discovered_papers)

        # Step 9: extract text chunks from PDFs
        state.evidence_chunks = build_evidence_chunks(downloaded_pdfs)

        # Step 9b: fallback to abstracts if no PDFs were downloaded
        if not state.evidence_chunks:
            print("[Fallback] No PDF chunks available — indexing abstracts instead.")
            from services.pdf_text_extractor import build_abstract_chunks
            state.evidence_chunks = build_abstract_chunks(state.discovered_papers)
            print(f"[Fallback] Built {len(state.evidence_chunks)} abstract chunks.")
        # Step 10: build and save FAISS index
        if state.evidence_chunks:
            index, metadata = build_faiss_index(state.evidence_chunks)
            if index is not None:
                index_path, metadata_path = save_faiss_index(index, metadata)

    # Step 11: review-mode local retrieval
    if state.mode.value == "review":
        review_results = search_local_index(state.user_query, top_k=5)

    print(f"\nSelected mode: {state.mode.value}")

    print("\nExtracted keywords:")
    print(state.extracted_keywords)

    print(f"\nNumber of discovered papers: {len(state.discovered_papers)}")

    if state.discovered_papers:
        print("\nTop discovered papers:")
        for i, paper in enumerate(state.discovered_papers[:5], start=1):
            print(f"{i}. {paper.title} ({paper.year}) - {paper.source}")

    if state.citation_metadata:
        print("\nGenerated citations:")
        for c in state.citation_metadata[:5]:
            print(f"\n{c.citation_key}")
            print(c.bibtex)

    if bib_path:
        print(f"\nBibTeX file saved at: {bib_path}")

    if downloaded_pdfs:
        print("\nDownloaded PDFs:")
        for title, path in downloaded_pdfs.items():
            print(f"- {title}")
            print(f"  saved to: {path}")
    else:
        print("\nNo PDFs were downloaded.")

    if state.evidence_chunks:
        print(f"\nNumber of evidence chunks: {len(state.evidence_chunks)}")

        print("\nSample evidence chunks:")
        for chunk in state.evidence_chunks[:3]:
            print(f"\nPaper: {chunk['paper_title']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print("Chunk Text:", chunk["chunk_text"][:300] + "...")
    else:
        print("\nNo evidence chunks were extracted.")

    if index_path and metadata_path:
        print("\nFAISS index saved:")
        print(f"- Index file: {index_path}")
        print(f"- Metadata file: {metadata_path}")
    else:
        print("\nNo FAISS index was created.")

    if review_results:
        print("\nTop local retrieval results:")
        for item in review_results:
            print(f"\nRank: {item['rank']}")
            print(f"Paper: {item['paper_title']}")
            print(f"Chunk ID: {item['chunk_id']}")
            print(f"Distance: {item['distance']:.4f}")
            print(item["chunk_text"][:350] + "...")
    elif state.mode.value == "review":
        print("\nNo local retrieval results found.")


if __name__ == "__main__":
    main()