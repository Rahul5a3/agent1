from __future__ import annotations

from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import INDEX_DIR


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_faiss_index(evidence_chunks: list[dict]) -> tuple[faiss.IndexFlatL2 | None, list[dict]]:
    """
    Build a FAISS index from evidence chunks.
    Returns:
        - FAISS index
        - metadata list aligned with vector positions
    """
    if not evidence_chunks:
        return None, []

    texts = [chunk["chunk_text"] for chunk in evidence_chunks]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    metadata = []
    for idx, chunk in enumerate(evidence_chunks):
        metadata.append(
            {
                "faiss_id": idx,
                "paper_title": chunk["paper_title"],
                "pdf_path": chunk["pdf_path"],
                "chunk_id": chunk["chunk_id"],
                "chunk_text": chunk["chunk_text"],
            }
        )

    return index, metadata


def save_faiss_index(index: faiss.IndexFlatL2, metadata: list[dict], index_name: str = "agent1_index") -> tuple[Path, Path]:
    """
    Save FAISS index and metadata to disk.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / f"{index_name}.faiss"
    metadata_path = INDEX_DIR / f"{index_name}_metadata.json"

    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return index_path, metadata_path