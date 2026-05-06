from __future__ import annotations

from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import INDEX_DIR


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_faiss_index(index_name: str = "agent1_index") -> tuple[faiss.Index | None, list[dict]]:
    """
    Load FAISS index and aligned metadata from disk.
    """
    index_path = INDEX_DIR / f"{index_name}.faiss"
    metadata_path = INDEX_DIR / f"{index_name}_metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        return None, []

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search_local_index(
    query: str,
    top_k: int = 5,
    index_name: str = "agent1_index",
    max_distance: float = 1.5,
) -> list[dict]:
    index, metadata = load_faiss_index(index_name=index_name)
    if index is None or not metadata:
        return []

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    results: list[dict] = []

    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0 or idx >= len(metadata):
            continue

        distance = float(distances[0][rank - 1])

        # Reject chunks that are too far from the query
        # L2 distance > 1.5 means the content is likely from a
        # completely different topic (e.g. old index from previous run)
        if distance > max_distance:
            print(f"[LocalRetriever] Skipping chunk (distance {distance:.4f} > threshold {max_distance})")
            continue

        item = metadata[idx].copy()
        item["rank"] = rank
        item["distance"] = distance
        results.append(item)

    if not results:
        print(f"[LocalRetriever] No chunks passed distance threshold {max_distance}. Index may be stale.")

    return results