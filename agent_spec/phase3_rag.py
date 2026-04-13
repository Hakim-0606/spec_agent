"""
Phase 3 — RAG with cAST Chunking
==================================
Embeds the top-5 function bodies (cAST chunks — one chunk = one function, never
split mid-function) using Salesforce/codet5p-110m-embedding, stores them in an
ephemeral Chroma collection, then returns the top-3 most semantically relevant
contexts for the Phase-4 LLM.

No LLM.  GPU optional (CPU fallback available).
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded heavy dependencies so import errors are isolated.
_embedding_model = None
# Override with EMBEDDING_MODEL env var.
# Default: all-MiniLM-L6-v2 (fast, already widely cached).
# For production code quality: Salesforce/codet5p-110m-embedding (requires trust_remote_code=True).
import os as _os
_MODEL_ID = _os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _get_embedding_model():
    """Load the embedding model once (lazy singleton).

    Uses transformers direct loading for codet5p (sentence-transformers incompatible),
    SentenceTransformer for everything else.
    """
    global _embedding_model
    if _embedding_model is None:
        if "codet5p" in _MODEL_ID.lower():
            # Import the shared encoder from embedding_indexer to avoid duplication.
            from .embedding_indexer import _CodeT5pEncoder
            _embedding_model = _CodeT5pEncoder(_MODEL_ID)
        else:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(_MODEL_ID)
    return _embedding_model


# ── Query builder ──────────────────────────────────────────────────────────────


def _build_rag_query(ticket: dict, mr_diff: str) -> str:
    """
    Build a semantic query string that captures bug intent beyond exact keywords.
    Combines ticket description + diff context lines + severity.
    """
    parts = [
        ticket.get("title", ""),
        ticket.get("description", ""),
    ]

    # Include a few lines of diff context (--- / +++ lines and ±3 context).
    diff_lines: List[str] = []
    for line in mr_diff.splitlines():
        if line.startswith(("+", "-", "@")):
            diff_lines.append(line)
        if len(diff_lines) >= 20:
            break
    if diff_lines:
        parts.append("\n".join(diff_lines))

    severity = ticket.get("severity", "")
    if severity:
        parts.append(f"severity:{severity}")

    return " ".join(p for p in parts if p).strip()


# ── cAST chunking ──────────────────────────────────────────────────────────────


def _make_chunk_document(fn: dict) -> str:
    """
    Build a rich text document for a cAST chunk.
    Includes metadata header + full function source so embeddings carry context.
    """
    file_info = fn.get("file", "")
    class_info = fn.get("class") or ""
    func_info = fn.get("function", "")
    lang = fn.get("language", "")
    callers = ", ".join(fn.get("callers") or [])
    callees = ", ".join(fn.get("callees") or [])
    source = fn.get("source", "")

    header = f"# file:{file_info} class:{class_info} function:{func_info} lang:{lang}"
    meta = f"# callers:[{callers}] callees:[{callees}]"
    return f"{header}\n{meta}\n{source}"


# ── Chroma-based semantic search ───────────────────────────────────────────────


def _semantic_rerank(
    functions: List[dict],
    query: str,
    top_k: int = 3,
) -> List[dict]:
    """
    Embed all function chunks + query with codet5p, store in an ephemeral Chroma
    collection, retrieve top-k by cosine similarity.

    Falls back to BM25-style keyword overlap if embeddings are unavailable.
    """
    try:
        import chromadb

        model = _get_embedding_model()
        documents = [_make_chunk_document(fn) for fn in functions]

        # Embed all chunks + the query in one batch for efficiency.
        all_texts = documents + [query]
        embeddings = model.encode(all_texts, batch_size=8, show_progress_bar=False)

        chunk_embeddings = [emb.tolist() for emb in embeddings[:-1]]
        query_embedding = embeddings[-1].tolist()

        # EphemeralClient crée une instance totalement isolée par appel —
        # évite les conflits de tenant quand plusieurs requêtes tournent en parallèle.
        client = chromadb.EphemeralClient()
        collection_name = "spec_rag_temp"
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [f"fn_{i}" for i in range(len(functions))]
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=chunk_embeddings,
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(functions)),
            include=["distances"],
        )

        # Map result ids back to function dicts, annotate with similarity score.
        id_to_fn = {f"fn_{i}": functions[i] for i in range(len(functions))}
        ranked: List[dict] = []
        result_ids = results["ids"][0]
        result_distances = results["distances"][0]
        for rid, dist in zip(result_ids, result_distances):
            fn = dict(id_to_fn[rid])
            # Chroma cosine distance is in [0, 2]; convert to similarity [0, 1].
            fn["semantic_score"] = float(1.0 - dist / 2.0)
            ranked.append(fn)

        return ranked

    except Exception as exc:
        # Graceful fallback: return top-k by existing score if embeddings fail.
        logger.warning("[phase3] Embedding/Chroma unavailable (%s) — using score fallback.", exc)
        sorted_fns = sorted(functions, key=lambda f: f.get("score", 0.0), reverse=True)
        for fn in sorted_fns:
            fn.setdefault("semantic_score", fn.get("score", 0.0))
        return sorted_fns[:top_k]


# ── LangGraph node ─────────────────────────────────────────────────────────────


def phase_rag(state: dict) -> dict:
    """
    LangGraph node — Phase 3.

    Reads:  ast_functions (top-5), ticket, mr_diff
    Writes: rag_contexts (top-3)
    """
    functions: List[dict] = state.get("ast_functions", [])
    ticket: dict = state.get("ticket", {})
    mr_diff: str = state.get("mr_diff", "")

    if not functions:
        return {**state, "rag_contexts": []}

    query = _build_rag_query(ticket, mr_diff)
    top3 = _semantic_rerank(functions, query, top_k=3)

    return {**state, "rag_contexts": top3}
