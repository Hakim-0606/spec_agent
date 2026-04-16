"""
Embedding Indexer — ChromaDB persistent index for semantic file search.

Provides:
    - index_repo(repo_path, repo_id)  → build / update persistent Chroma index
    - search(query, repo_id, top_k)   → top-k files by semantic similarity

Chunking  : 50 lines per chunk, 10 lines overlap.
Model     : BAAI/bge-small-en-v1.5 via fastembed (ONNX — no PyTorch required).
Cache     : re-indexing is skipped when git HEAD commit hash is unchanged.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from .constants import SKIP_DIRS, SUPPORTED_EXTENSIONS

# ── Constants ──────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 50   # lines per chunk
CHUNK_OVERLAP = 10   # overlap between consecutive chunks
UPSERT_BATCH  = 512  # max chunks per Chroma add() call

# fastembed model — overridable via env var.
# BAAI/bge-small-en-v1.5 : 33 MB, 384 dims, better quality than all-MiniLM-L6-v2.
# Other supported options : "sentence-transformers/all-MiniLM-L6-v2" (23 MB)
MODEL_ID    = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_PATH = os.environ.get(
    "CHROMA_PERSIST_PATH",
    str(Path.home() / ".cache" / "spec_agent" / "chroma"),
)

# ── Singleton embedding model ──────────────────────────────────────────────────

_embedding_model = None


def _get_model():
    """
    Load the fastembed TextEmbedding model once (module-level singleton).
    fastembed uses ONNX Runtime — no PyTorch dependency.
    """
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding
        logger.info("[EmbeddingIndexer] Loading fastembed model '%s'...", MODEL_ID)
        _embedding_model = TextEmbedding(model_name=MODEL_ID)
        logger.info("[EmbeddingIndexer] Model ready.")
    return _embedding_model


def _embed(texts: List[str]) -> List:
    """Embed a list of texts. Returns a list of numpy arrays."""
    model = _get_model()
    return list(model.embed(texts))


# ── Git helpers ────────────────────────────────────────────────────────────────

def _git_commit_hash(repo_path: str) -> str:
    """Return the current HEAD commit hash, or '' on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


# ── Line-based chunker ─────────────────────────────────────────────────────────

def _chunk_file(fpath: str) -> List[Dict]:
    """
    Split a source file into overlapping line-based chunks.

    Returns a list of:
        {"file": str, "text": str, "chunk_id": str, "start_line": int}
    """
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except (IOError, OSError):
        return []

    chunks: List[Dict] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    i = 0
    while i < len(lines):
        chunk_lines = lines[i : i + CHUNK_SIZE]
        text = "".join(chunk_lines).strip()
        if text:
            chunks.append({
                "file":       fpath,
                "text":       text,
                "chunk_id":   f"{fpath}::chunk_{i}",
                "start_line": i + 1,
            })
        i += step

    return chunks


# ── EmbeddingIndexer ───────────────────────────────────────────────────────────

class EmbeddingIndexer:
    """
    Manages a ChromaDB persistent index of source-file chunks.

    Public API
    ----------
    index_repo(repo_path, repo_id)
        Build or update the Chroma collection for a repository.
        No-op when git HEAD is unchanged since last indexing.

    search(query, repo_id, top_k)
        Semantic nearest-neighbour search.
        Returns [{file, embed_score}, ...] deduplicated by file, best-score kept.
    """

    def __init__(self, persist_path: str = CHROMA_PATH):
        self.persist_path = persist_path
        self._client = None

    # ── Chroma client (lazy) ───────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_path)
        return self._client

    # ── Collection name helpers ────────────────────────────────────────────────

    @staticmethod
    def _sanitise(name: str) -> str:
        """Produce a valid Chroma collection name (3-63 chars, alphanumeric/_/-)."""
        sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:63]
        return sanitised if len(sanitised) >= 3 else sanitised + "_ix"

    def _col_name(self, repo_id: str) -> str:
        return self._sanitise(repo_id)

    def _meta_col_name(self, repo_id: str) -> str:
        return self._sanitise(repo_id)[:57] + "_meta"

    # ── Commit-hash cache ──────────────────────────────────────────────────────

    def _get_indexed_commit(self, repo_id: str) -> str:
        try:
            client   = self._get_client()
            meta_col = self._meta_col_name(repo_id)
            names    = [c.name for c in client.list_collections()]
            if meta_col not in names:
                return ""
            col     = client.get_collection(meta_col)
            results = col.get(ids=["commit_hash"])
            docs    = results.get("documents") or []
            return docs[0] if docs else ""
        except Exception:
            return ""

    def _store_commit(self, repo_id: str, commit_hash: str) -> None:
        try:
            client   = self._get_client()
            meta_col = self._meta_col_name(repo_id)
            names    = [c.name for c in client.list_collections()]
            if meta_col in names:
                col = client.get_collection(meta_col)
            else:
                col = client.create_collection(meta_col)
            col.upsert(ids=["commit_hash"], documents=[commit_hash])
        except Exception as exc:
            logger.warning("[EmbeddingIndexer] Could not persist commit hash: %s", exc)

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index_repo(self, repo_path: str, repo_id: str) -> bool:
        """
        Build or refresh the Chroma index for *repo_path*.

        Returns True when indexing was performed, False when skipped or on error.
        """
        current_commit = _git_commit_hash(repo_path)
        stored_commit  = self._get_indexed_commit(repo_id)

        if current_commit and current_commit == stored_commit:
            logger.info(
                "[EmbeddingIndexer] Index up to date (commit %s) — skipping.",
                current_commit[:8],
            )
            return False

        logger.info(
            "[EmbeddingIndexer] Indexing repo '%s' @ %s...",
            repo_id, current_commit[:8] or "unknown",
        )

        try:
            client   = self._get_client()
            col_name = self._col_name(repo_id)

            # Full re-index: drop stale collection if present.
            existing = [c.name for c in client.list_collections()]
            if col_name in existing:
                client.delete_collection(col_name)
            collection = client.create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Walk repo, collect chunks.
            all_chunks: List[Dict] = []
            for root, dirs, fnames in os.walk(repo_path):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fname in fnames:
                    if Path(fname).suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    all_chunks.extend(_chunk_file(os.path.join(root, fname)))

            if not all_chunks:
                logger.warning("[EmbeddingIndexer] No source chunks found — index is empty.")
                return False

            # Batch-embed and upsert.
            for start in range(0, len(all_chunks), UPSERT_BATCH):
                batch      = all_chunks[start : start + UPSERT_BATCH]
                texts      = [c["text"] for c in batch]
                embeddings = _embed(texts)
                collection.add(
                    ids        = [c["chunk_id"]   for c in batch],
                    documents  = texts,
                    embeddings = [e.tolist() for e in embeddings],
                    metadatas  = [
                        {"file": c["file"], "start_line": c["start_line"]}
                        for c in batch
                    ],
                )
                logger.debug(
                    "[EmbeddingIndexer] %d/%d chunks indexed",
                    start + len(batch), len(all_chunks),
                )

            self._store_commit(repo_id, current_commit)
            logger.info(
                "[EmbeddingIndexer] Done — %d chunks stored for '%s'.",
                len(all_chunks), repo_id,
            )
            return True

        except Exception as exc:
            logger.warning("[EmbeddingIndexer] Indexing failed: %s", exc)
            return False

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, repo_id: str, top_k: int = 20) -> List[Dict]:
        """
        Semantic search over the indexed repo.

        Returns [{file: str, embed_score: float}, ...] deduplicated by file
        (best score per file kept), sorted descending.
        Falls back to [] if the index is unavailable.
        """
        try:
            client   = self._get_client()
            col_name = self._col_name(repo_id)

            existing = [c.name for c in client.list_collections()]
            if col_name not in existing:
                logger.warning(
                    "[EmbeddingIndexer] Collection '%s' not found — call index_repo first.",
                    col_name,
                )
                return []

            collection = client.get_collection(col_name)
            total      = collection.count()
            if total == 0:
                return []

            # Over-fetch to allow per-file deduplication.
            n_results       = min(top_k * 5, total)
            query_embedding = _embed([query])[0].tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"],
            )

            # Deduplicate: keep best (lowest cosine distance) score per file.
            best: Dict[str, float] = {}
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                fpath = meta.get("file", "")
                score = float(1.0 - dist / 2.0)   # distance in [0,2] -> similarity in [0,1]
                if fpath not in best or score > best[fpath]:
                    best[fpath] = score

            ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
            return [{"file": f, "embed_score": s} for f, s in ranked[:top_k]]

        except Exception as exc:
            logger.warning("[EmbeddingIndexer] Search failed: %s", exc)
            return []


# ── Module-level singleton ─────────────────────────────────────────────────────

_indexer: Optional[EmbeddingIndexer] = None


def get_indexer() -> EmbeddingIndexer:
    """Return the shared EmbeddingIndexer instance (created once per process)."""
    global _indexer
    if _indexer is None:
        _indexer = EmbeddingIndexer()
    return _indexer
