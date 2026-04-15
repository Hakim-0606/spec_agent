"""
Embedding Indexer — ChromaDB persistent index for semantic file search.

Provides:
    - index_repo(repo_path, repo_id)  → build / update persistent Chroma index
    - search(query, repo_id, top_k)   → top-k files by semantic similarity

Chunking  : 50 lines per chunk, 10 lines overlap.
Model     : all-MiniLM-L6-v2 (singleton, loaded once at first use).
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
EMBED_BATCH   = 32   # sentences per encode() call
UPSERT_BATCH  = 512  # max chunks per Chroma add() call

MODEL_ID    = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = os.environ.get(
    "CHROMA_PERSIST_PATH",
    str(Path.home() / ".cache" / "spec_agent" / "chroma"),
)

# ── Singleton embedding model ──────────────────────────────────────────────────

_embedding_model = None


class _CodeT5pEncoder:
    """
    Thin wrapper around Salesforce/codet5p-110m-embedding loaded via transformers.
    Exposes the same .encode(texts, batch_size, show_progress_bar) interface as
    SentenceTransformer so the rest of the code is unchanged.
    codet5p cannot be loaded through sentence-transformers because its custom config
    conflicts with the SentenceTransformer wrapper (missing 'is_decoder' attribute).
    """

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"[EmbeddingIndexer] Loading codet5p via transformers: '{model_id}'…")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model     = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self._model.eval()
        self._torch = torch
        logger.info("[EmbeddingIndexer] codet5p model ready.")

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **_,
    ):
        import numpy as np
        torch = self._torch
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            # codet5p returns the embedding at position 0 (pooled representation)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


def _get_model():
    """Load the embedding model once (module-level singleton).

    Uses _CodeT5pEncoder for Salesforce/codet5p-110m-embedding (transformers direct),
    and SentenceTransformer for all other models.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"[EmbeddingIndexer] Loading model '{MODEL_ID}'…")
        if "codet5p" in MODEL_ID.lower():
            _embedding_model = _CodeT5pEncoder(MODEL_ID)
        else:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(MODEL_ID)
            logger.info("[EmbeddingIndexer] Model ready.")
    return _embedding_model


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
        Returns [{file, embed_score}, …] deduplicated by file, best-score kept.
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
        # keep under 63 chars
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
            logger.warning(f"[EmbeddingIndexer] Could not persist commit hash: {exc}")

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
                f"[EmbeddingIndexer] Index up to date "
                f"(commit {current_commit[:8]}) — skipping."
            )
            return False

        logger.info(
            f"[EmbeddingIndexer] Indexing repo '{repo_id}' "
            f"@ {current_commit[:8] or 'unknown'}…"
        )

        try:
            model  = _get_model()
            client = self._get_client()
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
                batch = all_chunks[start : start + UPSERT_BATCH]
                texts = [c["text"] for c in batch]
                embeddings = model.encode(
                    texts, batch_size=EMBED_BATCH, show_progress_bar=False
                )
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
                    f"[EmbeddingIndexer] {start + len(batch)}/{len(all_chunks)} chunks indexed"
                )

            self._store_commit(repo_id, current_commit)
            logger.info(
                f"[EmbeddingIndexer] Done — {len(all_chunks)} chunks stored for '{repo_id}'."
            )
            return True

        except Exception as exc:
            logger.warning(f"[EmbeddingIndexer] Indexing failed: {exc}")
            return False

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, repo_id: str, top_k: int = 20) -> List[Dict]:
        """
        Semantic search over the indexed repo.

        Returns [{file: str, embed_score: float}, …] deduplicated by file
        (best score per file kept), sorted descending.
        Falls back to [] if the index is unavailable.
        """
        try:
            model    = _get_model()
            client   = self._get_client()
            col_name = self._col_name(repo_id)

            existing = [c.name for c in client.list_collections()]
            if col_name not in existing:
                logger.warning(
                    f"[EmbeddingIndexer] Collection '{col_name}' not found "
                    "— call index_repo first."
                )
                return []

            collection = client.get_collection(col_name)
            total      = collection.count()
            if total == 0:
                return []

            # Over-fetch to allow per-file deduplication.
            n_results      = min(top_k * 5, total)
            query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"],
            )

            # Deduplicate: keep best (lowest cosine distance) score per file.
            best: Dict[str, float] = {}
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                fpath = meta.get("file", "")
                score = float(1.0 - dist / 2.0)   # distance ∈ [0,2] → similarity ∈ [0,1]
                if fpath not in best or score > best[fpath]:
                    best[fpath] = score

            ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
            return [{"file": f, "embed_score": s} for f, s in ranked[:top_k]]

        except Exception as exc:
            logger.warning(f"[EmbeddingIndexer] Search failed: {exc}")
            return []


# ── Module-level singleton ─────────────────────────────────────────────────────

_indexer: Optional[EmbeddingIndexer] = None


def get_indexer() -> EmbeddingIndexer:
    """Return the shared EmbeddingIndexer instance (created once per process)."""
    global _indexer
    if _indexer is None:
        _indexer = EmbeddingIndexer()
    return _indexer
