"""
Phase 1 — BM25 + Embedding search with RRF fusion
===================================================
Scores every source file in the repo against query keywords derived from the
ticket and MR diff, then re-ranks with semantic embeddings via Chroma.

Two-signal retrieval pipeline
------------------------------
1. BM25 lexical ranking  → top-20 files  (rank_bm25)
2. Embedding ANN search  → top-20 files  (rank_embed)   ← NEW
3. Reciprocal Rank Fusion of both signals → top-10 final ← NEW

Fallback: if embeddings / ChromaDB are unavailable, Phase 1 silently
returns the BM25-only top-10 (original behaviour).

No LLM.  No GPU required.  Fully deterministic.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .constants import SKIP_DIRS, SUPPORTED_EXTENSIONS
from .embedding_indexer import get_indexer

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# Nombre maximum de fichiers passes au BM25 (protection RAM sur grands repos).
MAX_FILES = 5000

# Basic stop words — avoids polluting BM25 query with noise tokens.
STOP_WORDS = {
    "the", "a", "an", "is", "in", "it", "of", "to", "and", "or",
    "not", "for", "with", "this", "that", "are", "was", "be", "as",
    "at", "by", "from", "on", "but", "if", "then", "else", "return",
    "def", "class", "import", "self", "true", "false", "null", "none",
    "new", "var", "let", "const", "public", "private", "static", "void",
    "int", "str", "bool", "float", "list", "dict", "type", "get", "set",
}

# Score multiplier for files that appear in the MR diff.
MR_FILE_BOOST = 2.0

# BM25 retrieval width before fusion.
BM25_RETRIEVAL_TOP  = 20
EMBED_RETRIEVAL_TOP = 20

# Final output after fusion.
FUSION_TOP = 10

# RRF constant (standard value).
RRF_K = 60

# ── Helpers ────────────────────────────────────────────────────────────────────


def _git_recent_files(repo_path: str, n: int = 2000) -> List[str]:
    """
    Retourne les n fichiers modifies le plus recemment via git log.
    Fallback silencieux si git indisponible.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "--name-only",
             "--pretty=format:", "-500"],
            capture_output=True, text=True, timeout=10
        )
        seen: List[str] = []
        visited: set = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            abs_path = os.path.join(repo_path, line)
            if abs_path not in visited and os.path.isfile(abs_path):
                visited.add(abs_path)
                seen.append(abs_path)
            if len(seen) >= n:
                break
        return seen
    except Exception:
        return []


def collect_repo_files(
    repo_path: str,
    component: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Walk the repo and return (absolute_paths, file_contents) for source files.

    Optimisations pour grands repos :
    1. Si `component` est renseigne, filtre d'abord par dossier correspondant.
    2. Priorise les fichiers recemment modifies (git log).
    3. Plafonne a MAX_FILES fichiers pour proteger la RAM.
    """
    all_files: List[str] = []

    for root, dirs, file_names in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        # Filtre par composant si precise dans le ticket (ex: "auth/", "api")
        if component:
            rel_root = os.path.relpath(root, repo_path).replace("\\", "/")
            if not rel_root.startswith(component.strip("/")):
                # Garder quand meme les sous-dossiers directs de la racine
                if root != repo_path:
                    dirs[:] = [
                        d for d in dirs
                        if component.strip("/").split("/")[0] in d
                    ]
                    continue

        for fname in file_names:
            if Path(fname).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            all_files.append(os.path.join(root, fname))

    # Prioriser les fichiers recemment touches si le repo est grand
    if len(all_files) > MAX_FILES:
        recent = set(_git_recent_files(repo_path, n=MAX_FILES // 2))
        # Mettre les fichiers recents en premier, completer avec le reste
        prioritized = [f for f in all_files if f in recent]
        rest        = [f for f in all_files if f not in recent]
        all_files   = (prioritized + rest)[:MAX_FILES]

    files: List[str] = []
    contents: List[str] = []
    for fpath in all_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                contents.append(fh.read())
            files.append(fpath)
        except (IOError, OSError):
            pass

    return files, contents


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, strip stop words and short tokens."""
    raw = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
    return [
        t.lower()
        for t in raw
        if len(t) > 2 and t.lower() not in STOP_WORDS
    ]


def extract_keywords(ticket: dict, mr_diff: str) -> List[str]:
    """
    Build BM25 query tokens from:
      - ticket title + description + component + labels
      - identifiers appearing in +/- lines of the diff
    """
    text_parts = [
        ticket.get("title", ""),
        ticket.get("description", ""),
        ticket.get("component", ""),
        " ".join(ticket.get("labels", [])),
    ]
    tokens = _tokenize(" ".join(text_parts))

    # Pull identifiers from changed lines only (not context lines).
    for line in mr_diff.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            tokens.extend(_tokenize(line[1:]))

    # Deduplicate, preserving first-seen order.
    seen: set = set()
    unique: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def _parse_mr_file_paths(mr_diff: str) -> set:
    """Extract relative file paths from diff headers (+++ b/... or --- a/...)."""
    paths: set = set()
    for line in mr_diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                paths.add(path.lstrip("/"))
    return paths


def _file_in_diff(fpath: str, mr_file_paths: set) -> bool:
    """True when the repo file path ends with one of the diff-mentioned paths."""
    normalized = fpath.replace("\\", "/")
    return any(normalized.endswith(p) or p in normalized for p in mr_file_paths)


# ── RRF helpers ────────────────────────────────────────────────────────────────


def _derive_repo_id(repo_path: str) -> str:
    """
    Derive a stable, Chroma-safe collection identifier from the repo path.
    Uses the sanitised basename (e.g. '/tmp/repo_42' → 'repo_42').
    The EmbeddingIndexer uses git HEAD internally for cache invalidation.
    """
    basename = os.path.basename(repo_path.rstrip("/\\")) or "repo"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", basename)[:60]


def _rrf_fusion(
    bm25_ranked:  List[Dict],
    embed_ranked: List[Dict],
    k:     int = RRF_K,
    top_n: int = FUSION_TOP,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion of two ranked file lists.

    RRF score(f) = Σ_i  1 / (k + rank_i(f))

    A file absent from ranking i receives rank = len(ranking_i) + 1
    (penalty for missing signal).

    Args:
        bm25_ranked  : [{file, score}, …]  sorted desc by BM25 score
        embed_ranked : [{file, embed_score}, …]  sorted desc by embed score
        k            : smoothing constant (default 60)
        top_n        : number of results to return

    Returns:
        [{file, rrf_score, bm25_rank, embed_rank}, …]  top-n, sorted desc
    """
    # Build rank maps (1-based).
    bm25_rank  = {item["file"]: i + 1 for i, item in enumerate(bm25_ranked)}
    embed_rank = {item["file"]: i + 1 for i, item in enumerate(embed_ranked)}

    # Default rank for a file absent from a list.
    default_bm25  = len(bm25_ranked)  + 1
    default_embed = len(embed_ranked) + 1

    # Union of all candidate files.
    all_files = set(bm25_rank) | set(embed_rank)

    scored: List[Tuple[float, str, int, int]] = []
    for f in all_files:
        r_bm25  = bm25_rank.get(f,  default_bm25)
        r_embed = embed_rank.get(f, default_embed)
        rrf     = 1.0 / (k + r_bm25) + 1.0 / (k + r_embed)
        scored.append((rrf, f, r_bm25, r_embed))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "file":       f,
            "rrf_score":  rrf,
            "bm25_rank":  r_bm25,
            "embed_rank": r_embed,
        }
        for rrf, f, r_bm25, r_embed in scored[:top_n]
    ]


# ── Phase 0 file list reuse ────────────────────────────────────────────────────


def _files_from_structure(struct_files: list, repo_path: str) -> Tuple[List[str], List[str]]:
    """
    Construit (absolute_paths, contents) depuis la liste de Phase 0.
    Evite le double rglob quand project_structure est disponible.
    """
    files: List[str] = []
    contents: List[str] = []
    for entry in struct_files:
        rel = entry.get("path", "")
        if not rel:
            continue
        abs_path = os.path.join(repo_path, rel.replace("/", os.sep))
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                contents.append(fh.read())
            files.append(abs_path)
        except (IOError, OSError):
            pass
    return files, contents


# ── LangGraph node ─────────────────────────────────────────────────────────────


def phase_bm25(state: dict) -> dict:
    """
    LangGraph node — Phase 1.

    Reads:  ticket, mr_diff, repo_path, project_structure (Phase 0, optionnel)
    Writes: keywords, bm25_files, rrf_scores

    Optimisation : si project_structure["files"] existe (Phase 0 a tourné),
    la liste de fichiers est réutilisée sans re-scanner le repo.
    """
    ticket    = state["ticket"]
    mr_diff   = state["mr_diff"]
    repo_path = state["repo_path"]

    # Réutiliser la liste de Phase 0 si disponible (évite le double scan rglob).
    struct_files = state.get("project_structure", {}).get("files", [])
    if struct_files:
        logger.info("[phase1] Réutilisation de %d fichiers depuis Phase 0 (skip rglob).", len(struct_files))
        files, contents = _files_from_structure(struct_files, repo_path)
    else:
        component = ticket.get("component", "")
        files, contents = collect_repo_files(repo_path, component=component)

    if not files:
        return {**state, "keywords": [], "bm25_files": [], "rrf_scores": []}

    # ── BM25 ranking ───────────────────────────────────────────────────────────
    tokenized_corpus = [_tokenize(c) for c in contents]
    bm25 = BM25Okapi(tokenized_corpus)

    keywords = extract_keywords(ticket, mr_diff) or ["bug", "error", "exception"]
    scores   = bm25.get_scores(keywords).tolist()

    # Boost files that the MR diff directly touches.
    mr_paths = _parse_mr_file_paths(mr_diff)
    for i, fpath in enumerate(files):
        if _file_in_diff(fpath, mr_paths):
            scores[i] *= MR_FILE_BOOST

    # Top-20 for RRF input (wider than the old top-10).
    bm25_ranked: List[Dict] = sorted(
        [{"file": files[i], "score": scores[i]} for i in range(len(files))],
        key=lambda x: x["score"],
        reverse=True,
    )[:BM25_RETRIEVAL_TOP]

    # ── Embedding search + RRF ─────────────────────────────────────────────────
    rrf_debug: List[Dict] = []
    final_files: List[Dict] = []

    try:
        query     = " ".join(keywords)
        indexer   = get_indexer()
        repo_id   = _derive_repo_id(repo_path)

        # Build / refresh the persistent index (no-op if commit unchanged).
        indexer.index_repo(repo_path, repo_id)

        embed_ranked: List[Dict] = indexer.search(query, repo_id, top_k=EMBED_RETRIEVAL_TOP)

        if embed_ranked:
            fused = _rrf_fusion(bm25_ranked, embed_ranked, top_n=FUSION_TOP)

            # Build bm25_files in the format expected by Phase 2:
            # [{"file": str, "score": float}, …]
            # Use the original BM25 score for traceability; Phase 2 only
            # reads the "file" key, so any float is fine.
            bm25_score_map = {item["file"]: item["score"] for item in bm25_ranked}
            final_files = [
                {"file": item["file"], "score": bm25_score_map.get(item["file"], 0.0)}
                for item in fused
            ]

            # Debug / traceability record stored in rrf_scores.
            rrf_debug = [
                {
                    "file":       item["file"],
                    "rrf_score":  item["rrf_score"],
                    "bm25_rank":  item["bm25_rank"],
                    "embed_rank": item["embed_rank"],
                }
                for item in fused
            ]

            logger.info(
                f"[phase1] RRF fusion: BM25 top-{len(bm25_ranked)} ⊕ "
                f"Embed top-{len(embed_ranked)} → {len(final_files)} files"
            )
        else:
            # Embed search returned nothing (e.g. empty index) — BM25 fallback.
            logger.warning("[phase1] Embed search returned 0 results — using BM25 top-10.")
            final_files = bm25_ranked[:FUSION_TOP]

    except Exception as exc:
        logger.warning(
            f"[phase1] Embedding+RRF failed ({exc}) — falling back to BM25 only."
        )
        final_files = bm25_ranked[:FUSION_TOP]

    return {
        **state,
        "keywords":   keywords,
        "bm25_files": final_files,
        "rrf_scores": rrf_debug,
    }
