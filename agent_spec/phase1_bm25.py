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

# Score multiplier for files explicitly named in a stack trace.
# Stack trace signal is the strongest possible localization hint — ×5 puts
# those files at the top regardless of BM25 lexical score.
TRACE_FILE_BOOST = 5.0

# Score multiplier for files derived from an ImportError / ModuleNotFoundError
# module path in the ticket title.  Stronger than trace (×10) because the
# module path is an exact, unambiguous reference to the missing/broken file.
IMPORT_ERROR_BOOST = 10.0

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


# ── Stack trace parsing ────────────────────────────────────────────────────────


def _parse_stack_trace(error_trace: str) -> Dict:
    """
    Parse a stack trace string and return:
        file_paths     : list of relative file paths mentioned (forward slashes, deduped)
        exception_type : exception class name (e.g. 'KeyError', 'NullPointerException')
        function_names : list of function/method names from the trace frames
        line_numbers   : {file_path: last_seen_line_number}

    Supports Python, JavaScript/TypeScript, Java, Go, and generic patterns.
    The function_names list is ordered from innermost (crash site) to outermost (entry).
    """
    if not error_trace:
        return {"file_paths": [], "exception_type": "", "function_names": [], "line_numbers": {}}

    file_paths:     List[str] = []
    function_names: List[str] = []
    line_numbers:   Dict[str, int] = {}
    seen_files: set = set()
    seen_fns:   set = set()

    # ── Python: File "path/to/file.py", line 42, in function_name ────────────
    py_pat = re.compile(r'File "([^"]+\.py)", line (\d+)(?:, in (\w+))?')
    for m in py_pat.finditer(error_trace):
        raw_path = m.group(1).replace("\\", "/")
        # Keep only the repo-relative portion: strip everything up to the first
        # directory that looks like a project root (no leading slash, no /tmp/).
        path = re.sub(r"^.*?/(?=[a-zA-Z0-9_])", "", raw_path.lstrip("/"), count=1)
        lineno = int(m.group(2))
        fn_name = m.group(3) or ""

        if path and path not in seen_files:
            seen_files.add(path)
            file_paths.append(path)
        line_numbers[path] = lineno

        if fn_name and fn_name not in seen_fns and fn_name not in {"<module>", "<lambda>"}:
            seen_fns.add(fn_name)
            function_names.append(fn_name)

    # ── JavaScript/TypeScript: at fn (path/file.js:42:10) ────────────────────
    js_pat = re.compile(
        r"at\s+(?:\S+\s+)?\(([^)]+\.(js|ts|jsx|tsx|mjs)):(\d+):\d+\)"
    )
    for m in js_pat.finditer(error_trace):
        path = m.group(1).replace("\\", "/").lstrip("/")
        lineno = int(m.group(3))
        if path not in seen_files:
            seen_files.add(path)
            file_paths.append(path)
        line_numbers[path] = lineno

    # ── Java: at com.example.Class.method(FileName.java:42) ──────────────────
    java_pat = re.compile(r"at\s+([\w.$]+)\((\w+\.java):(\d+)\)")
    for m in java_pat.finditer(error_trace):
        fname   = m.group(2)
        lineno  = int(m.group(3))
        fn_part = m.group(1).rsplit(".", 1)[-1]  # leaf method name
        if fname not in seen_files:
            seen_files.add(fname)
            file_paths.append(fname)
        line_numbers[fname] = lineno
        if fn_part and fn_part not in seen_fns:
            seen_fns.add(fn_part)
            function_names.append(fn_part)

    # ── Go: goroutine / file.go:42 ────────────────────────────────────────────
    go_pat = re.compile(r"([a-zA-Z0-9_/.-]+\.go):(\d+)")
    for m in go_pat.finditer(error_trace):
        path   = m.group(1).replace("\\", "/").lstrip("/")
        lineno = int(m.group(2))
        if path not in seen_files:
            seen_files.add(path)
            file_paths.append(path)
        line_numbers[path] = lineno

    # ── Exception type ─────────────────────────────────────────────────────────
    # Python/JS: "SomeError: message" at the last line of the trace.
    exc_match = re.search(
        r"^([A-Z]\w*(?:Error|Exception|Fault|Warning|Panic|Interrupt))\b",
        error_trace,
        re.MULTILINE,
    )
    exception_type = exc_match.group(1) if exc_match else ""

    # Java: "Exception in thread ... com.example.SomeException"
    if not exception_type:
        java_exc = re.search(r"Exception in thread\s+\"\w+\"\s+(\S+Exception)", error_trace)
        if java_exc:
            exception_type = java_exc.group(1).rsplit(".", 1)[-1]

    return {
        "file_paths":     file_paths,
        "exception_type": exception_type,
        "function_names": function_names,
        "line_numbers":   line_numbers,
    }


def _extract_import_module_paths(title: str, description: str) -> List[str]:
    """
    Detect ImportError / ModuleNotFoundError in the ticket title or description
    and convert the Python module path to candidate file paths.

    Examples:
      "ImportError: cannot import name 'Customer' from ... 'apps.customers.models'"
        → ["apps/customers/models.py"]
      "ModuleNotFoundError: No module named 'apps.services.urls'"
        → ["apps/services/urls.py", "apps/services/urls/__init__.py"]

    Returns a list of relative file path candidates (forward slashes, no leading /).
    """
    combined = f"{title} {description[:500]}"
    candidates: List[str] = []

    # Pattern 1: ImportError: cannot import name '...' from '...' 'a.b.c'
    # The module is the last quoted token after "from"
    p1 = re.compile(
        r"ImportError[^'\"]*from[^'\"]*['\"]([a-zA-Z0-9_.]+)['\"]",
        re.IGNORECASE,
    )
    for m in p1.finditer(combined):
        mod = m.group(1).strip("'\" ")
        if mod:
            candidates.append(mod.replace(".", "/") + ".py")
            candidates.append(mod.replace(".", "/") + "/__init__.py")

    # Pattern 2: ModuleNotFoundError: No module named 'a.b.c'
    p2 = re.compile(r"No module named ['\"]([a-zA-Z0-9_.]+)['\"]", re.IGNORECASE)
    for m in p2.finditer(combined):
        mod = m.group(1).strip("'\" ")
        if mod:
            path = mod.replace(".", "/") + ".py"
            init = mod.replace(".", "/") + "/__init__.py"
            if path not in candidates:
                candidates.append(path)
            if init not in candidates:
                candidates.append(init)

    # Pattern 3: partially initialized module 'a.b.c'
    p3 = re.compile(r"partially initialized module ['\"]([a-zA-Z0-9_.]+)['\"]", re.IGNORECASE)
    for m in p3.finditer(combined):
        mod = m.group(1).strip("'\" ")
        if mod:
            path = mod.replace(".", "/") + ".py"
            if path not in candidates:
                candidates.append(path)

    return candidates


def _file_in_trace(fpath: str, trace_file_paths: List[str]) -> bool:
    """
    True when *fpath* (absolute or relative) matches one of the trace file paths.

    Matching is intentionally loose: a trace may report a path relative to the
    project root while BM25 holds the absolute path, or vice-versa.
    We check:
      1. fpath ends with trace_path         (e.g. /repo/foo/bar.py ends with foo/bar.py)
      2. trace_path ends with basename(fpath)  (for Java .java filename-only entries)
    """
    norm = fpath.replace("\\", "/")
    for tp in trace_file_paths:
        tp_norm = tp.replace("\\", "/")
        if norm.endswith(tp_norm) or tp_norm.endswith(norm.split("/")[-1]):
            return True
    return False


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
    ticket        = state["ticket"]
    mr_diff       = state["mr_diff"]
    repo_path     = state["repo_path"]
    extra_context = state.get("extra_context") or {}
    error_trace   = extra_context.get("error_trace", "")

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

    # ── Stack trace signal — highest priority ──────────────────────────────────
    trace_info     = _parse_stack_trace(error_trace)
    trace_paths    = trace_info["file_paths"]
    exception_type = trace_info["exception_type"]
    trace_fns      = trace_info["function_names"]

    # Build BM25 query: start from ticket+diff keywords, then prepend trace
    # function names and exception type so the most precise signals rank first.
    keywords = extract_keywords(ticket, mr_diff) or ["bug", "error", "exception"]

    # Prepend stack-trace function names (innermost first — crash site).
    trace_fn_tokens: List[str] = []
    for fn in trace_fns[:5]:
        for tok in _tokenize(fn):
            if tok not in keywords and tok not in trace_fn_tokens:
                trace_fn_tokens.append(tok)
    keywords = trace_fn_tokens + keywords

    # Prepend exception type tokens (e.g. "keyerror" → strong BM25 signal).
    if exception_type:
        for tok in reversed(_tokenize(exception_type)):
            if tok not in keywords:
                keywords.insert(0, tok)

    if trace_paths:
        logger.info(
            "[phase1] Stack trace: %d files, exception=%r, functions=%s",
            len(trace_paths), exception_type, trace_fns[:3],
        )

    scores = bm25.get_scores(keywords).tolist()

    # Boost files that the MR diff directly touches.
    mr_paths = _parse_mr_file_paths(mr_diff)
    for i, fpath in enumerate(files):
        if _file_in_diff(fpath, mr_paths):
            scores[i] *= MR_FILE_BOOST

    # Boost files explicitly named in the stack trace (applied after MR boost
    # so that a file in both diff and trace gets the full compound multiplier).
    if trace_paths:
        for i, fpath in enumerate(files):
            if _file_in_trace(fpath, trace_paths):
                scores[i] *= TRACE_FILE_BOOST

    # Boost files derived from ImportError / ModuleNotFoundError module path in
    # the ticket title / description.  Strongest signal (×10) because it names
    # the exact broken module.  Applied last so it compounds with MR + trace.
    import_module_paths = _extract_import_module_paths(
        ticket.get("title", ""),
        ticket.get("description", ""),
    )
    if import_module_paths:
        logger.info("[phase1] ImportError module paths detected: %s", import_module_paths)
        for i, fpath in enumerate(files):
            norm = fpath.replace("\\", "/")
            for mp in import_module_paths:
                if norm.endswith(mp) or norm.endswith(mp.replace("/__init__.py", ".py")):
                    scores[i] *= IMPORT_ERROR_BOOST
                    logger.info("[phase1] ImportError boost ×%.0f → %s", IMPORT_ERROR_BOOST, fpath)
                    break

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
