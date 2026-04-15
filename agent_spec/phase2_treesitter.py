"""
Phase 2 — tree-sitter + NetworkX PageRank + MR diff boosts
===========================================================
Parses the top-10 BM25 files with tree-sitter to extract every function
definition and every call relationship.  Builds a directed call graph with
NetworkX, ranks functions via PageRank, then applies MR diff signal boosts
before returning the top-5 candidates enriched with caller/callee lists.

No LLM.  No GPU.  Fully offline.

Compatible with tree-sitter >= 0.22 + individual language packages:
    pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript
                tree-sitter-java tree-sitter-go tree-sitter-rust

Scoring: (0.7 × keyword_match + 0.3 × pagerank_score) × diff_boost
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .diff_signal_extractor import (
    BOOST_NONE,
    DiffSignalExtractor,
)

logger = logging.getLogger(__name__)

import networkx as nx

# ── Language registry ──────────────────────────────────────────────────────────
# Maps file extension → (tree-sitter language key, module name)
# The module is imported lazily so missing languages degrade gracefully.

EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".swift": "swift",
}

# Module names for each language key (tree-sitter >= 0.22 individual packages).
_LANG_MODULE: Dict[str, str] = {
    "python":     "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java":       "tree_sitter_java",
    "go":         "tree_sitter_go",
    "rust":       "tree_sitter_rust",
    "cpp":        "tree_sitter_cpp",
    "c":          "tree_sitter_c",
    "c_sharp":    "tree_sitter_c_sharp",
    "ruby":       "tree_sitter_ruby",
    "kotlin":     "tree_sitter_kotlin",
    "swift":      "tree_sitter_swift",
}

FUNCTION_DEF_TYPES: Dict[str, List[str]] = {
    "python":     ["function_definition"],
    "javascript": ["function_declaration", "function_expression", "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "function_expression", "arrow_function", "method_definition"],
    "java":       ["method_declaration", "constructor_declaration"],
    "go":         ["function_declaration", "method_declaration"],
    "rust":       ["function_item"],
    "cpp":        ["function_definition"],
    "c":          ["function_definition"],
    "c_sharp":    ["method_declaration", "constructor_declaration"],
    "ruby":       ["method", "singleton_method"],
    "kotlin":     ["function_declaration"],
    "swift":      ["function_declaration"],
}

CALL_EXPR_TYPES: Dict[str, List[str]] = {
    "python":     ["call"],
    "javascript": ["call_expression", "new_expression"],
    "typescript": ["call_expression", "new_expression"],
    "java":       ["method_invocation", "object_creation_expression"],
    "go":         ["call_expression"],
    "rust":       ["call_expression", "method_call_expression"],
    "cpp":        ["call_expression"],
    "c":          ["call_expression"],
    "c_sharp":    ["invocation_expression", "object_creation_expression"],
    "ruby":       ["call", "method_call"],
    "kotlin":     ["call_expression"],
    "swift":      ["call_expression"],
}

# ── Parser cache (one Parser per language) ─────────────────────────────────────

_parser_cache: Dict[str, object] = {}


def _get_parser(lang: str):
    """Return a cached tree-sitter Parser for the given language key, or None."""
    if lang in _parser_cache:
        return _parser_cache[lang]

    module_name = _LANG_MODULE.get(lang)
    if not module_name:
        _parser_cache[lang] = None
        return None

    try:
        import importlib
        from tree_sitter import Language, Parser

        mod = importlib.import_module(module_name)

        # tree-sitter >= 0.22: Language(mod.language())
        ts_language = Language(mod.language())
        parser = Parser(ts_language)
        _parser_cache[lang] = parser
        return parser
    except Exception:
        _parser_cache[lang] = None
        return None


# ── AST traversal helpers ──────────────────────────────────────────────────────


def _find_all(node, types: List[str]) -> list:
    """Iterative DFS — all descendant nodes whose type is in `types`."""
    results = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type in types:
            results.append(cur)
        stack.extend(reversed(cur.children))
    return results


def _node_text(node) -> str:
    """Return the source text of a node (tree-sitter 0.22+: node.text is bytes)."""
    raw = node.text
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _func_name(node) -> Optional[str]:
    """Return the identifier name of a function-definition node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node).strip()
    # Fallback: first identifier child.
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child).strip()
    return None


def _call_name(node) -> Optional[str]:
    """
    Extract the leaf function name from a call expression.
    `obj.module.verify_token(...)` → `verify_token`
    """
    target = (
        node.child_by_field_name("function")
        or node.child_by_field_name("name")
        or (node.children[0] if node.children else None)
    )
    if target is None:
        return None

    text = _node_text(target).strip()
    parts = re.split(r"[.\->\(:]+", text)
    leaf = parts[-1].strip() if parts else text
    return leaf if re.match(r"^[a-zA-Z_]\w*$", leaf) else None


def _extract_class_name(func_node) -> Optional[str]:
    """Return the enclosing class/struct name, if any."""
    CLASS_TYPES = {
        "class_definition", "class_declaration", "class_body",
        "struct_item", "impl_item",
    }
    cur = func_node.parent
    while cur is not None:
        if cur.type in CLASS_TYPES:
            name_node = cur.child_by_field_name("name")
            if name_node:
                return _node_text(name_node).strip()
        cur = cur.parent
    return None


# ── Per-file parsing ───────────────────────────────────────────────────────────


def parse_file(fpath: str) -> List[Dict]:
    """
    Parse one source file.  Returns a list of function dicts:
    {
        file, language, function, class,
        start_line, end_line, source, signature,
        raw_callees  (set of called function names, unresolved)
    }
    """
    lang = EXT_TO_LANG.get(Path(fpath).suffix.lower())
    if not lang:
        return []

    parser = _get_parser(lang)
    if parser is None:
        return []

    try:
        with open(fpath, "rb") as fh:
            src_bytes = fh.read()
    except (IOError, OSError):
        return []

    try:
        tree = parser.parse(src_bytes)
    except Exception:
        return []

    func_def_types = FUNCTION_DEF_TYPES.get(lang, ["function_definition"])
    call_types = CALL_EXPR_TYPES.get(lang, ["call_expression"])

    functions: List[Dict] = []
    for fn in _find_all(tree.root_node, func_def_types):
        name = _func_name(fn)
        if not name:
            continue

        start_line = fn.start_point.row + 1   # convert 0-based → 1-based
        end_line = fn.end_point.row + 1
        source = src_bytes[fn.start_byte:fn.end_byte].decode("utf-8", errors="replace")
        signature = source.splitlines()[0].rstrip(":").strip()

        raw_callees: set = set()
        for cn in _find_all(fn, call_types):
            cname = _call_name(cn)
            if cname and cname != name:
                raw_callees.add(cname)

        functions.append({
            "file":        fpath,
            "language":    lang,
            "function":    name,
            "class":       _extract_class_name(fn),
            "start_line":  start_line,
            "end_line":    end_line,
            "source":      source,
            "signature":   signature,
            "raw_callees": raw_callees,
        })

    return functions


# ── Call-graph builder ─────────────────────────────────────────────────────────


def build_call_graph(all_functions: List[Dict]) -> nx.DiGraph:
    """
    Build a directed call graph.
    Node  : "filepath::function_name"
    Edge  : caller → callee
    """
    G = nx.DiGraph()

    name_index: Dict[str, List[str]] = {}
    for fn in all_functions:
        nid = f"{fn['file']}::{fn['function']}"
        G.add_node(nid, **{k: v for k, v in fn.items() if k != "raw_callees"})
        name_index.setdefault(fn["function"], []).append(nid)

    for fn in all_functions:
        caller_id = f"{fn['file']}::{fn['function']}"
        for callee_name in fn.get("raw_callees", set()):
            for target_id in name_index.get(callee_name, []):
                if target_id != caller_id:
                    G.add_edge(caller_id, target_id)

    return G


# ── Scoring ────────────────────────────────────────────────────────────────────


def _keyword_score(fn: Dict, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    haystack = (fn["function"] + " " + fn["source"]).lower()
    return sum(1 for kw in keywords if kw in haystack) / len(keywords)


def score_and_rank(
    all_functions: List[Dict],
    G: nx.DiGraph,
    keywords: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """Score = 0.7 × keyword_match + 0.3 × normalised_pagerank. Returns top-k."""
    if not all_functions:
        return []

    try:
        pr: Dict[str, float] = nx.pagerank(G, alpha=0.85)
    except Exception:
        pr = {}

    pr_max = max(pr.values(), default=1.0) or 1.0

    scored: List[Tuple[float, Dict]] = []
    for fn in all_functions:
        nid = f"{fn['file']}::{fn['function']}"
        kw_s = _keyword_score(fn, keywords)
        pg_s = pr.get(nid, 0.0) / pr_max
        combined = 0.7 * kw_s + 0.3 * pg_s

        callers = [
            G.nodes[p].get("function", p.split("::")[-1])
            for p in G.predecessors(nid) if p in G.nodes
        ]
        callees = [
            G.nodes[s].get("function", s.split("::")[-1])
            for s in G.successors(nid) if s in G.nodes
        ]

        scored.append((combined, {
            **fn,
            "callers":      callers,
            "callees":      callees,
            "score":        combined,
            "pagerank":     pg_s,
            "kw_score":     kw_s,
            "raw_callees":  None,   # drop internal field
        }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [fn for _, fn in scored[:top_k]]


# ── Diff boost application ─────────────────────────────────────────────────────


def _apply_diff_boosts(
    scored_functions: List[Dict],
    mr_diff: str,
    ticket: dict,
    top_k: int = 5,
) -> List[Dict]:
    """
    Apply MR diff signal boosts to pre-scored functions, re-rank, return top-k.

    Each returned function dict gains two new fields:
        diff_boost  (float) — multiplier applied  (1.0 when no signal)
        final_score (float) — score × diff_boost  (used for re-ranking)

    Falls back gracefully to original ranking (diff_boost=1.0) on any error.
    """
    try:
        extractor = DiffSignalExtractor(mr_diff, ticket)
        boosts    = extractor.compute_function_boosts(scored_functions)
    except Exception as exc:
        logger.warning(
            f"[phase2] Diff boost extraction failed ({exc}) — no boost applied."
        )
        boosts = {}

    boosted: List[Dict] = []
    for fn in scored_functions:
        fn_id       = f"{fn.get('file', '')}::{fn.get('function', '')}"
        boost       = boosts.get(fn_id, BOOST_NONE)
        base_score  = fn.get("score", 0.0)
        final_score = base_score * boost
        boosted.append({
            **fn,
            "diff_boost":  boost,
            "final_score": final_score,
        })

    boosted.sort(key=lambda x: x["final_score"], reverse=True)
    return boosted[:top_k]


# ── LangGraph node ─────────────────────────────────────────────────────────────


def phase_treesitter(state: dict) -> dict:
    """
    LangGraph node — Phase 2.

    Reads:  bm25_files, keywords, mr_diff, ticket
    Writes: repo_graph (nx.node_link_data dict), ast_functions (top-5), all_functions

    ast_functions entries include two new fields vs. the original:
        diff_boost  — MR diff signal multiplier applied to this function
        final_score — score × diff_boost (used for the final ranking)
    """
    bm25_files: List[Dict] = state.get("bm25_files", [])
    keywords:   List[str]  = state.get("keywords", [])
    mr_diff:    str        = state.get("mr_diff", "")
    ticket:     dict       = state.get("ticket", {})

    logger.info("[phase2] Parsing %d BM25 files with tree-sitter…", len(bm25_files))

    all_functions: List[Dict] = []
    skipped = 0
    for entry in bm25_files:
        parsed = parse_file(entry["file"])
        if parsed:
            all_functions.extend(parsed)
        else:
            skipped += 1

    # For files with no parseable functions (module-level code like urlpatterns,
    # settings, constants), create a synthetic "_module_" entry so they still
    # reach Phase 3 RAG and can be identified by the LLM.
    parsed_files = {fn.get("file") for fn in all_functions}
    synthetic_added = 0
    for entry in bm25_files:
        fpath = entry.get("file", "")
        if not fpath or fpath in parsed_files:
            continue
        ext  = Path(fpath).suffix.lower()
        lang = EXT_TO_LANG.get(ext, "")
        if not lang:
            continue
        try:
            content = Path(fpath).read_text(encoding="utf-8", errors="replace")
            lines   = content.splitlines()
            all_functions.append({
                "file":       fpath,
                "function":   "_module_level_",
                "class":      None,
                "start_line": 1,
                "end_line":   len(lines),
                "source":     "\n".join(lines[:100]),  # first 100 lines (covers imports + module body)
                "signature":  f"# module: {Path(fpath).name}",
                "callers":    [],
                "callees":    [],
                "raw_callees": set(),
                "language":   lang,
                "score":      0.0,
                "diff_boost": 1.0,
                "final_score": 0.0,
            })
            synthetic_added += 1
        except Exception:
            pass

    if synthetic_added:
        logger.info("[phase2] Added %d synthetic module-level entries for files with no functions.", synthetic_added)

    # ── Prop 4: Extract module headers (import blocks) for all BM25 files ────
    # Stores the first MAX_HEADER_LINES lines of each file — captures imports,
    # module-level variables, and class definitions. Used by Phase 4 to give the
    # LLM full visibility into circular imports and module-level bugs.
    MAX_HEADER_LINES = 60
    file_import_blocks: Dict[str, str] = {}
    for entry in bm25_files:
        fpath = entry.get("file", "")
        if not fpath:
            continue
        try:
            raw_lines = Path(fpath).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            block = "\n".join(raw_lines[:MAX_HEADER_LINES])
            if block.strip():
                file_import_blocks[fpath] = block
        except Exception:
            pass
    logger.info(
        "[phase2] Module headers extracted for %d/%d BM25 files.",
        len(file_import_blocks), len(bm25_files),
    )

    langs_found = sorted({fn.get("language", "?") for fn in all_functions})
    logger.info(
        "[phase2] Extracted %d functions from %d files (skipped %d, synthetic %d) | languages: %s",
        len(all_functions) - synthetic_added, len(bm25_files) - skipped, skipped, synthetic_added,
        ", ".join(langs_found) or "none",
    )

    if not all_functions:
        logger.warning("[phase2] No functions extracted — Phase 2 degraded (tree-sitter parse failed?).")
        return {**state, "repo_graph": {}, "ast_functions": [], "all_functions": []}

    G = build_call_graph(all_functions)

    # Score ALL parsed functions so that diff boosts can promote candidates
    # that were ranked below top-5 in the raw PageRank pass.
    n_all = len(all_functions)
    all_scored = score_and_rank(all_functions, G, keywords, top_k=n_all or 1)

    # Apply MR diff boosts and re-rank → final top-5.
    top5 = _apply_diff_boosts(all_scored, mr_diff, ticket, top_k=5)

    logger.info(
        "[phase2] Top-5 candidates: %s",
        ", ".join(
            "%s::%s (score=%.3f)" % (fn.get("file", "?"), fn.get("function", "?"), fn.get("final_score", 0))
            for fn in top5
        ) or "none",
    )

    # Serialise raw_callees (set → list) for JSON checkpointing.
    for fn in all_functions:
        fn["raw_callees"] = list(fn.get("raw_callees") or [])

    graph_data = nx.node_link_data(G)
    for node in graph_data.get("nodes", []):
        for k, v in list(node.items()):
            if isinstance(v, set):
                node[k] = list(v)

    return {
        **state,
        "repo_graph":         graph_data,
        "ast_functions":      top5,
        "all_functions":      all_functions,
        "file_import_blocks": file_import_blocks,   # Prop 4: module headers
    }
