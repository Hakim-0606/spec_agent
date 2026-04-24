"""
Tool definitions and executor for SpecAgentV2.
Uses standard OpenAI-format tool dicts — no LangChain decorators.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

from .state import SpecV2State

logger = logging.getLogger(__name__)

# ── Tool definitions (OpenAI format) ──────────────────────────────────────────

SPEC_V2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bm25_search",
            "description": (
                "Lexical BM25 search over the repository. Returns ranked files "
                "with snippet matches. Use when you have keywords from a ticket "
                "and need to find the most relevant files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keywords or identifiers from the ticket.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ast_analyse",
            "description": (
                "Run TreeSitter AST analysis on a specific file. Returns functions "
                "with source, start/end lines, callers, and callees. Also populates "
                "the call graph so get_callers works afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path from repo root (e.g. 'apps/auth/views.py').",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Embed a query and retrieve the 3 most semantically similar function bodies "
                "from candidate functions previously returned by ast_analyse."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic query describing the bug or behaviour.",
                    },
                    "candidate_functions": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Pass an empty array [] — the tool automatically uses functions accumulated from previous ast_analyse calls.",
                    },
                },
                "required": ["query", "candidate_functions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read lines from a file with ±5 lines context window. "
                "Use to verify a function body, check imports, or confirm a line number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path from repo root (e.g. 'apps/auth/views.py').",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based). Default 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-based). 0 = read 80 lines from start.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_repo",
            "description": (
                "Regex search across all source files. Returns up to 10 matches "
                "(file, line, content). Use to find where a symbol is defined or referenced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (e.g. 'def my_func|class MyClass').",
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extensions to search (e.g. ['.py', '.js']). Default: all source.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callers",
            "description": (
                "Get callers and callees of a function from the accumulated call graph. "
                "IMPORTANT: call ast_analyse on the target file first — if the call graph "
                "is empty this tool returns an informative message asking you to do so."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Exact function name to look up.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Relative file path (optional, narrows the search).",
                    },
                },
                "required": ["function_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_commits",
            "description": (
                "Get the last 5 git commits that touched a file. Use to identify "
                "which commit introduced the regression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file (e.g. 'apps/auth/views.py').",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_config",
            "description": (
                "Read project configuration files. "
                "If file_path is given, reads that file. "
                "Otherwise auto-detects and returns the most relevant config files "
                "(requirements.txt, package.json, settings.py, .env, go.mod, Cargo.toml, etc.). "
                "Use when a bug may be caused by a missing dependency, wrong setting, "
                "or misconfigured environment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Specific config file to read (optional — omit to auto-detect).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_tests",
            "description": (
                "Find and read the test file for a given source module. "
                "Tests document what the code is SUPPOSED to do — "
                "reading them reveals expected behaviour and helps confirm the bug."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "module_path": {
                        "type": "string",
                        "description": "Relative path of the source file (e.g. 'apps/customers/models.py').",
                    },
                },
                "required": ["module_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_todos",
            "description": (
                "Search for TODO / FIXME / HACK / BUG comments in a file or the whole repo. "
                "Developers often annotate known problems — these comments point directly "
                "to fragile or broken code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Scope to a specific file (optional — omit for repo-wide scan).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": (
                "Show git changes on a file or the whole repo over the last N commits. "
                "Use when you suspect the bug was introduced by a recent change — "
                "the diff shows exactly what lines were added or removed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to scope the diff (optional — omit for repo-wide diff).",
                    },
                    "commits": {
                        "type": "integer",
                        "description": "How many recent commits to diff (default 3, max 5).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_in_file",
            "description": (
                "Regex search inside a single file with ±N lines of context. "
                "Use when read_file returns too much and you need to find a specific "
                "pattern (function definition, variable assignment, decorator) quickly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (e.g. 'def my_func|class MyClass').",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context before/after each match (default 3, max 10).",
                    },
                },
                "required": ["file_path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_usages",
            "description": (
                "Find all places in the codebase where a symbol (function, class, variable) "
                "is referenced. Use when you need to understand the impact of a change "
                "or find all callers beyond what ast_analyse provides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Exact symbol name to search for (e.g. 'Customer', 'process_order').",
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict search to these extensions (e.g. ['.py']). Default: all source.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_structure",
            "description": (
                "Return the source-file tree of the repository (dirs + files). "
                "Use at the start of exploration to understand where modules live "
                "before searching. Skips venv, migrations, __pycache__, node_modules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_depth": {
                        "type": "integer",
                        "description": "Directory depth to show (default 3, max 4).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "import_graph",
            "description": (
                "Build the local import graph starting from a file and detect "
                "circular imports. Parses Python import statements and follows "
                "only in-repo modules. Use when you suspect a circular import or "
                "want to understand inter-module dependencies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path of the starting file (e.g. 'apps/customers/models.py').",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many import levels to follow (default 2, max 3).",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": (
                "Read up to 3 files at once. Use when the bug spans multiple files "
                "and you need to see their contents together (e.g. two sides of a "
                "circular import, a model and its serializer, a caller and callee)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of relative file paths (max 3).",
                    },
                },
                "required": ["file_paths"],
            },
        },
    },
]

# ── Phase-specific tool subsets ────────────────────────────────────────────────
#
# EXPLORE tools: discover which files are relevant (broad search).
# CONFIRM tools: read and verify the candidate (deep dive).
#
# Keeping the sets disjoint forces the model to commit to a candidate
# before reading code, preventing aimless tool calling.

_EXPLORE_NAMES = {
    "bm25_search", "ast_analyse", "search_in_repo", "semantic_search",
    "get_project_structure", "import_graph",
    "git_diff", "find_usages", "scan_todos", "read_config",
}
_CONFIRM_NAMES = {
    "read_file", "read_files", "grep_in_file",
    "get_callers", "get_recent_commits", "read_tests",
}

EXPLORE_TOOLS = [t for t in SPEC_V2_TOOLS if t["function"]["name"] in _EXPLORE_NAMES]
CONFIRM_TOOLS = [t for t in SPEC_V2_TOOLS if t["function"]["name"] in _CONFIRM_NAMES]


def get_tools_for_phase(state: SpecV2State) -> list:
    """Return the tool subset appropriate for the current phase."""
    return CONFIRM_TOOLS if state.get("phase") == "confirm" else EXPLORE_TOOLS


# ── Tool executor ──────────────────────────────────────────────────────────────


def execute_tool(call: dict, state: SpecV2State) -> str:
    """
    Dispatch a tool call and return the result as a string.
    Deduplicates identical (tool, args) pairs seen in the same session.
    """
    tool_name = call.get("name", "")
    args      = call.get("arguments", {})

    # Deduplication
    try:
        call_key = f"{tool_name}:{hash(frozenset(sorted(args.items())))}"
    except TypeError:
        call_key = f"{tool_name}:{hash(str(sorted(args.items())))}"

    if call_key in state["seen_tool_calls"]:
        logger.info("[specv2] deduplicated tool call: %s(%s)", tool_name, args)
        return (
            f"Tool '{tool_name}' was already called with these arguments. "
            "Use the previous result or try different arguments."
        )
    state["seen_tool_calls"].add(call_key)

    repo_path = state.get("repo_path", "")

    try:
        if tool_name == "bm25_search":
            return _tool_bm25_search(args, state, repo_path)
        elif tool_name == "ast_analyse":
            return _tool_ast_analyse(args, state, repo_path)
        elif tool_name == "semantic_search":
            return _tool_semantic_search(args, state)
        elif tool_name == "read_file":
            return _tool_read_file(args, repo_path)
        elif tool_name == "search_in_repo":
            return _tool_search_in_repo(args, repo_path)
        elif tool_name == "get_callers":
            return _tool_get_callers(args, state)
        elif tool_name == "get_recent_commits":
            return _tool_get_recent_commits(args, repo_path)
        elif tool_name == "get_project_structure":
            return _tool_get_project_structure(args, repo_path)
        elif tool_name == "import_graph":
            return _tool_import_graph(args, repo_path)
        elif tool_name == "read_files":
            return _tool_read_files(args, repo_path)
        elif tool_name == "git_diff":
            return _tool_git_diff(args, repo_path)
        elif tool_name == "grep_in_file":
            return _tool_grep_in_file(args, repo_path)
        elif tool_name == "find_usages":
            return _tool_find_usages(args, repo_path)
        elif tool_name == "read_config":
            return _tool_read_config(args, repo_path)
        elif tool_name == "read_tests":
            return _tool_read_tests(args, repo_path)
        elif tool_name == "scan_todos":
            return _tool_scan_todos(args, repo_path)
        else:
            return f"(unknown tool: {tool_name})"
    except Exception as exc:
        logger.warning("[specv2] tool '%s' raised: %s", tool_name, exc)
        return f"(tool error: {exc})"


# ── Individual tool implementations ───────────────────────────────────────────


def _tool_bm25_search(args: dict, state: SpecV2State, repo_path: str) -> str:
    from agent_spec.phase1_bm25 import (
        BM25Okapi,
        collect_repo_files,
        _tokenize,
        _parse_mr_file_paths,
        _file_in_diff,
        MR_FILE_BOOST,
    )

    query = args.get("query", "")
    top_k = int(args.get("top_k") or 5)

    if not query or not repo_path:
        return "(bm25_search: missing query or repo_path)"

    # Build index lazily and cache it
    if state["bm25_index"] is None:
        files, contents = collect_repo_files(repo_path)
        if not files:
            return "(bm25_search: no source files found in repo)"
        tokenized = [_tokenize(c) for c in contents]
        bm25      = BM25Okapi(tokenized)
        state["bm25_index"] = {"bm25": bm25, "files": files}
        logger.info("[specv2] BM25 index built: %d files", len(files))

    idx   = state["bm25_index"]
    bm25  = idx["bm25"]
    files = idx["files"]

    query_tokens = _tokenize(query) or [query]
    scores = bm25.get_scores(query_tokens).tolist()

    mr_diff  = state.get("mr_diff", "")
    mr_paths = _parse_mr_file_paths(mr_diff) if mr_diff else set()
    for i, fp in enumerate(files):
        if mr_paths and _file_in_diff(fp, mr_paths):
            scores[i] *= MR_FILE_BOOST

    ranked = sorted(
        [{"file": files[i], "score": scores[i]} for i in range(len(files))],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_k]

    lines = []
    for r in ranked:
        rel = _rel(r["file"], repo_path)
        lines.append(f"score={r['score']:.2f}  {rel}")
    return "\n".join(lines) if lines else "(bm25_search: no results)"


def _tool_ast_analyse(args: dict, state: SpecV2State, repo_path: str) -> str:
    from agent_spec.phase2_treesitter import parse_file, build_call_graph

    file_path = args.get("file_path", "")
    if not file_path:
        return "(ast_analyse: file_path required)"

    abs_path = str(Path(repo_path) / file_path) if repo_path else file_path
    fns = parse_file(abs_path)
    if not fns:
        return f"(ast_analyse: no functions found in '{file_path}' — check path or language support)"

    # Tag each function with the relative file path
    for fn in fns:
        fn["file"] = file_path.replace("\\", "/")

    # Accumulate into state (merge, not replace)
    if state["all_functions"] is None:
        state["all_functions"] = []

    existing_keys = {
        (f.get("file", ""), f.get("function", ""))
        for f in state["all_functions"]
    }
    for fn in fns:
        key = (fn.get("file", ""), fn.get("function", ""))
        if key not in existing_keys:
            state["all_functions"].append(fn)
            existing_keys.add(key)

    # Populate callers/callees lists on each accumulated function from call graph
    try:
        import networkx as nx
        G = build_call_graph(state["all_functions"])
        state["repo_graph"] = G  # store DiGraph directly
        for fn in state["all_functions"]:
            nid = f"{fn.get('file', '')}::{fn.get('function', '')}"
            if nid in G:
                fn["callers"] = [
                    n.split("::", 1)[-1] for n in G.predecessors(nid)
                ]
                fn["callees"] = [
                    n.split("::", 1)[-1] for n in G.successors(nid)
                ]
    except Exception as exc:
        logger.debug("[specv2] call graph build failed: %s", exc)

    lines = [f"Found {len(fns)} functions in '{file_path}':"]
    for fn in fns:
        callers = ", ".join(fn.get("callers") or []) or "none"
        callees = ", ".join(fn.get("callees") or []) or "none"
        lines.append(
            f"  {fn.get('function')} [L{fn.get('start_line', '?')}-L{fn.get('end_line', '?')}]"
            f"  callers=[{callers}]  callees=[{callees}]"
        )
    return "\n".join(lines)


def _tool_semantic_search(args: dict, state: SpecV2State) -> str:
    from agent_spec.phase3_rag import _semantic_rerank

    query      = args.get("query", "")
    candidates = args.get("candidate_functions") or []

    if not query:
        return "(semantic_search: query required)"
    if not candidates:
        # Fall back to state["all_functions"] if no explicit candidates
        candidates = state.get("all_functions") or []
    if not candidates:
        return (
            "(semantic_search: no candidate functions available. "
            "Call ast_analyse on a file first.)"
        )

    try:
        results = _semantic_rerank(candidates, query, top_k=3)
    except Exception as exc:
        return f"(semantic_search failed: {exc})"

    if not results:
        return "(semantic_search: no results)"

    lines = [f"Top-{len(results)} semantic matches for '{query[:60]}':"]
    for r in results:
        lines.append(
            f"  score={r.get('semantic_score', 0.0):.3f}  "
            f"{r.get('file', '?')}::{r.get('function', '?')}"
        )
    return "\n".join(lines)


def _tool_read_file(args: dict, repo_path: str) -> str:
    from agent_spec.phase35_tools import read_file as _read_file

    fp  = args.get("file_path", "")
    sl  = int(args.get("start_line") or 1)
    el  = int(args.get("end_line")   or 0)
    if el == 0:
        el = sl + 79

    result = _read_file(fp, sl, el, repo_path)
    return result or f"(read_file: file not found: {fp})"


def _tool_search_in_repo(args: dict, repo_path: str) -> str:
    from agent_spec.phase35_tools import search_in_repo as _search

    pattern = args.get("pattern", "")
    exts    = args.get("file_extensions") or None

    hits = _search(pattern, repo_path, exts)
    if not hits:
        return "(search_in_repo: no matches found)"
    return "\n".join(
        f"{_rel(h['file'], repo_path)}:{h['line']} → {h['content']}"
        for h in hits[:10]
    )


def _tool_get_callers(args: dict, state: SpecV2State) -> str:
    all_fns: Optional[List[dict]] = state.get("all_functions")

    if not all_fns:
        return (
            "Call graph not yet built. Call ast_analyse on the relevant file first, "
            "then retry get_callers."
        )

    fn_name   = args.get("function_name", "")
    file_hint = args.get("file_path", "")

    for fn in all_fns:
        if fn.get("function") == fn_name:
            if not file_hint or file_hint in fn.get("file", ""):
                callers = fn.get("callers") or []
                callees = fn.get("callees") or []
                parts   = []
                if callers:
                    parts.append(f"Callers : {', '.join(callers)}")
                if callees:
                    parts.append(f"Callees : {', '.join(callees)}")
                return "\n".join(parts) if parts else f"(no call graph data for '{fn_name}')"

    return f"(function '{fn_name}' not found in call graph — call ast_analyse on the file first)"


def _tool_get_recent_commits(args: dict, repo_path: str) -> str:
    import subprocess

    fp = args.get("file_path", "")
    if not fp:
        return "(get_recent_commits: file_path required)"
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "log", "-5",
             "--oneline", "--follow", "--", fp],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or "(no commits found)"
    except Exception as exc:
        return f"(git log failed: {exc})"


def _tool_git_diff(args: dict, repo_path: str) -> str:
    import subprocess

    file_path = args.get("file_path", "")
    commits   = min(int(args.get("commits") or 3), 5)

    def _run(cmd: list) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return r.stdout.strip()
        except Exception as exc:
            return f"(git error: {exc})"

    # Committed changes over last N commits
    cmd = ["git", "-C", repo_path, "diff", f"HEAD~{commits}..HEAD"]
    if file_path:
        cmd += ["--", file_path]
    output = _run(cmd)

    # Fall back to uncommitted changes if nothing committed
    if not output:
        cmd2 = ["git", "-C", repo_path, "diff", "HEAD"]
        if file_path:
            cmd2 += ["--", file_path]
        output = _run(cmd2)

    if not output:
        return f"(git_diff: no changes in the last {commits} commits)"

    if len(output) > 3000:
        output = output[:3000] + "\n... (truncated — use file_path to narrow)"
    return output


def _tool_grep_in_file(args: dict, repo_path: str) -> str:
    file_path = args.get("file_path", "")
    pattern   = args.get("pattern", "")
    context   = min(int(args.get("context_lines") or 3), 10)

    if not file_path or not pattern:
        return "(grep_in_file: file_path and pattern required)"

    abs_path = Path(repo_path) / file_path
    if not abs_path.exists():
        return f"(grep_in_file: file not found: {file_path})"

    try:
        lines = abs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        regex = re.compile(pattern)
    except re.error as exc:
        return f"(grep_in_file: invalid regex: {exc})"

    matches = []
    for i, line in enumerate(lines):
        if regex.search(line):
            start = max(0, i - context)
            end   = min(len(lines), i + context + 1)
            block = []
            for j in range(start, end):
                marker = "→" if j == i else " "
                block.append(f"{marker} {j + 1:4d} | {lines[j]}")
            matches.append("\n".join(block))

    if not matches:
        return f"(grep_in_file: no matches for '{pattern}' in {file_path})"

    header = f"Found {len(matches)} match(es) for '{pattern}' in {file_path}:\n"
    body   = ("\n" + "─" * 40 + "\n").join(matches[:10])
    suffix = f"\n... ({len(matches) - 10} more matches)" if len(matches) > 10 else ""
    return header + body + suffix


def _tool_find_usages(args: dict, repo_path: str) -> str:
    from agent_spec.phase35_tools import search_in_repo as _search

    symbol = args.get("symbol", "")
    exts   = args.get("file_extensions") or None

    if not symbol:
        return "(find_usages: symbol required)"

    # Escape for regex but keep the symbol recognisable
    escaped = re.escape(symbol)
    hits = _search(escaped, repo_path, exts)

    if not hits:
        return f"(find_usages: no usages of '{symbol}' found)"

    lines = [f"Found {len(hits)} usage(s) of '{symbol}':"]
    for h in hits[:15]:
        lines.append(
            f"  {_rel(h['file'], repo_path)}:{h['line']}  →  {h['content'].strip()}"
        )
    if len(hits) > 15:
        lines.append(f"  ... ({len(hits) - 15} more — add file_extensions to narrow)")
    return "\n".join(lines)


def _tool_read_config(args: dict, repo_path: str) -> str:
    repo = Path(repo_path)

    # If a specific file is requested, read it directly
    file_path = args.get("file_path", "")
    if file_path:
        target = repo / file_path
        if not target.exists():
            return f"(read_config: file not found: {file_path})"
        try:
            content = target.read_text(encoding="utf-8", errors="ignore")
            lines   = content.splitlines()
            preview = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines[:100]))
            suffix  = f"\n... ({len(lines)-100} more lines)" if len(lines) > 100 else ""
            return f"=== {file_path} ===\n{preview}{suffix}"
        except Exception as exc:
            return f"(read_config: error reading {file_path}: {exc})"

    # Auto-detect: ordered by cross-language relevance
    _CANDIDATES = [
        # Environment & secrets
        ".env", ".env.example", ".env.local", ".env.development",
        # Python
        "requirements.txt", "requirements-dev.txt", "requirements/base.txt",
        "pyproject.toml", "setup.cfg", "setup.py", "Pipfile",
        # Node / JS / TS
        "package.json", "tsconfig.json", ".npmrc",
        # Java / Kotlin
        "pom.xml", "build.gradle", "build.gradle.kts",
        "src/main/resources/application.yml",
        "src/main/resources/application.yaml",
        "src/main/resources/application.properties",
        # Go
        "go.mod",
        # Rust
        "Cargo.toml",
        # Ruby
        "Gemfile",
        # PHP
        "composer.json",
        # Infrastructure
        "docker-compose.yml", "docker-compose.yaml", "Makefile",
        # Django-specific (searched recursively below)
        "settings.py", "base.py", "urls.py",
    ]

    found = []

    # 1. Check exact paths at repo root
    for name in _CANDIDATES:
        p = repo / name
        if p.exists() and p.is_file():
            found.append(p)

    # 2. For Django settings.py / urls.py — search one level deeper
    if not any(f.name == "settings.py" for f in found):
        for p in repo.rglob("settings.py"):
            rel = p.relative_to(repo).parts
            if len(rel) <= 4:   # avoid deeply nested test fixtures
                found.append(p)
                break

    if not found:
        return "(read_config: no config files detected in repo root)"

    # Read up to 3 files, cap each at 80 lines
    parts = []
    for fp in found[:3]:
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
            lines   = content.splitlines()
            rel     = str(fp.relative_to(repo)).replace("\\", "/")
            preview = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines[:80]))
            suffix  = f"\n... ({len(lines)-80} more lines)" if len(lines) > 80 else ""
            parts.append(f"=== {rel} ===\n{preview}{suffix}")
        except Exception:
            continue

    if not found[3:]:
        return "\n\n".join(parts)

    skipped = [str(f.relative_to(repo)).replace("\\", "/") for f in found[3:]]
    return "\n\n".join(parts) + f"\n\n(Also found: {skipped} — call read_config(file_path=...) to read them)"


def _tool_read_tests(args: dict, repo_path: str) -> str:
    module_path = args.get("module_path", "")
    if not module_path:
        return "(read_tests: module_path required)"

    repo     = Path(repo_path)
    mod      = Path(module_path)
    stem     = mod.stem          # e.g. "models"
    parent   = mod.parent        # e.g. apps/customers
    app_name = parent.name       # e.g. "customers"

    # Candidate test file locations — ordered by likelihood
    candidates = [
        parent / "tests" / f"test_{stem}.py",
        parent / f"test_{stem}.py",
        parent / "tests.py",
        parent / "tests" / "tests.py",
        repo / "tests" / f"test_{app_name}_{stem}.py",
        repo / "tests" / f"test_{app_name}.py",
        repo / "tests" / app_name / f"test_{stem}.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
                rel     = str(candidate.relative_to(repo)).replace("\\", "/")
                lines   = content.splitlines()
                preview = "\n".join(
                    f"{i+1:4d} | {l}" for i, l in enumerate(lines[:120])
                )
                suffix  = f"\n... ({len(lines)-120} more lines)" if len(lines) > 120 else ""
                return f"=== Test file: {rel} ===\n{preview}{suffix}"
            except Exception as exc:
                return f"(read_tests: error reading {candidate}: {exc})"

    return (
        f"(read_tests: no test file found for '{module_path}'. "
        f"Tried: {[str(c.relative_to(repo)) for c in candidates]})"
    )


def _tool_scan_todos(args: dict, repo_path: str) -> str:
    from agent_spec.phase35_tools import search_in_repo as _search

    file_path = args.get("file_path", "")
    pattern   = r"#\s*(TODO|FIXME|HACK|BUG|XXX)\b"

    if file_path:
        # Scope to a single file using grep_in_file logic
        abs_path = Path(repo_path) / file_path
        if not abs_path.exists():
            return f"(scan_todos: file not found: {file_path})"
        try:
            lines   = abs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            regex   = re.compile(pattern, re.IGNORECASE)
            matches = []
            for i, line in enumerate(lines):
                if regex.search(line):
                    matches.append(f"  {file_path}:{i+1}  →  {line.strip()}")
        except Exception as exc:
            return f"(scan_todos: error: {exc})"
        if not matches:
            return f"(scan_todos: no TODO/FIXME/HACK/BUG found in {file_path})"
        return f"Found {len(matches)} annotation(s) in {file_path}:\n" + "\n".join(matches)

    # Repo-wide scan
    hits = _search(pattern, repo_path, None)
    if not hits:
        return "(scan_todos: no TODO/FIXME/HACK/BUG annotations found in repo)"

    lines_out = [f"Found {len(hits)} annotation(s) repo-wide:"]
    for h in hits[:20]:
        lines_out.append(
            f"  {_rel(h['file'], repo_path)}:{h['line']}  →  {h['content'].strip()}"
        )
    if len(hits) > 20:
        lines_out.append(
            f"  ... ({len(hits) - 20} more — use file_path to narrow)"
        )
    return "\n".join(lines_out)


def _tool_get_project_structure(args: dict, repo_path: str) -> str:
    from pathlib import Path as _Path

    max_depth = min(int(args.get("max_depth") or 3), 4)

    _SKIP_DIRS = {
        ".git", "__pycache__", "venv", ".venv", "env", "node_modules",
        "migrations", ".pytest_cache", "dist", "build", ".mypy_cache",
        "htmlcov", ".tox", ".eggs", "*.egg-info",
    }
    _SOURCE_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".rb", ".php", ".cs", ".kt", ".swift",
    }

    lines: list = []

    def _walk(path: _Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        dirs  = [e for e in entries if e.is_dir()  and e.name not in _SKIP_DIRS]
        files = [e for e in entries if e.is_file() and e.suffix in _SOURCE_EXTS]
        for d in dirs:
            lines.append(f"{prefix}{d.name}/")
            _walk(d, depth + 1, prefix + "  ")
        for f in files:
            lines.append(f"{prefix}{f.name}")

    _walk(_Path(repo_path), 0, "")

    if not lines:
        return "(get_project_structure: no source files found)"
    truncated = lines[:120]
    if len(lines) > 120:
        truncated.append(f"... ({len(lines) - 120} more entries)")
    return "\n".join(truncated)


def _tool_import_graph(args: dict, repo_path: str) -> str:
    import ast as _ast
    from pathlib import Path as _Path

    file_path = args.get("file_path", "")
    max_depth = min(int(args.get("depth") or 2), 3)

    if not file_path:
        return "(import_graph: file_path required)"

    repo = _Path(repo_path)
    start = repo / file_path
    if not start.exists():
        return f"(import_graph: file not found: {file_path})"

    def _module_to_path(module: str) -> Optional[_Path]:
        candidate = repo / (module.replace(".", "/") + ".py")
        if candidate.exists():
            return candidate
        pkg = repo / module.replace(".", "/") / "__init__.py"
        if pkg.exists():
            return pkg
        return None

    def _get_imports(filepath: _Path) -> List[str]:
        try:
            source = filepath.read_text(encoding="utf-8", errors="ignore")
            tree = _ast.parse(source, filename=str(filepath))
        except Exception:
            return []
        modules: List[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, _ast.Import):
                for alias in node.names:
                    modules.append(alias.name)
        return modules

    # BFS — build adjacency list of local imports
    graph: dict = {}
    visited: set = set()
    queue: list = [(start, 0)]

    while queue:
        current, depth = queue.pop(0)
        rel = str(current.relative_to(repo)).replace("\\", "/")
        if rel in visited or depth > max_depth:
            continue
        visited.add(rel)
        local: List[str] = []
        for mod in _get_imports(current):
            target = _module_to_path(mod)
            if target:
                trel = str(target.relative_to(repo)).replace("\\", "/")
                local.append(trel)
                if trel not in visited:
                    queue.append((target, depth + 1))
        if local:
            graph[rel] = local

    # Cycle detection via DFS
    cycles: list = []

    def _dfs(node: str, path: list, on_stack: set) -> None:
        if node in on_stack:
            idx = path.index(node)
            cycles.append(path[idx:] + [node])
            return
        if node not in graph:
            return
        on_stack.add(node)
        for nb in graph[node]:
            _dfs(nb, path + [nb], on_stack)
        on_stack.discard(node)

    start_rel = str(start.relative_to(repo)).replace("\\", "/")
    _dfs(start_rel, [start_rel], set())

    lines = [f"Import graph from '{file_path}' (depth={max_depth}):"]
    for src, targets in graph.items():
        for tgt in targets:
            lines.append(f"  {src}  →  {tgt}")

    if cycles:
        lines.append(f"\n⚠  CIRCULAR IMPORTS ({len(cycles)} found):")
        for cycle in cycles[:5]:
            lines.append("  " + " → ".join(cycle))
    else:
        lines.append("\n✓  No circular imports detected.")

    return "\n".join(lines)


def _tool_read_files(args: dict, repo_path: str) -> str:
    file_paths = args.get("file_paths") or []
    if not file_paths:
        return "(read_files: file_paths list required)"

    parts = []
    for fp in file_paths[:3]:  # cap at 3 to avoid token overflow
        content = _tool_read_file({"file_path": fp}, repo_path)
        parts.append(f"=== {fp} ===\n{content}")

    return "\n\n".join(parts)


# ── Helper ─────────────────────────────────────────────────────────────────────


def _rel(abs_path: str, repo_path: str) -> str:
    """Return path relative to repo_path, forward slashes."""
    try:
        return str(Path(abs_path).relative_to(repo_path)).replace("\\", "/")
    except (ValueError, TypeError):
        return abs_path.replace("\\", "/")
