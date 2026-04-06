"""
Phase 4 — LLM Confirmation + Reflexion Pattern
================================================
Sends a surgical, pre-filtered context (ticket + diff + top-3 functions with
callers/callees) to the local Ollama model and parses a structured JSON result.

Reflexion: if confidence < 0.7, the LLM self-critiques and a second call
           is made with the graph neighbourhood of the candidate function
           expanded into the context.

Budget: max 2 LLM calls per execution.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
REFLEXION_THRESHOLD = 0.7

# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert software engineer specialised in bug localisation.
You will receive a bug ticket, a merge-request diff, candidate functions with
their full source code, a code snippet extracted around the suspected bug line,
and pre-built patch constraints.

Your tasks:
1. Identify the SINGLE function that contains the root cause of the bug.
2. Produce a complete analysis document to guide the Agent Coder that will fix it.

Respond ONLY with valid JSON matching this exact schema — no prose, no markdown fences:
{
  "file":               "<relative file path>",
  "function":           "<function name>",
  "line":               <integer line number>,
  "root_cause":         "<one-sentence explanation of the root cause>",
  "confidence":         <float 0.0-1.0>,
  "problem_summary":    "<2-3 sentences. Format strictly: Comportement observé: [X]. Comportement attendu: [Y]. Condition de déclenchement: [Z].>",
  "code_context":       "<the provided code snippet with # BUG: or # PROBLÈME: inline comments added on the buggy lines — max 20 lines total>",
  "patch_constraints":  {
    "scope":            "<confirm or refine: modify only function() in file>",
    "preserve_tests":   [<list of test file paths that must not break — keep the pre-built list, add any you identify>],
    "forbidden_files":  [<list of file paths the Coder must not touch — keep the pre-built list>],
    "style_hint":       "<confirm or refine the detected code style conventions>"
  },
  "expected_behavior":  "<1-2 sentences describing what the function should do correctly after the patch>",
  "fallback_locations": [
    {"file": "<path>", "function": "<name>", "reason": "<why this caller or callee might be the actual root cause>"}
  ]
}
Do not omit any field. Use empty string "" or empty list [] as defaults when uncertain.
"""

_USER_TEMPLATE = """\
## Bug Ticket
ID:          {ticket_id}
Title:       {title}
Description: {description}
Severity:    {severity}
Component:   {component}

## Merge-Request Diff
```
{mr_diff}
```

## Candidate Functions ({n_candidates} candidates)
{candidates_block}

## Code Context (lines around the suspected bug location)
```
{code_context}
```

## Patch Constraints (pre-built — validate and refine if needed)
{patch_constraints_json}

Identify the root cause and respond with the complete JSON schema described.
All 13 fields are required. Use "" or [] for fields you cannot determine.
"""

_REFLEXION_TEMPLATE = """\
Your previous analysis returned low confidence ({confidence:.2f}).

Previous answer:
{prev_json}

## Expanded context — graph neighbours of the candidate

{expanded_block}

Re-analyse with this additional context.
Is the root cause in one of the neighbouring functions instead?

Respond again with the COMPLETE JSON schema (all 13 fields) — no prose, no markdown fences.
Preserve or improve all fields from the previous answer; update only what the new context changes.
"""


# ── Context builders ───────────────────────────────────────────────────────────


def _format_candidate(fn: dict, idx: int) -> str:
    callers = ", ".join(fn.get("callers") or []) or "none"
    callees = ", ".join(fn.get("callees") or []) or "none"
    source = fn.get("source", "").strip()
    lang = fn.get("language", "")
    return (
        f"### Candidate {idx + 1}: {fn.get('function')} "
        f"[{fn.get('file')}:{fn.get('start_line', '?')}]\n"
        f"Language: {lang}\n"
        f"Callers : [{callers}]\n"
        f"Callees : [{callees}]\n"
        f"```{lang}\n{source}\n```\n"
    )


def _build_main_prompt(
    state: dict,
    code_ctx: str = "",
    patch_constraints: Optional[Dict] = None,
) -> str:
    ticket   = state.get("ticket", {})
    mr_diff  = state.get("mr_diff", "")
    contexts = state.get("rag_contexts", [])

    candidates_block = "\n".join(
        _format_candidate(fn, i) for i, fn in enumerate(contexts)
    )

    # Serialise patch_constraints to a JSON string safe for .format() substitution.
    pc_json = json.dumps(patch_constraints or {}, indent=2, ensure_ascii=False)

    user_msg = _USER_TEMPLATE.format(
        ticket_id=ticket.get("id", "N/A"),
        title=ticket.get("title", ""),
        description=ticket.get("description", ""),
        severity=ticket.get("severity", ""),
        component=ticket.get("component", ""),
        mr_diff=mr_diff,
        n_candidates=len(contexts),
        candidates_block=candidates_block,
        code_context=code_ctx or "(no code context available)",
        patch_constraints_json=pc_json,
    )
    return user_msg


def _build_reflexion_prompt(prev_result: dict, expanded_fns: List[dict]) -> str:
    expanded_block = "\n".join(
        _format_candidate(fn, i) for i, fn in enumerate(expanded_fns)
    )
    return _REFLEXION_TEMPLATE.format(
        confidence=prev_result.get("confidence", 0.0),
        prev_json=json.dumps(prev_result, indent=2),
        expanded_block=expanded_block,
    )


# ── Ollama call ────────────────────────────────────────────────────────────────


def _call_ollama(user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Call the local Ollama instance with the structured-output format flag.
    Returns the raw response text.
    """
    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.1},   # low temperature for deterministic output
    )
    return response["message"]["content"]


# ── JSON parser ────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"file", "function", "line", "root_cause", "confidence"}


def _parse_llm_json(raw: str) -> Optional[Dict]:
    """
    Parse JSON from LLM output.  Tries strict parse first, then a regex
    extraction fallback in case the model wrapped the JSON in prose/fences.
    """
    # 1. Direct parse.
    try:
        data = json.loads(raw.strip())
        if _REQUIRED_KEYS <= set(data.keys()):
            return data
    except json.JSONDecodeError:
        pass

    # 2. Extract first JSON object from text.
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if _REQUIRED_KEYS <= set(data.keys()):
                return data
        except json.JSONDecodeError:
            pass

    return None


# ── Reflexion helper ───────────────────────────────────────────────────────────


def _expand_graph_neighbours(
    candidate_function: str,
    candidate_file: str,
    graph_data: dict,
    all_functions: List[dict],
    max_neighbours: int = 5,
) -> List[dict]:
    """
    Reconstruct the NetworkX graph and collect the direct predecessor + successor
    functions of the candidate.  Returns up to max_neighbours function dicts
    from all_functions.
    """
    try:
        G: nx.DiGraph = nx.node_link_graph(graph_data)
    except Exception:
        return []

    # Find the canonical node ID for the candidate.
    candidate_id = f"{candidate_file}::{candidate_function}"
    if candidate_id not in G:
        # Try a looser match by function name alone.
        matches = [n for n in G.nodes if n.endswith(f"::{candidate_function}")]
        if not matches:
            return []
        candidate_id = matches[0]

    neighbour_ids = set(G.predecessors(candidate_id)) | set(G.successors(candidate_id))

    # Map canonical IDs back to all_functions dicts.
    id_to_fn = {
        f"{fn['file']}::{fn['function']}": fn
        for fn in all_functions
    }
    neighbours: List[dict] = []
    for nid in list(neighbour_ids)[:max_neighbours]:
        fn = id_to_fn.get(nid)
        if fn:
            neighbours.append(fn)

    return neighbours


# ── Pre-LLM deterministic helpers ─────────────────────────────────────────────


def extract_code_context(file_path: str, line: int, window: int = 10) -> str:
    """
    Extract a numbered code snippet centred on *line* (1-based) from *file_path*.

    Returns *window* lines before and after *line* (max 2×window+1 lines total).
    Returns "" on any error (missing file, line=0, etc.) without raising.
    """
    if not file_path or not line:
        return ""
    try:
        p = Path(file_path)
        if not p.is_file():
            return ""
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, line - window - 1)          # 0-based inclusive
        end   = min(len(lines), line + window)      # 0-based exclusive
        numbered = [
            f"{start + i + 1:4d} | {ln}"
            for i, ln in enumerate(lines[start:end])
        ]
        return "\n".join(numbered)
    except Exception:
        return ""


def _find_test_files(repo_path: str, function_name: str) -> List[str]:
    """
    Search test files in *repo_path* that reference *function_name*.
    Looks in tests/, test/ directories and files matching test_*.py / *_test.py.
    Returns relative paths (forward slashes), capped at 10 results.
    """
    if not repo_path or not function_name:
        return []
    try:
        repo      = Path(repo_path)
        candidates: set = set()

        # By directory name
        for test_dir_name in ("tests", "test"):
            test_dir = repo / test_dir_name
            if test_dir.is_dir():
                candidates.update(test_dir.rglob("*.py"))

        # By filename pattern anywhere in the repo
        candidates.update(repo.rglob("test_*.py"))
        candidates.update(repo.rglob("*_test.py"))

        matches: List[str] = []
        for test_file in candidates:
            try:
                if function_name in test_file.read_text(encoding="utf-8", errors="replace"):
                    try:
                        rel = str(test_file.relative_to(repo)).replace("\\", "/")
                    except ValueError:
                        rel = str(test_file).replace("\\", "/")
                    matches.append(rel)
            except Exception:
                pass

        return sorted(matches)[:10]
    except Exception:
        return []


def _detect_style_hint(file_path: str) -> str:
    """
    Infer naming conventions and style from the source file.
    Returns a comma-separated hint string (e.g. "snake_case, pas de type hints, PEP8").
    """
    if not file_path:
        return "conventions existantes"
    try:
        p = Path(file_path)
        if not p.is_file():
            return "conventions existantes"
        content = p.read_text(encoding="utf-8", errors="replace")

        # snake_case vs camelCase — count function definitions
        snake = len(re.findall(r"\bdef [a-z][a-z0-9]*_[a-z0-9_]+\b", content))
        camel = len(re.findall(r"\bdef [a-z][a-z0-9]*[A-Z][a-zA-Z0-9]*\b", content))
        naming = "snake_case" if snake >= camel else "camelCase"

        # Type hints
        has_hints = bool(re.search(
            r":\s*(int|str|bool|float|list|dict|Optional|List|Dict|Any|None)\b", content
        ))
        type_hint = "type hints présents" if has_hints else "pas de type hints ajoutés"

        return f"{naming}, {type_hint}, PEP8"
    except Exception:
        return "conventions existantes"


def _get_forbidden_files(state: dict, component: str) -> List[str]:
    """
    Return files from bm25_files that are outside *component*.
    Capped at 10 entries.
    """
    if not component:
        return []
    repo_path = state.get("repo_path", "")
    comp      = component.strip().rstrip("/")
    forbidden: List[str] = []

    for entry in state.get("bm25_files", []):
        fpath = entry.get("file", "")
        if not fpath:
            continue
        try:
            rel = str(Path(fpath).relative_to(repo_path)).replace("\\", "/") if repo_path else fpath.replace("\\", "/")
        except ValueError:
            rel = fpath.replace("\\", "/")

        if not (rel.startswith(comp) or comp in rel):
            forbidden.append(rel)

    return sorted(forbidden)[:10]


def build_patch_constraints(state: dict, location: dict) -> Dict:
    """
    Build patch_constraints deterministically from state and the located function.

    Returns:
        {
            "scope":           str,
            "preserve_tests":  List[str],
            "forbidden_files": List[str],
            "style_hint":      str,
        }
    """
    file_path     = location.get("file", "")
    function_name = location.get("function", "")
    repo_path     = state.get("repo_path", "")
    ticket        = state.get("ticket", {})
    component     = ticket.get("component", "")

    scope          = f"Modifier uniquement {function_name}() dans {file_path}" if function_name else ""
    preserve_tests = _find_test_files(repo_path, function_name)
    forbidden      = _get_forbidden_files(state, component)
    style_hint     = _detect_style_hint(file_path)

    return {
        "scope":           scope,
        "preserve_tests":  preserve_tests,
        "forbidden_files": forbidden,
        "style_hint":      style_hint,
    }


def _validate_and_fill(result: dict, fallbacks: dict) -> dict:
    """
    Validate the 5 new fields in *result* and apply *fallbacks* where missing
    or malformed.  Logs a warning for every field repaired.  Never raises.
    """
    # problem_summary
    if not isinstance(result.get("problem_summary"), str) or not result["problem_summary"].strip():
        result["problem_summary"] = fallbacks.get("problem_summary", "")
        print("[phase4] problem_summary missing — fallback applied.", file=sys.stderr)

    # code_context
    if not isinstance(result.get("code_context"), str):
        result["code_context"] = fallbacks.get("code_context", "")
        print("[phase4] code_context missing — fallback applied.", file=sys.stderr)

    # patch_constraints — validate as a dict then sub-fields
    pc = result.get("patch_constraints")
    fb_pc = fallbacks.get("patch_constraints", {})
    if not isinstance(pc, dict):
        result["patch_constraints"] = fb_pc
        print("[phase4] patch_constraints missing — fallback applied.", file=sys.stderr)
    else:
        if not isinstance(pc.get("scope"), str):
            pc["scope"] = fb_pc.get("scope", "")
        if not isinstance(pc.get("preserve_tests"), list):
            pc["preserve_tests"] = fb_pc.get("preserve_tests", [])
        if not isinstance(pc.get("forbidden_files"), list):
            pc["forbidden_files"] = fb_pc.get("forbidden_files", [])
        if not isinstance(pc.get("style_hint"), str):
            pc["style_hint"] = fb_pc.get("style_hint", "")

    # expected_behavior
    if not isinstance(result.get("expected_behavior"), str) or not result["expected_behavior"].strip():
        result["expected_behavior"] = fallbacks.get("expected_behavior", "")
        print("[phase4] expected_behavior missing — fallback applied.", file=sys.stderr)

    # fallback_locations — must be a list of dicts with at least file + function
    fl = result.get("fallback_locations")
    if not isinstance(fl, list):
        result["fallback_locations"] = []
        print("[phase4] fallback_locations missing — reset to [].", file=sys.stderr)
    else:
        valid: List[dict] = []
        for item in fl:
            if isinstance(item, dict) and "file" in item and "function" in item:
                item.setdefault("reason", "")
                valid.append(item)
        result["fallback_locations"] = valid

    return result


# ── LangGraph node ─────────────────────────────────────────────────────────────


def phase_llm_confirm(state: dict) -> dict:
    """
    LangGraph node — Phase 4.

    Reads:  rag_contexts, ticket, mr_diff, repo_graph, all_functions, repo_path
    Writes: location, confidence

    location now contains 13 fields (8 existing + 5 new for the Agent Coder):
        file, function, line, root_cause, confidence, callers, callees, language,
        problem_summary, code_context, patch_constraints,
        expected_behavior, fallback_locations
    """
    contexts:      List[dict] = state.get("rag_contexts", [])
    all_functions: List[dict] = state.get("all_functions", [])
    graph_data:    dict       = state.get("repo_graph", {})
    model:         str        = state.get("llm_model", DEFAULT_MODEL)
    ticket:        dict       = state.get("ticket", {})

    _empty_constraints: Dict = {
        "scope": "", "preserve_tests": [], "forbidden_files": [], "style_hint": ""
    }

    if not contexts:
        fallback = {
            "file": "", "function": "", "line": 0,
            "root_cause": "No candidate functions identified.",
            "confidence": 0.0, "callers": [], "callees": [], "language": "",
            "problem_summary": "",
            "code_context": "",
            "patch_constraints": _empty_constraints,
            "expected_behavior": "",
            "fallback_locations": [],
        }
        return {**state, "location": fallback, "confidence": 0.0}

    # ── Pre-LLM: build deterministic context from top RAG candidate ───────────
    top_candidate = contexts[0]
    raw_code_ctx  = extract_code_context(
        top_candidate.get("file", ""),
        top_candidate.get("start_line", 0),
    )
    patch_constraints_prebuilt = build_patch_constraints(
        state,
        {
            "file":     top_candidate.get("file", ""),
            "function": top_candidate.get("function", ""),
            "line":     top_candidate.get("start_line", 0),
        },
    )

    # ── Call 1: main localisation ──────────────────────────────────────────────
    user_prompt = _build_main_prompt(state, raw_code_ctx, patch_constraints_prebuilt)
    try:
        raw    = _call_ollama(user_prompt, model=model)
        result = _parse_llm_json(raw)
    except Exception as exc:
        print(f"[phase4] LLM call failed: {exc}", file=sys.stderr)
        result = None

    if result is None:
        # Last-resort: pick top-1 RAG context deterministically.
        top = contexts[0]
        result = {
            "file":       top.get("file", ""),
            "function":   top.get("function", ""),
            "line":       top.get("start_line", 0),
            "root_cause": "LLM output unparseable — defaulting to top RAG hit.",
            "confidence": 0.3,
        }

    confidence = float(result.get("confidence", 0.0))

    # ── Reflexion: call 2 if confidence < threshold ────────────────────────────
    if confidence < REFLEXION_THRESHOLD and all_functions and graph_data:
        print(
            f"[phase4] Confidence {confidence:.2f} < {REFLEXION_THRESHOLD} — running Reflexion.",
            file=sys.stderr,
        )
        expanded = _expand_graph_neighbours(
            candidate_function=result.get("function", ""),
            candidate_file=result.get("file", ""),
            graph_data=graph_data,
            all_functions=all_functions,
        )

        if expanded:
            reflexion_prompt = _build_reflexion_prompt(result, expanded)
            try:
                raw2    = _call_ollama(reflexion_prompt, model=model)
                result2 = _parse_llm_json(raw2)
                if result2 is not None:
                    result     = result2
                    confidence = float(result.get("confidence", 0.0))
            except Exception as exc:
                print(f"[phase4] Reflexion LLM call failed: {exc}", file=sys.stderr)

    # ── Enrich with callers / callees / language from phase-2 data ────────────
    matched_fn = next(
        (
            fn for fn in all_functions
            if fn.get("function") == result.get("function")
            and fn.get("file") == result.get("file")
        ),
        None,
    )
    result["callers"]  = (matched_fn or {}).get("callers")  or []
    result["callees"]  = (matched_fn or {}).get("callees")  or []
    result["language"] = (matched_fn or {}).get("language") or ""

    # Guarantee all existing required keys are present.
    result.setdefault("file", "")
    result.setdefault("function", "")
    result.setdefault("line", 0)
    result.setdefault("root_cause", "")
    result.setdefault("confidence", confidence)
    result.setdefault("language", "")

    # ── Post-LLM: re-build deterministic fields for the actual location ────────
    # The LLM may have selected a different file/line than the top RAG candidate.
    actual_code_ctx = (
        extract_code_context(result.get("file", ""), result.get("line", 0))
        or raw_code_ctx
    )
    actual_constraints = build_patch_constraints(
        state,
        {
            "file":     result.get("file", ""),
            "function": result.get("function", ""),
            "line":     result.get("line", 0),
        },
    )

    # Fallback values for the 5 new fields.
    title   = ticket.get("title", "")
    desc    = ticket.get("description", "")
    summary_fb = f"{title} — {desc[:200]}".strip(" —") if (title or desc) else ""

    llm_fallbacks: Dict = {
        "problem_summary":    summary_fb,
        "code_context":       actual_code_ctx,
        "patch_constraints":  actual_constraints,
        "expected_behavior":  f"Corriger le bug décrit dans : {result.get('root_cause', '')}",
        "fallback_locations": [],
    }
    result = _validate_and_fill(result, llm_fallbacks)

    return {**state, "location": result, "confidence": confidence}

def normalize_path(file_path: str, repo_path: str) -> str:
    abs_path = os.path.abspath(os.path.join(repo_path, file_path))
    rel = os.path.relpath(abs_path, repo_path).replace("\\", "/")
    # Garde le chemin original si la normalisation produit des ../
    if rel.startswith(".."):
        return file_path.replace("\\", "/")
    return rel
