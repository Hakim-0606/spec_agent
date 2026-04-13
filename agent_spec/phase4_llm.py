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
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

import networkx as nx

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
REFLEXION_THRESHOLD = 0.7

# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert software engineer specialised in bug localisation for an automated repair pipeline.
Your output is consumed directly by an Agent Coder that will write the fix — precision is critical.

## What you receive
- A bug ticket (ID, title, description, severity, component)
- A merge-request diff (the change that introduced or exposed the bug)
- Up to 3 candidate functions with full source, ranked by relevance (Candidate 1 = highest)
- A code snippet centred on the most likely bug line
- Pre-built patch constraints (scope, test files, forbidden files, style)

## Reasoning steps (follow in order — do NOT output this section)
1. Read the diff: +/- lines are the strongest signal. Which candidate does the diff touch?
2. Read that candidate's source. Find the exact line(s) where the logic is wrong.
3. Check callers and callees: could the bug be one level up (bad input) or down (broken helper)?
4. Assign confidence using the calibration scale below.
5. MISSING FILE CHECK — if the error is ImportError / ModuleNotFoundError / missing module:
   a. Identify the EXISTING file that contains the bad import or reference (this goes in "file").
   b. Identify the EXACT line of the bad import/include call (this goes in "line").
   c. List every file that must be CREATED to fix the error in "missing_files".
   d. For each missing file, provide a "template" with the minimal valid content the Coder must write.
   Rule: "file" must ALWAYS point to an EXISTING file in the repo — never to the missing file.

   CONCRETE EXAMPLE (ModuleNotFoundError: No module named 'apps.services.urls'):
   WRONG → file = "apps/services/urls.py"   ← this file does NOT EXIST, never put it in "file"
   RIGHT → file = "fiber_crm/urls.py"        ← this EXISTING file has: include('apps.services.urls')
           line = <line number of include(...)>
           missing_files = [{"path": "apps/services/urls.py", "template": "..."}]
   If the caller file appears in your fallback_locations, that is your "file" — move it there.

## Confidence calibration
0.90 – 1.00 : Buggy line visible in both the diff and the function source.
0.70 – 0.89 : Strong match between diff and candidate; exact line is inferred.
0.50 – 0.69 : Plausible match; list fallback_locations for the Coder.
Below 0.50  : Evidence weak — reflexion will trigger a second analysis automatically.

## Output — respond ONLY with valid JSON, no prose, no markdown fences
{
  "file":             "<relative path of the EXISTING file that contains the root cause — forward slashes>",
  "function":         "<exact function or method name as it appears in the source — must exist in 'file'>",
  "line":             <integer: 1-based line number of the root cause inside 'file'; 0 if unknown>,
  "root_cause":       "<one precise sentence: WHAT is wrong, WHERE it is, and WHY it breaks — e.g. 'include(apps.services.urls) in fiber_crm/urls.py references a module that does not exist'>",
  "confidence":       <float 0.0–1.0 calibrated to the scale above>,
  "problem_summary":  "Observed behaviour: [what the system currently does wrong]. Expected behaviour: [what it should do]. Trigger condition: [when or how the bug manifests].",
  "code_context":     "<copy the exact lines around the bug from 'file'; append '# BUG: <short reason>' on each buggy line — 25 lines max>",
  "patch_constraints": {
    "scope":           "<exact instruction: 'In <file>, fix line <N> — <what to change>. If missing_files is non-empty, create each listed file with the provided template.'>",
    "preserve_tests":  [<test file paths that reference the buggy function — keep pre-built list and add any you identify>],
    "forbidden_files": [<files the Coder must NOT touch — keep pre-built list>],
    "style_hint":      "<naming convention + type hints + formatting rules observed in 'file'>"
  },
  "expected_behavior": "<1-2 sentences: exact state the system must reach after the fix — e.g. 'The dev server starts without ImportError; all existing URL routes remain functional'>",
  "missing_files": [
    {
      "path":     "<relative path of the file to CREATE — forward slashes>",
      "reason":   "<why this file must be created — e.g. referenced by include() but absent from repo>",
      "template": "<minimal valid content the Coder must write verbatim — include all required imports, class/function stubs, and urlpatterns if applicable>"
    }
  ],
  "fallback_locations": [
    {"file": "<path>", "function": "<name>", "reason": "<why this caller or callee might also hold a root cause>"}
  ]
}
Rules:
- Do not omit any field. Use "" or [] as defaults when uncertain.
- "line" must be an integer (not a string).
- "confidence" must be a float (not a string).
- "file" must be an EXISTING file — never set it to a file listed in "missing_files".
- SELF-CHECK before output: if "file" == any path in missing_files[].path, you violated the rule — fix it.
- All file paths must use forward slashes and be relative to the repo root.
- "missing_files" must be [] when no file needs to be created.
- "template" in missing_files must be complete enough for the Coder to write the file without guessing.
- "patch_constraints.scope" must reference the "file" value, NOT a file in missing_files.
"""

_USER_TEMPLATE = """\
## Bug Ticket
ID:          {ticket_id}
Title:       {title}
Description: {description}
Severity:    {severity}
Component:   {component}

## Merge-Request Diff  ← PRIMARY SIGNAL — focus on + and - lines first
```diff
{mr_diff}
```

## Candidate Functions ({n_candidates} ranked by relevance — Candidate 1 is the strongest match)
{candidates_block}

## Code Snippet (centred on the most likely bug line — copy this verbatim into "code_context" with # BUG: annotations)
```
{code_context}
```

## Patch Constraints (pre-built — confirm or refine each sub-field)
{patch_constraints_json}

---
Return the complete JSON (all 13 fields). Use "" or [] for any field you cannot determine.
Do not wrap the JSON in markdown fences or add any prose outside the JSON object.
"""

_REFLEXION_TEMPLATE = """\
Your previous analysis returned low confidence ({confidence:.2f}).
The callers and callees of the candidate function are provided below for deeper inspection.

## Previous answer
{prev_json}

## Expanded context — direct callers and callees of the candidate
{expanded_block}

## Re-analysis — answer these three questions before updating the JSON
1. CALLER bug? Does a caller pass wrong arguments or call the function in an invalid state?
2. CALLEE bug? Does the candidate rely on a helper that returns incorrect data?
3. Same function? Is the root cause still in the original candidate, now confirmed with more context?

Update ONLY the fields that the expanded context changes.
Raise "confidence" only if the new evidence genuinely resolves the ambiguity.
Preserve all other fields from the previous answer unchanged.

Respond with the COMPLETE JSON (all 13 fields) — no prose, no markdown fences.
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
    tool_search_results: Optional[List[dict]] = None,
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

    # Append project structure summary if available (Phase 0 output).
    project_structure = state.get("project_structure", {})
    if project_structure:
        from .phase0_workspace import _format_structure_for_prompt
        struct_summary = _format_structure_for_prompt(project_structure, max_files=50)
        user_msg += f"\n\n## Structure complète du projet\n{struct_summary}\n"

    # Append tool_search_results section if available (Phase 3.5 output).
    if tool_search_results:
        lines = ["\n## Occurrences trouvées dans le repo"]
        for r in tool_search_results[:5]:
            lines.append(f"{r['file']}:{r['line']} → {r['content']}")
        user_msg += "\n".join(lines) + "\n"

    # Append extra_context from the orchestrator (high-value signals).
    extra_context: dict = state.get("extra_context") or {}
    if extra_context:
        lines = ["\n## Additional Context from Orchestrator"]

        if extra_context.get("error_trace"):
            lines.append(f"\n### Stack Trace  ← USE THIS to pinpoint the exact crash line")
            lines.append("```")
            lines.append(extra_context["error_trace"][:2000])
            lines.append("```")

        if extra_context.get("affected_files"):
            lines.append(f"\n### Affected Files (pre-identified)")
            for f in extra_context["affected_files"][:10]:
                lines.append(f"  - {f}")

        if extra_context.get("commit_sha"):
            lines.append(f"\n### Commit that introduced the bug: {extra_context['commit_sha']}")

        if extra_context.get("retry_feedback"):
            lines.append(f"\n### Coder Feedback (previous fix failed)  ← IMPORTANT: adjust your analysis")
            lines.append(extra_context["retry_feedback"][:500])

        if extra_context.get("priority_hints"):
            lines.append(f"\n### Priority Areas")
            for h in extra_context["priority_hints"][:5]:
                lines.append(f"  - {h}")

        if extra_context.get("related_issues"):
            lines.append(f"\n### Related Issues/MRs: {', '.join(str(x) for x in extra_context['related_issues'])}")

        # Any other unknown fields — append as-is.
        known = {"error_trace", "affected_files", "commit_sha", "retry_feedback", "priority_hints", "related_issues"}
        for k, v in extra_context.items():
            if k not in known:
                lines.append(f"\n### {k}")
                lines.append(str(v)[:300])

        user_msg += "\n".join(lines) + "\n"

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
    return response.message.content


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

    # 2. Extract first complete JSON object from text (handles nested braces).
    match = re.search(r"\{.*\}", raw, re.DOTALL)
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
    Validate the new fields in *result* and apply *fallbacks* where missing
    or malformed.  Logs a warning for every field repaired.  Never raises.
    """
    # problem_summary
    if not isinstance(result.get("problem_summary"), str) or not result["problem_summary"].strip():
        result["problem_summary"] = fallbacks.get("problem_summary", "")
        logger.warning("[phase4] problem_summary missing — fallback applied.")

    # code_context
    if not isinstance(result.get("code_context"), str):
        result["code_context"] = fallbacks.get("code_context", "")
        logger.warning("[phase4] code_context missing — fallback applied.")

    # patch_constraints — validate as a dict then sub-fields
    pc = result.get("patch_constraints")
    fb_pc = fallbacks.get("patch_constraints", {})
    if not isinstance(pc, dict):
        result["patch_constraints"] = fb_pc
        logger.warning("[phase4] patch_constraints missing — fallback applied.")
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
        logger.warning("[phase4] expected_behavior missing — fallback applied.")

    # missing_files — list of {path, reason, template}
    mf = result.get("missing_files")
    if not isinstance(mf, list):
        result["missing_files"] = []
        logger.warning("[phase4] missing_files absent — reset to [].")
    else:
        valid_mf: List[dict] = []
        for item in mf:
            if isinstance(item, dict) and "path" in item:
                item.setdefault("reason", "")
                item.setdefault("template", "")
                valid_mf.append(item)
        result["missing_files"] = valid_mf

    # fallback_locations — must be a list of dicts with at least file + function
    fl = result.get("fallback_locations")
    if not isinstance(fl, list):
        result["fallback_locations"] = []
        logger.warning("[phase4] fallback_locations missing — reset to [].")
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
    contexts:             List[dict] = state.get("rag_contexts", [])
    all_functions:        List[dict] = state.get("all_functions", [])
    ast_functions:        List[dict] = state.get("ast_functions", [])
    tool_search_results:  List[dict] = state.get("tool_search_results", [])
    graph_data:           dict       = state.get("repo_graph", {})
    model:                str        = state.get("llm_model", DEFAULT_MODEL)
    ticket:               dict       = state.get("ticket", {})

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

    # Prefer source_real (Phase 3.5) over re-reading the file — better quality
    # as it includes real line numbers and the ±5-line context window.
    _top_ast_fn = next(
        (fn for fn in ast_functions
         if fn.get("function") == top_candidate.get("function")
         and fn.get("file") == top_candidate.get("file")),
        None,
    )
    _top_source_real = (_top_ast_fn or {}).get("source_real", "")
    raw_code_ctx = _top_source_real or extract_code_context(
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
    user_prompt = _build_main_prompt(
        state, raw_code_ctx, patch_constraints_prebuilt, tool_search_results
    )
    try:
        raw    = _call_ollama(user_prompt, model=model)
        result = _parse_llm_json(raw)
    except Exception as exc:
        logger.error("[phase4] LLM call failed: %s", exc)
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
        logger.info(
            "[phase4] Confidence %.2f < %.2f — running Reflexion.",
            confidence, REFLEXION_THRESHOLD,
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
                logger.error("[phase4] Reflexion LLM call failed: %s", exc)

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

    # Fallback : deduce language from file extension when phase-2 had no match.
    if not result["language"]:
        _ext = Path(result.get("file", "")).suffix.lower()
        result["language"] = {
            ".py": "python",  ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
            ".go": "go",      ".rs": "rust",        ".cpp": "cpp",
            ".c":  "c",       ".cs": "c_sharp",     ".rb": "ruby",
            ".kt": "kotlin",  ".swift": "swift",
        }.get(_ext, "")

    # Guarantee all existing required keys are present.
    result.setdefault("file", "")
    result.setdefault("function", "")
    result.setdefault("line", 0)
    result.setdefault("root_cause", "")
    result.setdefault("confidence", confidence)
    result.setdefault("language", "")

    # ── Post-LLM: re-build deterministic fields for the actual location ────────
    # The LLM may have selected a different file/line than the top RAG candidate.
    # Prefer source_real from ast_functions (Phase 3.5) for the LLM-identified fn.
    _llm_ast_fn = next(
        (fn for fn in ast_functions
         if fn.get("function") == result.get("function")
         and fn.get("file") == result.get("file")),
        None,
    )
    _llm_source_real = (_llm_ast_fn or {}).get("source_real", "")
    actual_code_ctx = (
        _llm_source_real
        or extract_code_context(result.get("file", ""), result.get("line", 0))
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

    # Fallback values for all new fields.
    title   = ticket.get("title", "")
    desc    = ticket.get("description", "")
    summary_fb = f"{title} — {desc[:200]}".strip(" —") if (title or desc) else ""

    llm_fallbacks: Dict = {
        "problem_summary":    summary_fb,
        "code_context":       actual_code_ctx,
        "patch_constraints":  actual_constraints,
        "expected_behavior":  f"Corriger le bug décrit dans : {result.get('root_cause', '')}",
        "missing_files":      [],
        "fallback_locations": [],
    }
    result = _validate_and_fill(result, llm_fallbacks)

    # ── Auto-correct: if 'file' is a missing file, swap to the caller ─────────
    _missing_paths = {
        mf.get("path", "").replace("\\", "/")
        for mf in result.get("missing_files", [])
    }
    _result_file = result.get("file", "").replace("\\", "/")

    if _result_file and _result_file in _missing_paths:
        # Find the first fallback that is NOT itself missing
        _caller_file = ""
        _caller_fn   = ""
        for _fb in result.get("fallback_locations", []):
            _fb_file = _fb.get("file", "").replace("\\", "/")
            if _fb_file and _fb_file not in _missing_paths:
                _caller_file = _fb_file
                _caller_fn   = _fb.get("function", result.get("function", ""))
                break

        if _caller_file:
            logger.warning(
                "[phase4] 'file' was a missing file (%s) — auto-corrected to caller: %s",
                _result_file, _caller_file,
            )
            result["file"] = _caller_file
            if _caller_fn:
                result["function"] = _caller_fn

            # Rebuild code_context from the corrected caller file
            _repo_path  = state.get("repo_path", "")
            _abs_caller = str(Path(_repo_path) / _caller_file) if _repo_path else _caller_file
            _new_ctx    = extract_code_context(_abs_caller, result.get("line", 0))
            if _new_ctx:
                result["code_context"] = _new_ctx

            # Rebuild patch_constraints with the corrected caller file
            _corrected_pc = build_patch_constraints(
                state,
                {"file": _abs_caller, "function": result["function"], "line": result.get("line", 0)},
            )
            # Describe the full fix: modify caller + create missing file(s)
            _missing_list = ", ".join(mf["path"] for mf in result.get("missing_files", []))
            _corrected_pc["scope"] = (
                f"In {_caller_file}, fix the import/include at line {result.get('line', '?')}. "
                f"Also create: {_missing_list} using the provided templates."
            )
            result["patch_constraints"] = _corrected_pc

            # Refresh language from new file extension if still empty
            if not result.get("language"):
                _ext2 = Path(_caller_file).suffix.lower()
                result["language"] = {
                    ".py": "python",  ".js": "javascript", ".ts": "typescript",
                    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
                    ".go": "go",      ".rs": "rust",        ".cpp": "cpp",
                    ".c":  "c",       ".cs": "c_sharp",     ".rb": "ruby",
                    ".kt": "kotlin",  ".swift": "swift",
                }.get(_ext2, "")

    return {**state, "location": result, "confidence": confidence}
