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
You are an expert software engineer specialised in bug localisation.

## What you receive
- A bug ticket (title, description, severity)
- A merge-request diff (the change that introduced or exposed the bug)
- Ranked candidate functions with full source (Candidate 1 = most likely)
- A code snippet around the most likely bug line
- Tool search results (grep hits for key symbols)
- Module headers: the first ~60 lines of each top-ranked file — import statements,
  module-level variables, class definitions. USE THESE to spot circular imports,
  missing modules, wrong import paths, and module-level bugs.

## Reasoning steps (think through these — do NOT write them in your output)
1. Read the diff: +/- lines are the strongest signal.
2. Read each candidate's source. Find the exact line where the logic breaks.
3. Check callers and callees for one-level-up / one-level-down bugs.
4. For ImportError / ModuleNotFoundError: identify the EXISTING file that contains
   the bad import (never point to the missing file itself).
5. Assign confidence using the scale below.

## Confidence calibration
0.90–1.00 : Buggy line visible in both diff and function source.
0.70–0.89 : Strong diff↔candidate match; exact line inferred.
0.50–0.69 : Plausible match.
Below 0.50 : Weak evidence.

## Output format — YOUR ENTIRE RESPONSE must be this JSON and nothing else
{{"file": "<relative path, forward slashes, MUST exist in repo>", "function": "<exact function name>", "line": <integer, 0 if unknown>, "root_cause": "<one precise sentence>", "confidence": <float 0.0-1.0>}}

STRICT RULES:
- Respond with ONLY the JSON object. First char = {{. Last char = }}.
- No markdown fences, no prose, no explanation before or after.
- "line" is an integer. "confidence" is a float. NEVER use strings for these.
- "file" must be an EXISTING file in the repo. Never pick migrations/ or auto-generated files.
"""

_USER_TEMPLATE = """\
## Bug Ticket
ID:          {ticket_id}
Title:       {title}
Description: {description}
Severity:    {severity}
Component:   {component}

## Merge-Request Diff  ← PRIMARY SIGNAL — focus on + and - lines first
{mr_diff}

## Candidate Functions ({n_candidates} ranked by relevance — Candidate 1 is the strongest match)
{candidates_block}

## Code Snippet (centred on the most likely bug line)
{code_context}

## Pre-built context (scope, style, constraints)
{patch_constraints_json}

Respond with ONLY the 5-field JSON object described in your instructions.
"""

_REFLEXION_TEMPLATE = """\
Your previous analysis returned low confidence ({confidence:.2f}).
Review the expanded caller/callee context below and update your answer.

## Previous answer
{prev_json}

## Expanded context — direct callers and callees of the candidate
{expanded_block}

Respond with the SAME 5-field JSON, updating only the fields the new evidence changes.
Raise "confidence" only if the new evidence genuinely resolves the ambiguity.
First char = {{. Last char = }}. No prose.
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

    # Prop 1: Inject module headers (import blocks) for top BM25 files.
    # This gives the LLM direct visibility into import chains, circular imports,
    # and module-level code — critical for ImportError / ModuleNotFoundError bugs.
    file_import_blocks: dict = state.get("file_import_blocks", {})
    if file_import_blocks:
        # Show top-5 BM25 files in BM25 rank order (preserves relevance signal).
        bm25_files: List[dict] = state.get("bm25_files", [])
        ordered_paths = [e.get("file", "") for e in bm25_files if e.get("file")]
        # Fall back to dict insertion order if BM25 list is empty.
        if not ordered_paths:
            ordered_paths = list(file_import_blocks.keys())

        shown = 0
        header_lines = ["\n## Module headers — imports & module-level code (top BM25 files)"]
        header_lines.append(
            "← USE THIS to detect circular imports, missing modules, and wrong import paths"
        )
        for fpath in ordered_paths:
            block = file_import_blocks.get(fpath, "")
            if not block:
                continue
            try:
                from pathlib import Path as _Path
                rel = str(_Path(fpath).relative_to(state.get("repo_path", fpath)))
            except (ValueError, TypeError):
                rel = fpath
            rel = rel.replace("\\", "/")
            header_lines.append(f"\n### {rel}")
            header_lines.append(block)
            shown += 1
            if shown >= 5:
                break
        user_msg += "\n".join(header_lines) + "\n"

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


# ── LLM backend abstraction ────────────────────────────────────────────────────
#
# Set LM_STUDIO_URL=http://localhost:1234  → uses OpenAI-compatible API (LM Studio).
# Leave unset                              → uses Ollama (default, backward-compat).
#
# The model name is always read from the state / $OLLAMA_MODEL — same env var,
# same value works for both backends (pass the model name loaded in LM Studio).


def _chat_completion(
    messages:    list,
    model:       str,
    *,
    json_mode:   bool  = False,
    temperature: float = 0.1,
) -> str:
    """
    Single-turn chat completion, routing to LM Studio or Ollama.

    Args:
        messages    : OpenAI-style message list [{role, content}, …].
        model       : Model identifier (same value for both backends).
        json_mode   : Request JSON-only output (format="json" / response_format).
        temperature : Sampling temperature.

    Returns:
        Raw response text from the model.
    """
    lm_url = os.environ.get("LM_STUDIO_URL", "").rstrip("/")

    if lm_url:
        # ── LM Studio (OpenAI-compatible) ─────────────────────────────────────
        from openai import OpenAI
        base_url = lm_url if lm_url.endswith("/v1") else lm_url + "/v1"
        client   = OpenAI(base_url=base_url, api_key="lm-studio")
        kwargs: dict = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
        }
        # LM Studio only accepts "json_schema" or "text" — not "json_object".
        # The system prompt already instructs JSON output, so no response_format needed.
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    else:
        # ── Ollama ────────────────────────────────────────────────────────────
        import ollama
        kwargs = {
            "model":    model,
            "messages": messages,
            "options":  {"temperature": temperature},
        }
        if json_mode:
            kwargs["format"] = "json"
        resp = ollama.chat(**kwargs)
        return resp.message.content or ""


# ── Ollama calls ───────────────────────────────────────────────────────────────

# Auto-generated file patterns — fallback should never resolve to these.
_AUTO_GENERATED_PATTERNS = (
    "/migrations/", "\\migrations\\",
    "/__snapshots__/", "\\__snapshots__\\",
    "/generated/", "/proto_gen/", "/stubs/",
    "0001_initial", "0002_", "0003_",        # numbered Django migrations
)


def _is_auto_generated(file_path: str) -> bool:
    """Return True when *file_path* points to auto-generated code."""
    norm = file_path.replace("\\", "/")
    return any(pat.replace("\\", "/") in norm for pat in _AUTO_GENERATED_PATTERNS)


# Filenames that are almost never the source of an application bug.
_NOISE_FILENAMES: frozenset = frozenset({
    "manage.py",      # Django CLI entry-point
    "wsgi.py",        # WSGI adapter
    "asgi.py",        # ASGI adapter
    "conftest.py",    # pytest fixtures
    "setup.py",       # packaging
    "setup.cfg",
})


def _is_noise_candidate(file_path: str) -> bool:
    """
    Return True when a candidate file is almost certainly NOT the bug location.

    Excluded:
    - auto-generated files (migrations, proto stubs, etc.)
    - test files  (test_*.py / *_test.py / tests/ directory)
    - known entry-point / config files (manage.py, wsgi.py, …)
    - Django app-config stubs (apps.py contains only AppConfig, not business logic)
    """
    norm = file_path.replace("\\", "/")
    name = norm.rsplit("/", 1)[-1]

    if _is_auto_generated(file_path):
        return True

    # Test files
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if "/tests/" in norm or "/test/" in norm:
        return True

    # Known noise filenames
    if name in _NOISE_FILENAMES:
        return True

    # Django apps.py — contains only AppConfig boilerplate, never the bug source
    if name == "apps.py":
        return True

    return False


def _filter_candidates(contexts: List[dict]) -> List[dict]:
    """
    Remove noise candidates from the RAG context list, keeping at least 1.

    Filtered-out files are logged so the pipeline stays transparent.
    If filtering would leave 0 candidates, the original list is returned intact
    (better a noisy candidate than no candidate).
    """
    filtered = [ctx for ctx in contexts if not _is_noise_candidate(ctx.get("file", ""))]
    if not filtered:
        logger.warning(
            "[phase4] Pre-filter removed ALL %d candidates — keeping original list.",
            len(contexts),
        )
        return contexts
    removed = len(contexts) - len(filtered)
    if removed:
        noise_files = [
            ctx.get("file", "?") for ctx in contexts
            if _is_noise_candidate(ctx.get("file", ""))
        ]
        logger.info(
            "[phase4] Pre-filter removed %d noise candidate(s): %s",
            removed, noise_files,
        )
    return filtered


def _call_ollama(user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Main LLM call — 5-field schema (file, function, line, root_cause, confidence).
    Routes to LM Studio or Ollama depending on LM_STUDIO_URL.
    The 8 extended fields are always built deterministically after this call.
    """
    return _chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        model=model,
        json_mode=True,
        temperature=0.1,
    )


# ── JSON parser ────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"file", "function", "line", "root_cause", "confidence"}


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences that LLMs often insert."""
    text = text.strip()
    # Remove ```json or ``` at the start
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove ``` at the end
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first complete JSON object from *text* using balanced-brace
    scanning.  More reliable than a greedy regex when the model appends prose
    after the closing brace.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_types(data: dict) -> dict:
    """
    Coerce 'line' to int and 'confidence' to float — small models often return
    these as strings ("7" / "0.8") which breaks downstream consumers.
    """
    try:
        data["line"] = int(data["line"])
    except (TypeError, ValueError, KeyError):
        data["line"] = 0

    try:
        data["confidence"] = float(data["confidence"])
    except (TypeError, ValueError, KeyError):
        data["confidence"] = 0.0

    return data


def _parse_llm_json(raw: str) -> Optional[Dict]:
    """
    Parse JSON from LLM output. Pipeline:
      1. Strip markdown fences (```json ... ```)
      2. Direct json.loads on cleaned text
      3. Balanced-brace extraction then json.loads (handles leading/trailing prose)
    After a successful parse, coerce 'line' → int and 'confidence' → float.
    Logs the first 300 chars of raw output when all attempts fail.
    """
    if not raw:
        return None

    cleaned = _strip_fences(raw)

    # 1. Direct parse on cleaned text.
    try:
        data = json.loads(cleaned)
        if _REQUIRED_KEYS <= set(data.keys()):
            return _coerce_types(data)
    except json.JSONDecodeError:
        pass

    # 2. Balanced-brace extraction — handles prose before/after the JSON block.
    extracted = _extract_json_object(cleaned)
    if extracted:
        try:
            data = json.loads(extracted)
            if _REQUIRED_KEYS <= set(data.keys()):
                return _coerce_types(data)
        except json.JSONDecodeError:
            pass

    logger.warning(
        "[phase4] JSON parse failed. Raw output (first 300 chars): %r",
        raw[:300],
    )
    return None


# ── Deterministic fallback when LLM fails ─────────────────────────────────────

# Patterns that indicate a class / function *definition* line (not import/usage).
_DEFINITION_RE = re.compile(r"^\s*(class|def)\s+(\w+)")


def _module_path_to_file_candidates(module_dotted: str) -> List[str]:
    """Convert a dotted Python module name to candidate file paths."""
    slash = module_dotted.replace(".", "/")
    return [slash + ".py", slash + "/__init__.py"]


def _resolve_import_error_file(ticket: dict, repo_path: str) -> Optional[dict]:
    """
    If the ticket describes an ImportError / ModuleNotFoundError, try to resolve
    the module path to an actual file in the repo.

    Returns a location dict (confidence 0.55) on success, None otherwise.
    """
    title = ticket.get("title", "") or ""
    desc  = (ticket.get("description") or "")[:500]
    combined = f"{title} {desc}"

    # Collect candidate module dotted paths.
    module_names: List[str] = []

    p_import = re.compile(
        r"ImportError[^'\"]*from[^'\"]*['\"]([a-zA-Z0-9_.]+)['\"]", re.IGNORECASE
    )
    p_nomod = re.compile(r"No module named ['\"]([a-zA-Z0-9_.]+)['\"]", re.IGNORECASE)
    p_partial = re.compile(
        r"partially initialized module ['\"]([a-zA-Z0-9_.]+)['\"]", re.IGNORECASE
    )

    for pat in (p_import, p_nomod, p_partial):
        for m in pat.finditer(combined):
            mod = m.group(1).strip("'\" ")
            if mod and mod not in module_names:
                module_names.append(mod)

    if not module_names or not repo_path:
        return None

    repo = Path(repo_path)
    for mod in module_names:
        for candidate in _module_path_to_file_candidates(mod):
            full = repo / candidate
            if full.is_file() and not _is_auto_generated(str(full)):
                rel = candidate  # already relative
                logger.info(
                    "[phase4] ImportError fallback → module '%s' resolved to %s",
                    mod, rel,
                )
                return {
                    "file":       rel,
                    "function":   "_module_level_",
                    "line":       1,
                    "root_cause": (
                        f"Module '{mod}' referenced in the error is defined at "
                        f"{rel} — inspect imports and circular dependencies here."
                    ),
                    "confidence": 0.55,
                }

    return None


def _build_8_fields_deterministic(
    location:            dict,
    ticket:              dict,
    state:               dict,
    ast_functions:       List[dict],
    contexts:            List[dict],
    repo_path:           str,
) -> dict:
    """
    Build the 8 extended fields from code analysis — no LLM involved.
    Called AFTER the LLM returns the 5 core fields (file/function/line/root_cause/confidence).

    Returns a dict with:
        problem_summary, code_context, patch_constraints,
        expected_behavior, missing_files, fallback_locations
    """
    file_path  = location.get("file", "").replace("\\", "/")
    function   = location.get("function", "")
    line       = location.get("line", 0)
    root_cause = location.get("root_cause", "")
    title      = ticket.get("title", "")
    desc       = (ticket.get("description") or "")

    # ── 1. problem_summary — clean 1-2 sentence summary (no raw markdown) ───────
    def _clean_desc(text: str) -> str:
        """Strip markdown headings/bullets, return first 2 sentences."""
        text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)  # remove headings
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)        # remove bullets
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)                   # remove inline code
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2]).strip()

    desc_clean = _clean_desc(desc[:600]) if desc else ""
    if title and desc_clean:
        problem_summary = f"{title}. {desc_clean}"[:350]
    elif title:
        problem_summary = title
    else:
        problem_summary = desc_clean[:350]

    # ── 2. code_context — read actual file at identified line ─────────────────
    _abs = (
        str(Path(repo_path) / file_path)
        if repo_path and file_path and not Path(file_path).is_absolute()
        else file_path
    )
    _ast_fn = next(
        (fn for fn in ast_functions
         if fn.get("function") == function
         and fn.get("file", "").replace("\\", "/") == file_path),
        None,
    )
    code_context = (
        (_ast_fn or {}).get("source_real", "")
        or extract_code_context(_abs, line)
        or ""
    )

    # ── 3. patch_constraints — deterministic builder already in place ─────────
    patch_constraints = build_patch_constraints(
        state,
        {"file": _abs, "function": function, "line": line},
    )

    # ── 4. expected_behavior — derived from ticket title ──────────────────────
    if title:
        expected_behavior = f"Le projet fonctionne sans erreur : {title}"
    else:
        expected_behavior = "Le bug est corrigé et les tests existants passent."

    # ── 5. missing_files — from ImportError module resolution ─────────────────
    missing_files: List[dict] = []
    import_hit = _resolve_import_error_file(ticket, repo_path)
    if import_hit:
        hit_file = import_hit.get("file", "").replace("\\", "/")
        # If the LLM picked a different file and the import resolution found the
        # module file, log it — but don't override the LLM choice here (that's
        # handled later in auto-correct).  We do note it in fallback_locations.
        if hit_file and hit_file != file_path:
            logger.info(
                "[phase4] ImportError module resolved to %s (LLM chose %s)",
                hit_file, file_path,
            )

    # ── 6. fallback_locations — BM25 top candidates excluding chosen file ─────
    fallback_locations: List[dict] = []
    seen: set = set()
    for i, ctx in enumerate(contexts[:5]):
        ctx_file = ctx.get("file", "").replace("\\", "/")
        try:
            rel = str(Path(ctx_file).relative_to(repo_path)).replace("\\", "/")
        except (ValueError, TypeError):
            rel = ctx_file
        if rel and rel != file_path and rel not in seen:
            seen.add(rel)
            fallback_locations.append({
                "file":     rel,
                "function": ctx.get("function", ""),
                "reason":   f"BM25+embedding candidate #{i + 1}",
            })

    return {
        "problem_summary":    problem_summary,
        "code_context":       code_context,
        "patch_constraints":  patch_constraints,
        "expected_behavior":  expected_behavior,
        "missing_files":      missing_files,
        "fallback_locations": fallback_locations,
    }


def _build_deterministic_fallback(
    contexts:            List[dict],
    tool_search_results: List[dict],
    ticket:              dict,
    repo_path:           str,
) -> dict:
    """
    Build the best possible location without usable LLM output.

    Search order (highest confidence first):
    0. ImportError / ModuleNotFoundError module resolution — maps dotted module
       name from the ticket title to an actual file in the repo (confidence 0.55).
    1. tool_search_results — a *definition* line (class/def) in a non-generated file.
       These results were searched from ticket keywords, so a class/def hit is
       very likely the actual bug origin (e.g. `class Customer` in models.py).
    2. ast_functions from RAG contexts — ranked by the full pipeline, skip if
       file is auto-generated.
    3. Raw top RAG context — last resort.
    """
    # 0. ImportError / ModuleNotFoundError direct file resolution.
    import_hit = _resolve_import_error_file(ticket, repo_path)
    if import_hit:
        return import_hit

    # 1. Definition in tool_search_results.
    for hit in tool_search_results:
        content = hit.get("content", "")
        fpath   = hit.get("file", "")
        m       = _DEFINITION_RE.match(content)
        if m and not _is_auto_generated(fpath):
            kind    = m.group(1)   # "class" or "def"
            name    = m.group(2)
            lineno  = hit.get("line", 0)
            logger.info(
                "[phase4] Deterministic fallback → definition hit: %s %s at %s:%d",
                kind, name, fpath, lineno,
            )
            return {
                "file":       fpath,
                "function":   name,
                "line":       lineno,
                "root_cause": (
                    f"{kind} `{name}` found at {fpath}:{lineno} — "
                    "LLM failed; confirm this is the bug origin."
                ),
                "confidence": 0.35,
            }

    # 2. Top RAG context, skip auto-generated files.
    for ctx in contexts:
        fpath = ctx.get("file", "")
        if fpath and not _is_auto_generated(fpath):
            logger.info(
                "[phase4] Deterministic fallback → RAG context: %s::%s",
                fpath, ctx.get("function", ""),
            )
            return {
                "file":       fpath,
                "function":   ctx.get("function", ""),
                "line":       ctx.get("start_line", 0),
                "root_cause": "LLM output unparseable — deterministic fallback to best non-generated RAG hit.",
                "confidence": 0.25,
            }

    # 3. Absolute last resort.
    logger.warning("[phase4] Deterministic fallback: no usable context found.")
    return {
        "file": "", "function": "", "line": 0,
        "root_cause": "Unable to locate bug — no valid LLM output or search results.",
        "confidence": 0.0,
    }


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

    # Ensure file_path is always relative (scope + forbidden use it for display)
    try:
        rel_file = str(Path(file_path).relative_to(repo_path)).replace("\\", "/") if repo_path else file_path
    except ValueError:
        rel_file = file_path.replace("\\", "/")

    scope          = f"Modify only {function_name}() in {rel_file}" if function_name else f"Fix {rel_file}"
    preserve_tests = _find_test_files(repo_path, function_name)
    forbidden      = [f for f in _get_forbidden_files(state, component) if f != rel_file]
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
            if isinstance(item, dict) and item.get("path", "").strip():
                item.setdefault("reason", "")
                item.setdefault("template", "")
                valid_mf.append(item)
        result["missing_files"] = valid_mf

    # fallback_locations — must be a list of dicts with non-empty file + function
    fl = result.get("fallback_locations")
    if not isinstance(fl, list):
        result["fallback_locations"] = []
        logger.warning("[phase4] fallback_locations missing — reset to [].")
    else:
        valid: List[dict] = []
        for item in fl:
            if (isinstance(item, dict)
                    and item.get("file", "").strip()
                    and item.get("function", "").strip()):
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
    contexts:             List[dict] = _filter_candidates(state.get("rag_contexts", []))
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

    # ── Solution 2: ImportError hard override — bypass LLM entirely ──────────
    # When the ticket contains an ImportError/ModuleNotFoundError, the module
    # path directly maps to a file. This is 100 % deterministic and far more
    # reliable than asking a small LLM to guess. If the file exists on disk
    # we use it directly and skip the LLM call.
    _repo_path = state.get("repo_path", "")
    _import_hit = _resolve_import_error_file(ticket, _repo_path)
    if _import_hit and _import_hit.get("file"):
        logger.info(
            "[phase4] ImportError override — resolved to '%s', skipping LLM.",
            _import_hit["file"],
        )
        _import_hit["confidence"] = 0.75
        _import_hit.setdefault("function", "_module_level_")
        _import_hit.setdefault("line", 1)
        _import_hit.setdefault("root_cause",
            f"Circular import or missing module: "
            f"{_import_hit['file'].replace('/', '.').removesuffix('.py')}"
        )
        # Build language from extension.
        _imp_ext = Path(_import_hit["file"]).suffix.lower()
        _import_hit["language"] = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
            ".go": "go", ".rs": "rust", ".rb": "ruby",
        }.get(_imp_ext, "")
        _import_hit["callers"] = []
        _import_hit["callees"] = []
        # Build the 8 extended fields deterministically.
        _imp_extended = _build_8_fields_deterministic(
            _import_hit, ticket, state, ast_functions, contexts, _repo_path
        )
        _import_hit.update(_imp_extended)
        _import_hit["coder_instructions"] = _build_coder_instructions(_import_hit)
        _import_hit["task_file"] = _write_task_md(_import_hit, state)
        return {**state, "location": _import_hit, "confidence": _import_hit["confidence"]}

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

    # ── Single LLM call — 5-field schema (file/function/line/root_cause/confidence)
    # The 8 extended fields are ALWAYS built deterministically after this call.
    # This eliminates "unparseable output" errors caused by complex 13-field schemas.
    user_prompt = _build_main_prompt(
        state, raw_code_ctx, patch_constraints_prebuilt, tool_search_results
    )
    result: Optional[Dict] = None
    try:
        raw    = _call_ollama(user_prompt, model=model)
        result = _parse_llm_json(raw)
        if result is not None:
            logger.info(
                "[phase4] LLM localisation succeeded — file=%s confidence=%.2f",
                result.get("file", "?"), result.get("confidence", 0.0),
            )
    except Exception as exc:
        logger.error("[phase4] LLM call failed: %s", exc)

    if result is None:
        # ── Deterministic fallback ────────────────────────────────────────────
        # LLM returned unparseable output or raised an exception.
        # Use tool_search_results (definition hits) → best non-generated RAG context.
        logger.warning("[phase4] LLM output unparseable — using deterministic fallback.")
        result = _build_deterministic_fallback(
            contexts, tool_search_results, ticket, state.get("repo_path", "")
        )

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
    # ── Build the 8 extended fields deterministically ─────────────────────────
    # These fields are NEVER asked from the LLM — always derived from code.
    # This guarantees they are always present and always correct.
    extended = _build_8_fields_deterministic(
        result, ticket, state, ast_functions, contexts, state.get("repo_path", "")
    )
    result.update(extended)

    # ── Auto-correct: if 'file' does not exist on disk, swap to the caller ──────
    # Triggers in two cases:
    #   (A) LLM declared it in missing_files  — classic violation
    #   (B) LLM left missing_files=[] but the file simply isn't on disk
    _repo_path_ac = state.get("repo_path", "")
    _missing_paths = {
        mf.get("path", "").replace("\\", "/")
        for mf in result.get("missing_files", [])
    }
    _result_file = result.get("file", "").replace("\\", "/")

    # Disk-existence check (case B): add to _missing_paths so the correction
    # block below fires, and synthesise a missing_files entry with an empty template.
    if _result_file and _result_file not in _missing_paths:
        _abs_check = Path(_repo_path_ac) / _result_file if _repo_path_ac else Path(_result_file)
        if not _abs_check.is_file():
            logger.warning(
                "[phase4] 'file' (%s) not found on disk — treating as missing file (LLM forgot missing_files)",
                _result_file,
            )
            _missing_paths.add(_result_file)
            # Synthesise a missing_files entry so the Coder knows to create it.
            existing_mf = result.get("missing_files") or []
            if not any(
                (mf.get("path") or "").replace("\\", "/") == _result_file
                for mf in existing_mf
            ):
                existing_mf.append({
                    "path":     _result_file,
                    "reason":   f"File referenced but absent from repo — auto-detected by phase 4",
                    "template": result.get("code_context", ""),  # use annotated snippet as seed
                })
                result["missing_files"] = existing_mf

    if _result_file and _result_file in _missing_paths:
        # Find the first fallback that is NOT itself missing.
        # Search order: fallback_locations → ast_functions from state.
        _caller_file = ""
        _caller_fn   = ""

        for _fb in result.get("fallback_locations", []):
            _fb_file = _fb.get("file", "").replace("\\", "/")
            if _fb_file and _fb_file not in _missing_paths:
                _caller_file = _fb_file
                _caller_fn   = _fb.get("function", result.get("function", ""))
                break

        if not _caller_file:
            # Second pass: search for the file that actually REFERENCES the missing module.
            # Priority: file whose source contains the missing module name (grep-style).
            _missing_module = _result_file.replace("/", ".").removesuffix(".py")  # apps/services/urls.py → apps.services.urls
            _candidates_af  = list(state.get("ast_functions", [])) + list(state.get("all_functions", []))
            _seen_af: set   = set()

            for _af in _candidates_af:
                _af_file = _af.get("file", "").replace("\\", "/")
                if not _af_file or _af_file in _missing_paths or _af_file in _seen_af:
                    continue
                _seen_af.add(_af_file)
                _af_abs = Path(_repo_path_ac) / _af_file if _repo_path_ac else Path(_af_file)
                if not _af_abs.is_file():
                    continue
                # Prefer a file whose source mentions the missing module path
                _af_source = _af.get("source", "") + _af.get("source_real", "")
                _mentions  = (
                    _result_file in _af_source
                    or _missing_module in _af_source
                    or _result_file.split("/")[-2] in _af_source  # "services"
                )
                if _mentions:
                    _caller_file = _af_file
                    _caller_fn   = _af.get("function", result.get("function", ""))
                    logger.warning(
                        "[phase4] found caller by source grep: %s::%s (mentions '%s')",
                        _caller_file, _caller_fn, _missing_module,
                    )
                    break

            # Third pass: any existing ast_functions file (last resort)
            if not _caller_file:
                for _af in _candidates_af:
                    _af_file = _af.get("file", "").replace("\\", "/")
                    if not _af_file or _af_file in _missing_paths:
                        continue
                    _af_abs = Path(_repo_path_ac) / _af_file if _repo_path_ac else Path(_af_file)
                    if _af_abs.is_file():
                        _caller_file = _af_file
                        _caller_fn   = _af.get("function", result.get("function", ""))
                        logger.warning(
                            "[phase4] last-resort caller: %s::%s",
                            _caller_file, _caller_fn,
                        )
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

    result["coder_instructions"] = _build_coder_instructions(result)
    logger.info("[phase4] coder_instructions built (%d chars)", len(result["coder_instructions"]))

    # Write task.md next to the repo (specs/ dir inside repo_path)
    result["task_file"] = _write_task_md(result, state)

    return {**state, "location": result, "confidence": confidence}


def _write_task_md(location: dict, state: dict) -> str:
    """
    Write specs/task_{issue_id}.md inside repo_path.
    Returns the absolute path, or '' on failure.
    """
    repo_path = state.get("repo_path", "")
    ticket    = state.get("ticket", {})
    issue_id  = ticket.get("id", "spec")
    if not repo_path:
        return ""
    try:
        specs_dir = Path(repo_path) / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        task_path = specs_dir / f"task_{issue_id}.md"
        content   = _build_task_md_content(location, ticket)
        task_path.write_text(content, encoding="utf-8")
        logger.info("[phase4] task.md written: %s", task_path)
        return str(task_path.resolve())
    except Exception as exc:
        logger.warning("[phase4] Could not write task.md: %s", exc)
        return ""


def _build_task_md_content(location: dict, ticket: dict) -> str:
    """
    Build a concise to-do list task file consumed by Agent Coder.
    Injected verbatim into the coder LLM prompt — keep it short and actionable.
    """
    # ── Extract fields ────────────────────────────────────────────────────────
    pc            = location.get("patch_constraints") or {}
    missing_files = location.get("missing_files") or []
    fallbacks     = location.get("fallback_locations") or []
    confidence    = float(location.get("confidence", 0.0))
    language      = location.get("language", "")
    bug_file      = location.get("file", "")
    bug_fn        = location.get("function", "") or "_module_level_"
    bug_line      = location.get("line", 0)
    root_cause    = location.get("root_cause", "")
    expected      = location.get("expected_behavior", "")
    code_ctx      = location.get("code_context", "")
    preserve      = pc.get("preserve_tests") or []
    forbidden     = pc.get("forbidden_files") or []
    style         = pc.get("style_hint", "")
    title         = ticket.get("title", "Bug Fix")

    # ── Sanitize: never expose internal error messages ────────────────────────
    _BAD_MARKERS = (
        "LLM output unparseable",
        "defaulting to top RAG hit",
        "deterministic fallback",
        "LLM failed",
    )
    def _sanitize(value: str, fallback: str) -> str:
        return fallback if any(m in value for m in _BAD_MARKERS) else value

    root_cause = _sanitize(root_cause, "See ticket description for root cause.")
    expected   = _sanitize(expected,   "Project runs without error after fix.")

    # ── Header ────────────────────────────────────────────────────────────────
    lines = []
    lines.append(f"# Task — {title}\n")
    meta_parts = [f"Confidence: {confidence:.0%}", language or "unknown"]
    if bug_file:
        meta_parts.append(bug_file)
    lines.append(f"> {' | '.join(meta_parts)}\n")
    lines.append("---\n")

    # ── To-Do list ────────────────────────────────────────────────────────────
    lines.append("## To-Do\n")
    step = 1

    # Step 1 — Open the file
    if bug_file:
        loc_str = f"`{bug_file}`"
        if bug_line:
            loc_str += f" line {bug_line}"
        loc_str += f" — `{bug_fn}`"
        lines.append(f"- [ ] **{step}. OPEN** {loc_str}")
    else:
        lines.append(f"- [ ] **{step}. OPEN** — file not identified, see Fallbacks below")
    step += 1

    # Step 2 — Understand
    lines.append(f"- [ ] **{step}. UNDERSTAND** — {root_cause}")
    step += 1

    # Step 3 — Fix (existing file)
    fix_detail = expected
    if style:
        fix_detail += f"  Style: {style}"
    lines.append(f"- [ ] **{step}. FIX** — {fix_detail}")
    step += 1

    # Step N — Create missing files (if any)
    for mf in missing_files:
        mf_path = mf.get("path", "")
        mf_why  = mf.get("reason", "")
        lines.append(f"- [ ] **{step}. CREATE** `{mf_path}` — {mf_why}")
        step += 1

    # Step N — Do not touch
    do_not = list(forbidden) + [t for t in preserve]
    if do_not:
        joined = ", ".join(f"`{f}`" for f in do_not)
        lines.append(f"- [ ] **{step}. DO NOT TOUCH** — {joined}")
        step += 1

    # Step N — Validate
    validate_items = ["syntax check passes", "`git diff` non-empty"]
    if preserve:
        validate_items.append(f"tests pass: {', '.join(f'`{t}`' for t in preserve)}")
    lines.append(f"- [ ] **{step}. VALIDATE** — {' + '.join(validate_items)}")

    lines.append("")

    # ── Code at bug location ──────────────────────────────────────────────────
    if code_ctx:
        lines.append(f"## Code at bug location (line {bug_line})\n")
        lines.append(f"```{language}")
        lines.append(code_ctx)
        lines.append("```")
        lines.append("")

    # ── Missing file templates ────────────────────────────────────────────────
    if missing_files:
        lines.append("## Files to create\n")
        for mf in missing_files:
            tmpl = mf.get("template", "")
            if tmpl:
                lines.append(f"### `{mf.get('path', '')}`")
                lines.append(f"```{language}")
                lines.append(tmpl)
                lines.append("```")
                lines.append("")

    # ── Fallbacks ─────────────────────────────────────────────────────────────
    if fallbacks:
        lines.append(
            f"## Fallbacks (confidence {confidence:.0%} — try these if primary fix fails)\n"
        )
        for fb in fallbacks[:3]:
            fb_file = fb.get("file", "")
            fb_fn   = fb.get("function", "")
            lines.append(f"- `{fb_file}` -> `{fb_fn}`")
        lines.append("")

    return "\n".join(lines)


# ── Coder instructions builder ────────────────────────────────────────────────


def _build_coder_instructions(location: dict) -> str:
    """
    Generate a human-readable description of the bug for the console output.
    No numbered steps — pure description format.
    The to-do list lives in task.md; this is the narrative summary.
    """
    bug_file      = location.get("file", "")
    bug_function  = location.get("function", "")
    bug_line      = location.get("line", 0)
    root_cause    = location.get("root_cause", "")
    expected      = location.get("expected_behavior", "")
    confidence    = float(location.get("confidence", 0.0))
    language      = location.get("language", "")
    missing_files = location.get("missing_files") or []
    fallbacks     = location.get("fallback_locations") or []
    pc            = location.get("patch_constraints") or {}
    scope         = pc.get("scope", "")
    style         = pc.get("style_hint", "")
    preserve      = pc.get("preserve_tests") or []
    forbidden     = pc.get("forbidden_files") or []
    callers       = location.get("callers") or []
    callees       = location.get("callees") or []

    lines = ["## Bug Report\n"]

    # ── Location ──────────────────────────────────────────────────────────────
    if bug_file:
        loc_parts = [f"File: `{bug_file}`"]
        if bug_function:
            loc_parts.append(f"Function: `{bug_function}`")
        if bug_line:
            loc_parts.append(f"Line: {bug_line}")
        if language:
            loc_parts.append(f"Language: {language}")
        loc_parts.append(f"Confidence: {confidence:.0%}")
        lines.append("  ".join(loc_parts))
    else:
        lines.append("File not identified — see fallback locations below.")
    lines.append("")

    # ── Root cause & expected ─────────────────────────────────────────────────
    lines.append(f"**Root cause**: {root_cause}")
    lines.append("")
    lines.append(f"**Expected after fix**: {expected}")
    lines.append("")

    # ── Call graph (when available) ───────────────────────────────────────────
    if callers:
        lines.append(f"**Called by**: {', '.join(callers[:3])}")
    if callees:
        lines.append(f"**Calls**: {', '.join(callees[:3])}")
    if callers or callees:
        lines.append("")

    # ── Constraints ───────────────────────────────────────────────────────────
    if scope:
        lines.append(f"**Scope**: {scope}")
    if style:
        lines.append(f"**Style**: {style}")
    if forbidden:
        lines.append(f"**Do NOT touch**: {', '.join(f'`{f}`' for f in forbidden)}")
    if preserve:
        lines.append(f"**Preserve tests**: {', '.join(f'`{t}`' for t in preserve)}")
    if scope or style or forbidden or preserve:
        lines.append("")

    # ── Missing files ─────────────────────────────────────────────────────────
    if missing_files:
        lines.append("**Files to create**:")
        for mf in missing_files:
            lines.append(f"  - `{mf.get('path', '')}` — {mf.get('reason', '')}")
        lines.append("")

    # ── Fallbacks ─────────────────────────────────────────────────────────────
    if fallbacks:
        lines.append(
            f"**Fallbacks** (confidence {confidence:.0%} — if primary fix fails):"
        )
        for fb in fallbacks[:3]:
            lines.append(f"  - `{fb.get('file')}` -> `{fb.get('function')}` — {fb.get('reason', '')}")
        lines.append("")

    return "\n".join(lines)
