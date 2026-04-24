"""
SpecV2Handler — Bridge between the Orchestrator (AgentInput) and SpecAgentV2 pipeline.

Drop-in replacement for SpecHandler (v1) in agents/spec_agent/handler.py.
Same AgentInput → AgentOutput contract. Calls run_spec_agent_v2() instead of run_agent_spec().

Usage (inside a FastAPI app):
    from agent_spec_v2.handler import SpecV2Handler
    handler = SpecV2Handler()
    output  = handler.process(agent_input.dict())
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _rel_path(abs_path: str, workspace_path: str) -> str:
    """Return abs_path relative to workspace_path (forward slashes). Falls back to abs_path."""
    if not abs_path or not workspace_path:
        return abs_path.replace("\\", "/") if abs_path else ""
    try:
        return str(Path(abs_path).relative_to(workspace_path)).replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


class SpecV2Handler:
    """
    Orchestrator-facing handler for the SpecAgentV2 pipeline.

    Identical interface to SpecHandler (v1) — fully backward compatible.
    The only difference: calls run_spec_agent_v2() (ReAct + 15 tools) instead
    of run_agent_spec() (deterministic 4-phase pipeline).
    """

    # Confidence thresholds — kept identical to SpecHandler v1.
    _THRESHOLD_SUCCESS = 0.7
    _THRESHOLD_PARTIAL = 0.4

    # ── Public entry point ─────────────────────────────────────────────────────

    def process(self, request: dict) -> dict:
        """
        Receive an AgentInput (serialised dict) and return an AgentOutput dict.

        Args:
            request: AgentInput serialised as a plain dict.

        Returns:
            AgentOutput serialised as a plain dict.
        """
        start_ms = time.perf_counter()

        try:
            result = self._run(request)
        except Exception as exc:
            logger.exception("[SpecV2Handler] Unhandled exception: %s", exc)
            elapsed_ms = int((time.perf_counter() - start_ms) * 1000)
            return {
                "status":     "failed",
                "output":     {},
                "confidence": 0.0,
                "error":      str(exc),
                "metadata": {
                    "execution_time_ms": elapsed_ms,
                    "llm_model":         request.get("metadata", {}).get("llm_model", ""),
                    "agent_version":     "v2",
                    "warnings":          [],
                },
            }

        elapsed_ms = int((time.perf_counter() - start_ms) * 1000)
        result["metadata"]["execution_time_ms"] = elapsed_ms
        return result

    # ── Internal pipeline ──────────────────────────────────────────────────────

    def _run(self, request: dict) -> dict:

        # 1. Extract top-level fields from AgentInput (same as SpecHandler v1).
        workspace_path   = request["workspace_path"]
        ticket           = request.get("ticket", {})
        ticket_summary   = request.get("ticket_summary", {})
        mr_diff          = request.get("mr_diff", "")
        metadata         = request.get("metadata", {})
        step_id          = request.get("step_id", "spec")
        previous_outputs = request.get("previous_outputs", {})

        # 2. Build enriched ticket for run_spec_agent_v2().
        issue_id = (
            ticket_summary.get("issue_id")
            or ticket.get("issue_id")
            or step_id
        )
        ticket_for_pipeline: Dict[str, Any] = {
            "id":                  str(issue_id),
            "title":               ticket_summary.get("title")       or ticket.get("title", ""),
            "description":         ticket_summary.get("description") or ticket.get("description", ""),
            "summary":             ticket_summary.get("summary", ""),
            "severity":            ticket_summary.get("priority", "normal"),
            "component":           ticket_summary.get("scope", ""),
            "labels":              ticket_summary.get("labels", []),
            "branch":              ticket_summary.get("branch", ""),
            "acceptance_criteria": ticket_summary.get("acceptance_criteria", []),
            "constraints":         ticket_summary.get("constraints", ""),
            "non_goals":           ticket_summary.get("non_goals", ""),
            "hinted_scope":        ticket_summary.get("hinted_scope", []),
            "mr_diff":             mr_diff,
        }

        # 3. Resolve execution config from metadata.
        llm_model = metadata.get("llm_model") or None
        max_turns = int(metadata.get("max_turns") or 15)

        # Carry forward retry feedback from a previous spec attempt.
        retry_feedback = (previous_outputs.get("spec") or {}).get("error") or None
        if retry_feedback:
            ticket_for_pipeline["retry_feedback"] = retry_feedback

        logger.info(
            "[SpecV2Handler] Starting — step_id=%s  issue=%s  repo=%s  model=%s",
            step_id, issue_id, workspace_path, llm_model or "default",
        )

        # 4. Call SpecAgentV2 pipeline.
        from agent_spec_v2 import run_spec_agent_v2
        location: dict = run_spec_agent_v2(
            ticket    = ticket_for_pipeline,
            mr_diff   = mr_diff,
            repo_path = workspace_path,
            model     = llm_model or "llama3.2",
            max_turns = max_turns,
        )

        confidence = float(location.get("confidence", 0.0))
        logger.info("[SpecV2Handler] Pipeline done — confidence=%.2f", confidence)

        # 5. Write spec.md to disk (same path convention as v1).
        spec_file = self._write_spec_file(
            location, ticket_summary, workspace_path, issue_id
        )

        # 6. Format AgentOutput.output (SpecAgentOutput-compatible).
        output = self._format_output(location, spec_file, ticket_summary, workspace_path)

        # 7. Determine status.
        if confidence >= self._THRESHOLD_SUCCESS:
            status = "success"
        elif confidence >= self._THRESHOLD_PARTIAL:
            status = "partial"
        else:
            status = "failed"

        warnings: List[str] = []
        if not location.get("file"):
            warnings.append("bug file not identified")
        if not location.get("function"):
            warnings.append("bug function not identified")
        if confidence < self._THRESHOLD_SUCCESS:
            warnings.append(
                f"low confidence ({confidence:.2f}) — review fallback_locations"
            )

        return {
            "status":     status,
            "output":     output,
            "confidence": confidence,
            "error":      None,
            "metadata": {
                "execution_time_ms": 0,          # filled by process()
                "agent":             "spec_v2",
                "step_id":           step_id,
                "llm_model":         llm_model or "",
                "agent_version":     "v2",
                "warnings":          warnings,
            },
        }

    # ── Spec file writer ───────────────────────────────────────────────────────

    def _write_spec_file(
        self,
        location:       dict,
        ticket_summary: dict,
        workspace_path: str,
        issue_id:       Any,
    ) -> str:
        """
        Write specs/spec_{issue_id}.md inside workspace_path.
        Returns the relative path from workspace_path (e.g. "specs/spec_42.md").
        Falls back to the relative path even if writing fails.
        """
        specs_dir = Path(workspace_path) / "specs"
        spec_path = specs_dir / f"spec_{issue_id}.md"

        patch_constraints: dict = location.get("patch_constraints") or {}
        missing_files:     list = location.get("missing_files")     or []
        fallbacks:         list = location.get("fallback_locations") or []

        # Optional sections
        missing_section = ""
        if missing_files:
            lines = ["## Files to Create\n"]
            for mf in missing_files:
                lines.append(f"### `{mf.get('path', '')}`")
                lines.append(f"**Reason**: {mf.get('reason', '')}\n")
                if mf.get("template"):
                    lines.append("```")
                    lines.append(mf["template"])
                    lines.append("```\n")
            missing_section = "\n".join(lines)

        fallback_section = ""
        if fallbacks:
            rows = "\n".join(
                f"- `{fb.get('file')}::{fb.get('function')}` — {fb.get('reason', '')}"
                for fb in fallbacks
            )
            fallback_section = f"## Fallback Locations\n{rows}\n"

        content = f"""\
# Bug Fix Specification — {ticket_summary.get("title", "Untitled")}

## Summary
{ticket_summary.get("summary", "")}

## Problem
{location.get("problem_summary", "")}

## Root Cause
{location.get("root_cause", "")}

## Bug Location
- **File**: `{location.get("file", "")}`
- **Function**: `{location.get("function", "")}`
- **Line**: {location.get("line", 0)}
- **Language**: {location.get("language", "")}

## Code Context
```{location.get("language", "")}
{location.get("code_context", "")}
```

## Expected Behavior
{location.get("expected_behavior", "")}

## Patch Constraints
- **Scope**: {patch_constraints.get("scope", "")}
- **Preserve tests**: {patch_constraints.get("preserve_tests", [])}
- **Forbidden files**: {patch_constraints.get("forbidden_files", [])}
- **Style**: {patch_constraints.get("style_hint", "")}

## Call Graph
- **Callers**: {location.get("callers", [])}
- **Callees**: {location.get("callees", [])}

{missing_section}
{fallback_section}
## Acceptance Criteria
{self._format_list(ticket_summary.get("acceptance_criteria", []))}

## Constraints
{ticket_summary.get("constraints", "")}
"""

        try:
            specs_dir.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(content, encoding="utf-8")
            logger.info("[SpecV2Handler] spec.md written: %s", spec_path)
        except Exception as exc:
            logger.warning(
                "[SpecV2Handler] Could not write spec.md (%s) — continuing.", exc
            )

        return f"specs/spec_{issue_id}.md"

    # ── Output formatter ───────────────────────────────────────────────────────

    def _format_output(
        self,
        location:       dict,
        spec_file:      str,
        ticket_summary: dict,
        workspace_path: str = "",
    ) -> dict:
        """
        Build the output dict consumed by the Coder agent.

        Design principles:
        - All fields are machine-readable (no embedded prose / markdown blobs).
        - code_context, coder_instructions, task_file are direct fields — not
          buried inside implementation_notes — so the Coder can access them
          without parsing markdown.
        - All file paths are relative to workspace_path (forward slashes).
        - confidence is NOT duplicated here; it lives at the AgentOutput top level.
        """
        patch_constraints: dict = location.get("patch_constraints") or {}
        fallbacks:         list = location.get("fallback_locations") or []
        hinted_scope:      list = ticket_summary.get("hinted_scope") or []

        # suggested_files: bug file first, then fallbacks, then hinted_scope hints
        seen:            set        = set()
        suggested_files: List[str] = []
        for f in (
            [location.get("file")]
            + [fb.get("file") for fb in fallbacks]
            + hinted_scope
        ):
            if f and f not in seen:
                seen.add(f)
                suggested_files.append(f)

        # constraints: scope + ticket constraints + forbidden files
        constraints: List[str] = []
        if patch_constraints.get("scope"):
            constraints.append(patch_constraints["scope"])
        tc = ticket_summary.get("constraints")
        if tc:
            constraints.append(str(tc))
        for f in patch_constraints.get("forbidden_files") or []:
            constraints.append(f"Do not modify: {f}")

        return {
            # ── Localisation — WHERE to fix ───────────────────────────────────
            "bug_file":           location.get("file", ""),
            "bug_function":       location.get("function", ""),
            "bug_line":           location.get("line", 0),
            "language":           location.get("language", ""),
            # ── Bug context — WHAT is wrong ───────────────────────────────────
            "code_context":       location.get("code_context", ""),
            "root_cause":         location.get("root_cause", ""),
            "expected_behavior":  location.get("expected_behavior", ""),
            # ── Coder instructions — HOW to fix ───────────────────────────────
            "coder_instructions": location.get("coder_instructions", ""),
            # ── Constraints — BOUNDARIES ──────────────────────────────────────
            "patch_constraints":  patch_constraints,
            "acceptance_criteria": ticket_summary.get("acceptance_criteria") or [],
            "constraints":        constraints,
            "suggested_files":    suggested_files,
            # ── Impact — SIDE EFFECTS ─────────────────────────────────────────
            "callers":            location.get("callers") or [],
            "callees":            location.get("callees") or [],
            "fallback_locations": fallbacks,
            "missing_files":      location.get("missing_files") or [],
            # ── References ────────────────────────────────────────────────────
            "task_file":          _rel_path(location.get("task_file", ""), workspace_path),
            "spec_file":          spec_file,   # already relative: specs/spec_{id}.md
        }

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_list(items) -> str:
        """Format a list (or string) as a markdown bullet list."""
        if not items:
            return ""
        if isinstance(items, str):
            return items
        return "\n".join(f"- {item}" for item in items)
