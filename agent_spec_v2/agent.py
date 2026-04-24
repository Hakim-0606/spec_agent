"""
SpecAgentV2 — Intelligent Bug Localisation Agent
=================================================
The LLM drives the full analysis loop, deciding which tools to call
and in what order. Compatible output schema with spec_agent_v1.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .prompts import (
    SYSTEM_PROMPT,
    OUTPUT_SCHEMA_REMINDER,
    PHASE_TRANSITION_TEMPLATE,
    REACT_OBSERVATION_REQUEST,
)
from .state import SpecV2State
from .tools import get_tools_for_phase, execute_tool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Phase budgets — tool calls per phase (not conversation turns).
# Thinking-only turns don't count against these budgets.
EXPLORE_BUDGET = 3   # max real tool calls in the explore phase
CONFIRM_BUDGET = 3   # max real tool calls in the confirm phase
MAX_ACTIONS    = EXPLORE_BUDGET + CONFIRM_BUDGET  # 6 total

_REQUIRED_KEYS = {"file", "function", "line", "root_cause", "confidence"}

# Routing hints injected into the prompt based on error type.
# These are SUGGESTIONS for the LLM, not hard rules.
_ERROR_STRATEGY_HINTS: dict = {
    "ImportError": (
        "Best first tool: `import_graph` on the module file to detect circular imports. "
        "Or use `search_in_repo` with the search_pattern above to find which file has the bad import."
    ),
    "ModuleNotFoundError": (
        "Best first tool: `search_in_repo` to find where the missing module is imported. "
        "Then `get_project_structure` to verify the module file exists."
    ),
    "AttributeError": (
        "Best first tool: `search_in_repo` to find the class definition (e.g. `class MyClass`). "
        "Then `ast_analyse` to check which methods/attributes it actually has."
    ),
    "TypeError": (
        "Best first tool: `search_in_repo` to locate the function/method being called. "
        "Then `read_file` to inspect its signature and argument types."
    ),
    "NameError": (
        "Best first tool: `search_in_repo` to find where the symbol should be defined "
        "(check imports, global scope, typos)."
    ),
    "KeyError": (
        "Best first tool: `read_file` on the traceback file to see the exact line "
        "and understand which dict key is missing."
    ),
    "IndexError": (
        "Best first tool: `read_file` on the traceback file to see the exact line "
        "and understand the list/array access."
    ),
    "RecursionError": (
        "Best first tool: `import_graph` or `get_callers` to find the recursive call cycle."
    ),
    "RuntimeError": (
        "Best first tool: `bm25_search` with keywords from the error message to locate relevant files."
    ),
}

_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".cpp": "cpp",
    ".c": "c", ".cs": "c_sharp", ".rb": "ruby",
    ".kt": "kotlin", ".swift": "swift",
}


# ── LLM backend (copied from agent_spec/phase4_llm.py — self-contained) ───────


class _SpecLLMBackend:
    """
    Thin wrapper normalizing Ollama vs OpenAI-compatible tool-calling.

    Ollama and OpenAI differ in three ways:
      1. Response structure  : response["message"]  vs  response.choices[0].message
      2. Tool call arguments : dict                 vs  JSON string
      3. Tool result message : no id required       vs  tool_call_id required
    """

    def __init__(self, model: str, temperature: float = 0.1):
        self.model       = model
        self.temperature = temperature

        lm_url = os.environ.get("LM_STUDIO_URL", "").rstrip("/")
        if lm_url:
            from openai import OpenAI
            base_url      = lm_url if lm_url.endswith("/v1") else lm_url + "/v1"
            self._client  = OpenAI(base_url=base_url, api_key="lm-studio")
            self._backend = "openai"
            logger.info("[specv2] LLM backend: LM Studio at %s", base_url)
        else:
            try:
                import ollama as _ollama_mod
                self._ollama  = _ollama_mod
                self._client  = None
                self._backend = "ollama"
                logger.info("[specv2] LLM backend: Ollama")
            except ImportError:
                raise RuntimeError(
                    "No LLM backend available. "
                    "Set LM_STUDIO_URL or install the 'ollama' package."
                )

    def chat(self, messages: list, tools: list):
        """
        Returns:
            content       : str
            tool_calls    : list of {"name": str, "arguments": dict, "_id": str}
            assistant_msg : dict  — append to conversation history
            make_tool_msg : callable(result: str, call: dict) → dict
        """
        if self._backend == "openai":
            return self._chat_openai(messages, tools)
        return self._chat_ollama(messages, tools)

    def _chat_openai(self, messages: list, tools: list):
        kwargs: dict = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools

        resp    = self._client.chat.completions.create(**kwargs)
        oai_msg = resp.choices[0].message
        content = oai_msg.content or ""

        normalized = []
        oai_tcs = oai_msg.tool_calls or []
        for tc in oai_tcs:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            normalized.append({"name": tc.function.name, "arguments": args, "_id": tc.id})

        assistant_msg: dict = {"role": "assistant", "content": content}
        if oai_tcs:
            assistant_msg["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in oai_tcs
            ]

        def make_tool_msg(result: str, call: dict) -> dict:
            return {"role": "tool", "tool_call_id": call["_id"], "content": result}

        return content, normalized, assistant_msg, make_tool_msg

    def _chat_ollama(self, messages: list, tools: list):
        resp    = self._ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools or [],
            options={"temperature": self.temperature},
        )
        raw_msg = resp["message"]
        content = raw_msg.get("content") or ""

        normalized = []
        for tc in (raw_msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            normalized.append({
                "name":      fn.get("name", ""),
                "arguments": fn.get("arguments", {}),
                "_id":       "",
            })

        assistant_msg: dict = {"role": "assistant", "content": content}
        if raw_msg.get("tool_calls"):
            assistant_msg["tool_calls"] = raw_msg["tool_calls"]

        def make_tool_msg(result: str, call: dict) -> dict:  # noqa: ARG001
            return {"role": "tool", "content": result}

        return content, normalized, assistant_msg, make_tool_msg


# ── JSON helpers ───────────────────────────────────────────────────────────────


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3 / o1-style models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_tool_call_xml(text: str) -> str:
    """Remove <tool_call>...</tool_call> blocks Qwen3 emits when tools=[]."""
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Optional[str]:
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
                return text[start: i + 1]
    return None


def _coerce_types(data: dict) -> dict:
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
    if not raw:
        return None
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
        if _REQUIRED_KEYS <= set(data.keys()):
            return _coerce_types(data)
    except json.JSONDecodeError:
        pass
    extracted = _extract_json_object(cleaned)
    if extracted:
        try:
            data = json.loads(extracted)
            if _REQUIRED_KEYS <= set(data.keys()):
                return _coerce_types(data)
        except json.JSONDecodeError:
            pass
    logger.warning("[specv2] JSON parse failed. Raw (first 300 chars): %r", raw[:300])
    return None


# ── Stop criteria ──────────────────────────────────────────────────────────────


def _should_stop(state: SpecV2State, result: Optional[dict]) -> bool:
    if result is not None:
        conf = result.get("confidence", 0.0)
        # High-confidence answer after any real tool call → stop immediately.
        if conf >= 0.85 and state["action_count"] >= 1:
            return True
        # Normal confidence → require at least explore + one confirm action.
        if conf >= 0.7 and state["action_count"] >= 2:
            return True
    # Total action budget exhausted.
    if state["action_count"] >= MAX_ACTIONS:
        return True
    # Absolute safety net on raw turns (catches infinite thinking loops).
    if state["turn"] >= state["max_turns"]:
        return True
    return False


# ── SpecAgentV2 ────────────────────────────────────────────────────────────────


class SpecAgentV2:
    def __init__(self, model: str = DEFAULT_MODEL, max_turns: int = 15):
        self.model     = model
        self.max_turns = max_turns

    def run(self, ticket: dict, mr_diff: str, repo_path: str) -> dict:
        """
        Run the agentic loop and return the final location dict.
        Compatible output with spec_agent_v1.
        """
        # Detect input mode
        has_ticket = bool(ticket.get("title") or ticket.get("description"))
        has_diff   = bool(mr_diff.strip())
        if has_ticket and has_diff:
            mode = "C"
        elif has_diff:
            mode = "B"
        else:
            mode = "A"
        logger.info("[specv2] Input mode: %s  (ticket=%s, diff=%s)", mode, has_ticket, has_diff)

        # Inject architecture.md if available
        arch_md = ""
        if repo_path:
            try:
                from agent_spec.architecture_cache import get_architecture_cache
                md_file = get_architecture_cache().md_path(repo_path)
                if md_file.exists():
                    arch_md = md_file.read_text(encoding="utf-8")
            except Exception:
                pass

        # Extract structured signals from ticket text (error type, file, symbol)
        signals = self._pre_analyse_ticket(ticket) if has_ticket else {}
        if signals:
            logger.info("[specv2] ticket signals: %s", signals)

        # Build initial user message
        user_msg = self._build_user_message(ticket, mr_diff, mode, arch_md, signals)

        # Initialise state
        state: SpecV2State = {
            "ticket":          ticket,
            "mr_diff":         mr_diff,
            "repo_path":       repo_path,
            "llm_model":       self.model,
            "max_turns":       self.max_turns,
            "messages":        [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "turn":             0,
            "tool_calls_count": 0,   # backward compat
            "action_count":     0,   # real tool calls only
            "seen_tool_calls":  set(),
            "phase":            "explore",
            "bm25_index":       None,
            "all_functions":    None,
            "repo_graph":       None,
            "hypothesis":       None,
            "location":         None,
            "confidence":       0.0,
        }

        backend = _SpecLLMBackend(model=self.model)
        result: Optional[Dict] = None

        # ── Main agentic loop (ReAct: Reason → Act each cycle) ────────────────
        while not _should_stop(state, result):
            state["turn"] += 1
            logger.info(
                "[specv2] turn %d/%d  phase=%s  actions=%d",
                state["turn"], self.max_turns, state["phase"], state["action_count"],
            )

            # ── Reason step: observe last result before acting (skip turn 1) ──
            # The model verbalises what it learned and what it plans to do next.
            # If it's ready to answer it outputs JSON and we stop immediately.
            if state["turn"] > 1:
                reason_result = self._reason_step(backend, state)
                if reason_result is not None:
                    result = reason_result
                    state["hypothesis"] = reason_result
                    # Reason step is a full synthesis — trust conf≥0.85 immediately.
                    if reason_result.get("confidence", 0.0) >= 0.85:
                        break
                if _should_stop(state, result):
                    break

            # ── Action step: call a tool (or emit JSON if still no tool) ──────
            action_result, tools_called = self._action_step(backend, state)
            if action_result is not None:
                result = action_result
                state["hypothesis"] = action_result
            if _should_stop(state, result):
                break

            if not tools_called:
                # If the model has a hypothesis but can't act in explore phase,
                # advance to confirm so read_file becomes available.
                if (state["phase"] == "explore"
                        and state.get("hypothesis") is not None
                        and state["action_count"] >= 1):
                    # Replace the generic nudge with a phase transition message.
                    if state["messages"] and state["messages"][-1]["role"] == "user":
                        state["messages"].pop()
                    state["phase"] = "confirm"
                    hyp = state["hypothesis"]
                    hyp_text = (
                        f"Your current best candidate: `{hyp['file']}` "
                        f"— confidence {hyp['confidence']:.0%}. "
                        f"Use read_file to read it and confirm the exact buggy line."
                    )
                    state["messages"].append({
                        "role":    "user",
                        "content": PHASE_TRANSITION_TEMPLATE.format(
                            action_count=state["action_count"],
                            hypothesis_text=hyp_text,
                        ),
                    })
                    logger.info(
                        "[specv2] early phase transition: explore -> confirm  hypothesis=%s",
                        hyp.get("file", "?"),
                    )
                continue  # nudge already appended if no transition

            # ── Phase transition: explore → confirm (budget hit) ───────────────
            if (state["phase"] == "explore"
                    and state["action_count"] >= EXPLORE_BUDGET
                    and result is None):
                state["phase"] = "confirm"
                hyp = state.get("hypothesis")
                if hyp:
                    hyp_text = (
                        f"Your current best candidate: `{hyp['file']}` "
                        f"— confidence {hyp['confidence']:.0%}. "
                        f"Read it now to confirm the exact buggy line."
                    )
                else:
                    hyp_text = (
                        "No strong hypothesis yet. "
                        "Read the most relevant file from your search results."
                    )
                state["messages"].append({
                    "role":    "user",
                    "content": PHASE_TRANSITION_TEMPLATE.format(
                        action_count=state["action_count"],
                        hypothesis_text=hyp_text,
                    ),
                })
                logger.info(
                    "[specv2] phase: explore -> confirm  hypothesis=%s",
                    hyp.get("file", "?") if hyp else "none",
                )

            # ── Force final JSON when total action budget exhausted ────────────
            if state["action_count"] >= MAX_ACTIONS and result is None:
                logger.info("[specv2] action budget exhausted — forcing final JSON.")
                state["messages"].append({
                    "role":    "user",
                    "content": (
                        "Action budget reached. Based on everything gathered, "
                        "output ONLY the 5-field JSON now:\n" + OUTPUT_SCHEMA_REMINDER
                    ),
                })
                try:
                    content2, _, assistant_msg2, _ = backend.chat(state["messages"], [])
                    state["messages"].append(assistant_msg2)
                    visible2 = _strip_thinking(content2) if content2 else ""
                    if visible2:
                        result = _parse_llm_json(visible2)
                        if result:
                            logger.info(
                                "[specv2] forced JSON: file=%s conf=%.2f",
                                result.get("file", "?"), result.get("confidence", 0.0),
                            )
                except Exception as exc:
                    logger.warning("[specv2] forced JSON call failed: %s", exc)
                break

        # Fallback if no valid result
        if result is None:
            logger.warning("[specv2] loop exhausted — using deterministic fallback.")
            result = self._deterministic_fallback(state)

        return self._build_full_output(result, state)

    # ── ReAct steps ────────────────────────────────────────────────────────────

    def _reason_step(
        self, backend: "_SpecLLMBackend", state: SpecV2State
    ) -> Optional[Dict]:
        """
        Inject REACT_OBSERVATION_REQUEST, call LLM with no tools.
        The model verbalises what it learned and what it plans next.
        Returns parsed JSON if the model is ready to answer, else None.
        """
        state["messages"].append({
            "role":    "user",
            "content": REACT_OBSERVATION_REQUEST,
        })
        try:
            content, _, assistant_msg, _ = backend.chat(state["messages"], [])
        except Exception as exc:
            logger.warning("[specv2] reason step failed: %s", exc)
            return None

        state["messages"].append(assistant_msg)
        visible = _strip_thinking(content) if content else ""
        visible = _strip_tool_call_xml(visible)
        if not visible:
            logger.debug("[specv2] reason step: empty response (or stripped XML tool call)")
            return None

        logger.info("[specv2] reason: %s", visible[:200])
        parsed = _parse_llm_json(visible)
        if parsed is not None:
            logger.info(
                "[specv2] reason step produced JSON: file=%s conf=%.2f",
                parsed.get("file", "?"), parsed.get("confidence", 0.0),
            )
        return parsed

    def _action_step(
        self, backend: "_SpecLLMBackend", state: SpecV2State
    ):
        """
        Call LLM with the current phase's tool set and execute any tool calls.
        Returns (result_or_None, tools_were_called: bool).
        """
        current_tools = get_tools_for_phase(state)
        try:
            content, tool_calls, assistant_msg, make_tool_msg = backend.chat(
                state["messages"], current_tools
            )
        except Exception as exc:
            logger.error("[specv2] action step failed: %s", exc)
            return None, False

        state["messages"].append(assistant_msg)
        visible = _strip_thinking(content) if content else ""
        result = None

        # Enforce phase: drop tool calls not offered in current phase
        allowed_names = {t["function"]["name"] for t in current_tools}
        skipped = [c["name"] for c in tool_calls if c["name"] not in allowed_names]
        tool_calls = [c for c in tool_calls if c["name"] in allowed_names]
        if skipped:
            logger.warning("[specv2] out-of-phase tool calls dropped: %s", skipped)
            state["messages"].append({
                "role":    "user",
                "content": (
                    f"Tool(s) {skipped} are not available in the {state['phase']} phase. "
                    f"Available tools: {sorted(allowed_names)}."
                ),
            })

        # Check content for JSON (model may answer directly without a tool call)
        if visible:
            parsed = _parse_llm_json(visible)
            if parsed is not None:
                logger.info(
                    "[specv2] JSON in action content: file=%s conf=%.2f",
                    parsed.get("file", "?"), parsed.get("confidence", 0.0),
                )
                result = parsed

        # No tool calls — nudge and return
        if not tool_calls:
            if not visible:
                logger.warning("[specv2] action step: empty response, nudging.")
                state["messages"].append({
                    "role":    "user",
                    "content": (
                        "Continue: call the next tool if you need more information, "
                        "or output the JSON answer if you have enough evidence.\n"
                        + OUTPUT_SCHEMA_REMINDER
                    ),
                })
            elif result is None:
                logger.warning("[specv2] action step: no tool call, no JSON, nudging.")
                state["messages"].append({
                    "role":    "user",
                    "content": OUTPUT_SCHEMA_REMINDER,
                })
            return result, False

        # Execute tool calls
        for call in tool_calls:
            state["action_count"]     += 1
            state["tool_calls_count"] += 1
            logger.info(
                "[specv2] [%s] tool call %d: %s(%s)",
                state["phase"], state["action_count"],
                call["name"], call["arguments"],
            )
            tool_result = execute_tool(call, state)
            logger.debug("[specv2] tool result (%.200s)", tool_result)
            state["messages"].append(make_tool_msg(tool_result, call))

        return result, True

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _pre_analyse_ticket(ticket: dict) -> dict:
        """
        Extract structured signals from ticket text before any tool call.
        Returns a dict of ready-to-use hints (all keys optional).
        Does NOT replace LLM reasoning — only surfaces what is already explicit
        in the error message (file paths, symbol names, error type).
        """
        text = " ".join(filter(None, [
            ticket.get("title", ""),
            ticket.get("description", ""),
        ]))

        signals: dict = {}

        # ── Error type ────────────────────────────────────────────────────────
        m = re.search(
            r"\b(ImportError|ModuleNotFoundError|AttributeError|TypeError|"
            r"NameError|KeyError|ValueError|IndexError|RuntimeError|"
            r"RecursionError|ZeroDivisionError|FileNotFoundError|"
            r"PermissionError|OSError|AssertionError)\b",
            text,
        )
        if m:
            signals["error_type"] = m.group(1)

        # ── Full traceback — extract ALL frames ───────────────────────────────
        # Matches every: File "path.py", line N, in func_name
        raw_frames = re.findall(
            r'File ["\']([^"\']+\.py)["\'],\s*line\s*(\d+)(?:,\s*in\s+(\w+))?',
            text,
        )
        if raw_frames:
            frames = [
                {"file": f, "line": int(l), "function": fn or "?"}
                for f, l, fn in raw_frames
            ]
            signals["traceback_frames"] = frames
            # Deepest frame = where the exception was raised = most likely the bug
            deepest = frames[-1]
            signals["traceback_file"] = deepest["file"]
            signals["traceback_line"] = deepest["line"]
            signals["traceback_function"] = deepest["function"]

        # ── ImportError specifics ─────────────────────────────────────────────
        # "cannot import name 'X' from [partially initialized module] 'a.b.c'"
        m = re.search(
            r"cannot import name ['\"](\w+)['\"]"
            r"(?:\s+from(?:\s+partially initialized module)?\s+['\"]([a-zA-Z0-9_.]+)['\"])?",
            text,
        )
        if m:
            signals["import_symbol"] = m.group(1)
            if m.group(2):
                signals["import_from"]     = m.group(2)
                signals["module_file"]     = m.group(2).replace(".", "/") + ".py"
                signals["search_pattern"]  = (
                    f"from {m.group(2)} import {m.group(1)}"
                )

        # ── ModuleNotFoundError ───────────────────────────────────────────────
        if "error_type" in signals and signals["error_type"] == "ModuleNotFoundError":
            m = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
            if m:
                signals["missing_module"] = m.group(1)
                signals["module_file"]    = m.group(1).replace(".", "/") + ".py"

        # ── AttributeError specifics ─────────────────────────────────────────
        # "'MyClass' object has no attribute 'my_method'"
        m = re.search(
            r"['\"](\w+)['\"] object has no attribute ['\"](\w+)['\"]", text
        )
        if m:
            signals["attr_class"] = m.group(1)
            signals["attr_name"]  = m.group(2)

        # ── Error-type routing hint ───────────────────────────────────────────
        error_type = signals.get("error_type", "")
        if error_type in _ERROR_STRATEGY_HINTS:
            signals["strategy_hint"] = _ERROR_STRATEGY_HINTS[error_type]

        return signals

    def _build_user_message(
        self, ticket: dict, mr_diff: str, mode: str, arch_md: str,
        signals: dict = None,
    ) -> str:
        parts = [f"## Input Mode: {mode}\n"]

        if ticket:
            parts.append("## Bug Ticket")
            parts.append(f"ID:          {ticket.get('id', 'N/A')}")
            parts.append(f"Title:       {ticket.get('title', '')}")
            parts.append(f"Description: {ticket.get('description', '')}")
            parts.append(f"Severity:    {ticket.get('severity', '')}")
            parts.append(f"Component:   {ticket.get('component', '')}")
            parts.append("")

        if mr_diff:
            parts.append("## MR Diff  ← PRIMARY SIGNAL in Mode B/C")
            parts.append(mr_diff[:4000])
            parts.append("")

        if arch_md:
            parts.append("## Project Architecture")
            parts.append(arch_md[:3000])
            parts.append("")

        # ── Inject pre-analysed signals as ready-to-use hints ─────────────────
        if signals:
            parts.append("## Pre-Analysis — Extracted Signals")
            if signals.get("error_type"):
                parts.append(f"- Error type     : `{signals['error_type']}`")
            if signals.get("traceback_frames"):
                frames = signals["traceback_frames"]
                parts.append(f"- Traceback ({len(frames)} frame(s), deepest last):")
                for fr in frames:
                    parts.append(
                        f"    `{fr['file']}` line {fr['line']}"
                        + (f" in `{fr['function']}`" if fr["function"] != "?" else "")
                    )
                parts.append(
                    f"  → Deepest frame (likely the bug): "
                    f"`{signals['traceback_file']}` line {signals['traceback_line']}"
                )
            elif signals.get("traceback_file"):
                parts.append(
                    f"- Traceback file : `{signals['traceback_file']}`"
                    f"  line {signals.get('traceback_line', '?')}"
                )
            if signals.get("search_pattern"):
                parts.append(
                    f"- Search pattern : `{signals['search_pattern']}`"
                    f"  → call search_in_repo immediately to find the offending file"
                )
            if signals.get("module_file"):
                parts.append(f"- Module file    : `{signals['module_file']}`")
            if signals.get("attr_class"):
                parts.append(
                    f"- AttributeError : class `{signals['attr_class']}`"
                    f"  missing `{signals['attr_name']}`"
                )
            if signals.get("strategy_hint"):
                parts.append(f"- Strategy hint  : {signals['strategy_hint']}")
            parts.append("")

        parts.append(
            "Analyse the above. Decide which tool gives you the most useful "
            "information first, then call it. Work toward the 5-field JSON answer."
        )
        return "\n".join(parts)

    def _deterministic_fallback(self, state: SpecV2State) -> dict:
        from agent_spec.phase4_llm import _build_deterministic_fallback

        all_fns   = state.get("all_functions") or []
        ticket    = state.get("ticket", {})
        repo_path = state.get("repo_path", "")

        return _build_deterministic_fallback(
            contexts=all_fns,
            tool_search_results=[],
            ticket=ticket,
            repo_path=repo_path,
        )

    def _build_full_output(self, result: dict, state: SpecV2State) -> dict:
        """Enrich the 5-field result with the 8 extended fields + task.md."""
        from agent_spec.phase4_llm import (
            _build_8_fields_deterministic,
            _build_coder_instructions,
            _write_task_md,
        )

        ticket    = state.get("ticket", {})
        repo_path = state.get("repo_path", "")
        all_fns   = state.get("all_functions") or []

        # Enrich callers / callees / language from accumulated AST data
        matched_fn = next(
            (
                fn for fn in all_fns
                if fn.get("function") == result.get("function")
                and fn.get("file", "").replace("\\", "/") == result.get("file", "").replace("\\", "/")
            ),
            None,
        )
        result["callers"]  = (matched_fn or {}).get("callers")  or []
        result["callees"]  = (matched_fn or {}).get("callees")  or []
        result["language"] = (matched_fn or {}).get("language") or ""

        if not result["language"]:
            ext = Path(result.get("file", "")).suffix.lower()
            result["language"] = _EXT_TO_LANG.get(ext, "")

        result.setdefault("file",       "")
        result.setdefault("root_cause", "")
        result.setdefault("confidence", 0.0)
        result.setdefault("language",   "")

        # For module-level bugs (circular imports, bad imports) the LLM often
        # returns function=null/""  and line=0. Fill in sensible defaults so
        # downstream consumers never see N/A.
        if not result.get("function"):
            result["function"] = "_module_level_"
        if not result.get("line"):
            result["line"] = 1

        # Build minimal state dict for reuse of v1 helpers.
        # Pass an empty ticket to _build_8_fields_deterministic so that its
        # internal _resolve_import_error_file() call doesn't fire and emit
        # misleading "ImportError fallback" log lines — the LLM already
        # identified the correct file; we trust its answer here.
        compat_state = {
            "ticket":        {},          # suppress ImportError re-detection
            "repo_path":     repo_path,
            "bm25_files":    [],
            "all_functions": all_fns,
            "ast_functions": all_fns,
        }

        extended = _build_8_fields_deterministic(
            result, {}, compat_state, all_fns, all_fns, repo_path
        )
        result.update(extended)

        result["coder_instructions"] = _build_coder_instructions(result)
        result["task_file"]          = _write_task_md(result, compat_state)

        return result
