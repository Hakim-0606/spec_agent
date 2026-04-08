"""
LangGraph StateGraph — Agent Spec
===================================
Wires the four phases into a linear pipeline with PostgreSQL checkpointing
for crash-resumable execution.

Topology:
    workspace → bm25 → treesitter → rag → tools → llm → END

Checkpointing:
    - If POSTGRES_URI env var is set: PostgresSaver (persistent, resumable)
    - Otherwise: MemorySaver (in-process, for local dev / testing)
"""

import os
from typing import Any, Optional

from langgraph.graph import StateGraph, END

from .state import SpecState
from .phase0_workspace import phase_workspace
from .phase1_bm25 import phase_bm25
from .phase2_treesitter import phase_treesitter
from .phase3_rag import phase_rag
from .phase35_tools import phase_tools
from .phase4_llm import phase_llm_confirm


# ── Checkpointer factory ───────────────────────────────────────────────────────


def _make_checkpointer():
    """
    Returns a PostgresSaver if POSTGRES_URI is configured, else MemorySaver.

    PostgreSQL URI format:
        postgresql://user:password@host:5432/dbname
    """
    postgres_uri = os.environ.get("POSTGRES_URI", "")

    if postgres_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg

            conn = psycopg.connect(postgres_uri, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()  # Creates the checkpointing tables if they don't exist.
            print(f"[graph] Using PostgreSQL checkpointer at {postgres_uri[:30]}…")
            return saver
        except ImportError:
            print(
                "[graph] langgraph-checkpoint-postgres not installed. "
                "Falling back to MemorySaver."
            )
        except Exception as exc:
            print(
                f"[graph] Cannot connect to PostgreSQL ({exc}). "
                "Falling back to MemorySaver."
            )

    from langgraph.checkpoint.memory import MemorySaver
    print("[graph] Using in-memory checkpointer (set POSTGRES_URI for persistence).")
    return MemorySaver()


# ── Graph builder ──────────────────────────────────────────────────────────────


def build_graph():
    """Build and compile the Agent Spec StateGraph."""
    builder = StateGraph(SpecState)

    builder.add_node("workspace", phase_workspace)
    builder.add_node("bm25", phase_bm25)
    builder.add_node("treesitter", phase_treesitter)
    builder.add_node("rag", phase_rag)
    builder.add_node("tools", phase_tools)
    builder.add_node("llm", phase_llm_confirm)

    builder.set_entry_point("workspace")
    builder.add_edge("workspace", "bm25")
    builder.add_edge("bm25", "treesitter")
    builder.add_edge("treesitter", "rag")
    builder.add_edge("rag", "tools")
    builder.add_edge("tools", "llm")
    builder.add_edge("llm", END)

    checkpointer = _make_checkpointer()
    return builder.compile(checkpointer=checkpointer)


# ── Public entry point ─────────────────────────────────────────────────────────

_graph = None


def run_agent_spec(
    ticket: dict,
    mr_diff: str,
    repo_path: str,
    thread_id: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> dict:
    """
    Run the Agent Spec pipeline and return the bug location dict.

    Args:
        ticket    : Structured GitLab ticket dict.
        mr_diff   : Unified diff string from the MR.
        repo_path : Absolute path to the locally cloned repository.
        thread_id : Checkpoint thread ID.  Use the ticket ID for resumability.
        llm_model : Ollama model name (default: $OLLAMA_MODEL or 'llama3.2').

    Returns:
        {
            "file", "function", "line", "root_cause",
            "confidence", "callers", "callees", "language",
            "problem_summary", "code_context",
            "patch_constraints", "expected_behavior",
            "fallback_locations"
        }
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial_state: SpecState = {
        "ticket": ticket,
        "mr_diff": mr_diff,
        "repo_path": repo_path,
        # Phase outputs — initialised empty.
        "keywords": [],
        "bm25_files": [],
        "repo_graph": {},
        "ast_functions": [],
        "all_functions": [],
        "rag_contexts": [],
        "project_structure":   {},
        "tool_search_results": [],
        "location": {},
        "confidence": 0.0,
        "rrf_scores": [],
        "llm_model": llm_model or "",
    }

    config = {"configurable": {"thread_id": thread_id or ticket.get("id", "default")}}

    final_state = _graph.invoke(initial_state, config=config)
    return final_state["location"]
