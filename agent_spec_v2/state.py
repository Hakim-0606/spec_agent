from typing import Any, List, Optional, Set, TypedDict


class SpecV2State(TypedDict):
    # Inputs
    ticket:    dict
    mr_diff:   str
    repo_path: str
    llm_model: str
    max_turns: int

    # Loop control
    messages:         list
    turn:             int
    tool_calls_count: int        # total tool calls — kept for backward compat
    action_count:     int        # real tool calls only (not thinking/empty turns)
    seen_tool_calls:  Set[str]
    phase:            str        # "explore" | "confirm"

    # Lazily populated by tools (None = not yet computed)
    bm25_index:    Optional[object]       # {"bm25": BM25Okapi, "files": List[str]}
    all_functions: Optional[List[dict]]   # accumulated from ast_analyse calls
    repo_graph:    Optional[Any]          # networkx.DiGraph built by build_call_graph

    # Best answer found so far — updated whenever the LLM produces valid JSON
    hypothesis: Optional[dict]            # {file, function, line, root_cause, confidence}

    # Final output
    location:   Optional[dict]
    confidence: float
