"""
agent_spec_v2 — Intelligent Bug Localisation Agent

Drop-in replacement for spec_agent_v1's phase_llm_confirm node.
The LLM drives the full analysis loop using 7 tools.

# TODO: replace phase_llm_confirm with run_spec_agent_v2
# from agent_spec_v2 import run_spec_agent_v2
"""

from .agent import SpecAgentV2
from .handler import SpecV2Handler


def run_spec_agent_v2(
    ticket:    dict = None,
    mr_diff:   str  = "",
    repo_path: str  = "",
    model:     str  = "llama3.2",
    max_turns: int  = 15,
) -> dict:
    """
    Drop-in replacement for spec_agent_v1's phase_llm_confirm node.

    Input modes:
      ticket only  → ticket={"title": ..., "description": ...}, mr_diff=""
      diff only    → ticket={}, mr_diff="--- a/...\\n+++ b/..."
      both         → ticket={...}, mr_diff="..."
    """
    ticket = ticket or {}
    agent  = SpecAgentV2(model=model, max_turns=max_turns)
    return agent.run(ticket, mr_diff, repo_path)


__all__ = ["run_spec_agent_v2", "SpecAgentV2", "SpecV2Handler"]
