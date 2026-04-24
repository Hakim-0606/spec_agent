"""
Spec Agent V2 — FastAPI microservice entry point.

Drop-in replacement for agents/spec_agent/main.py (v1).
Same /execute, /health, /ready endpoints — orchestrator sees no difference.
Uses SpecV2Handler (ReAct + 15 tools) instead of SpecHandler (4-phase deterministic).
"""

import logging
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# AgentInput: use the shared schema when running inside the Orchestrateur project,
# fall back to a minimal compatible model when running standalone.
try:
    from shared.schemas.agent_io import AgentInput
except ImportError:
    from pydantic import BaseModel
    from typing import Any, Dict

    class AgentInput(BaseModel):  # type: ignore[no-redef]
        step_id:          str
        workspace_path:   str
        ticket:           Dict[str, Any] = {}
        ticket_summary:   Dict[str, Any] = {}
        mr_diff:          str            = ""
        metadata:         Dict[str, Any] = {}
        previous_outputs: Dict[str, Any] = {}
        step_description: str            = ""
        agent_type:       str            = "spec"

app = FastAPI(title="Spec Agent V2", version="2.0.0")

from agent_spec_v2.handler import SpecV2Handler

handler = SpecV2Handler()


@app.post("/execute")
async def execute(request: AgentInput):
    """
    Receive an AgentInput, run the ReAct bug-localisation pipeline, return AgentOutput.
    """
    try:
        result = handler.process(request.dict())
        return result
    except Exception as exc:
        logger.exception("[main] Unhandled exception in /execute: %s", exc)
        return {
            "status":     "failed",
            "output":     {},
            "confidence": 0.0,
            "error":      str(exc),
            "metadata":   {"agent": "spec_v2", "step_id": request.step_id},
        }


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "spec_v2"}


@app.get("/ready")
async def ready():
    return {"status": "ready", "agent": "spec_v2"}
