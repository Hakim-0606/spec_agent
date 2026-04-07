"""
spec_adapter.py — Adaptateur Agent Spec pour l'Orchestrateur



Prérequis côté Orchestrateur :
    pip install git+https://github.com/Hakim-0606/spec_agent.git

Interface respectée : executor.py → _prepare_context() + _validate_agent_result()
"""

import logging
from pathlib import Path

from agent_spec.graph import run_agent_spec

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6


def spec_agent(step_id: str, context: dict) -> dict:
    """
    Point d'entrée de l'Agent Spec pour l'Orchestrateur.

    Args:
        step_id : Identifiant de l'étape (transmis tel quel dans task_id).
        context : Dictionnaire préparé par executor.py._prepare_context().

    Returns:
        Dict conforme à _validate_agent_result() :
        {
            "status":     "success" | "failed",
            "output":     dict | None,
            "confidence": float,
            "task_id":    str,
            "agent":      "spec",
        }
    """

    # ── Étape 1 : Extraire les inputs ─────────────────────────────────────────
    ticket = context.get("ticket", {})
    metadata = context.get("metadata", {})

    mr_diff = metadata.get("mr_diff", "")
    repo_path = metadata.get("repo_path", "")
    thread_id = metadata.get("thread_id") or step_id
    llm_model = metadata.get("llm_model", None)

    # ── Étape 2 : Valider les inputs obligatoires ──────────────────────────────
    if not repo_path or not Path(repo_path).is_dir():
        logger.error(
            "[spec_agent] repo_path invalide ou inexistant : %r", repo_path
        )
        return {
            "status": "failed",
            "output": None,
            "confidence": 0.0,
            "task_id": step_id,
            "agent": "spec",
            "error": f"repo_path invalide ou inexistant : {repo_path!r}",
        }

    if not ticket or "title" not in ticket:
        logger.error(
            "[spec_agent] ticket manquant ou sans clé 'title' : %r", ticket
        )
        return {
            "status": "failed",
            "output": None,
            "confidence": 0.0,
            "task_id": step_id,
            "agent": "spec",
            "error": "ticket manquant ou sans clé 'title'",
        }

    if not mr_diff:
        logger.warning(
            "[spec_agent] mr_diff vide — le pipeline continuera sans diff"
        )

    # ── Étape 3 : Appeler run_agent_spec ──────────────────────────────────────
    try:
        result = run_agent_spec(
            ticket=ticket,
            mr_diff=mr_diff,
            repo_path=repo_path,
            thread_id=thread_id,
            llm_model=llm_model,
        )
    except Exception as exc:
        logger.exception("[spec_agent] Erreur inattendue dans run_agent_spec")
        return {
            "status": "failed",
            "output": None,
            "confidence": 0.0,
            "task_id": step_id,
            "agent": "spec",
            "error": str(exc),
        }

    # ── Étape 4 : Construire la réponse succès ─────────────────────────────────
    confidence = float(result.get("confidence", 0.0))

    response = {
        "status": "success",
        "output": result,
        "confidence": confidence,
        "task_id": step_id,
        "agent": "spec",
    }

    # ── Étape 5 : Vérifier le seuil de confiance ──────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            "[spec_agent] confidence trop basse : %.2f (seuil : %.2f)",
            confidence,
            CONFIDENCE_THRESHOLD,
        )
        response["status"] = "failed"
        response["reason"] = "low_confidence"

    return response
