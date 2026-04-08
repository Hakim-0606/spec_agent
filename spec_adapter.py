"""
spec_adapter.py — Adaptateur Agent Spec pour l'Orchestrateur

Livrable : donner ce fichier au camarade responsable de l'Orchestrateur.
Il doit être placé dans app/agents/spec.py de son repo.

Prérequis côté Orchestrateur :
    pip install git+https://github.com/Hakim-0606/spec_agent.git

Interface respectée : executor.py → _prepare_context() + _validate_agent_result()
"""

import hashlib
import logging
from pathlib import Path

from agent_spec.graph import run_agent_spec

logger = logging.getLogger(__name__)

# Aligné sur router.py confidence_threshold (défaut 0.7).
# Le Router gère déjà la basse confiance via retry — ce seuil évite
# uniquement les résultats vraiment inutilisables.
CONFIDENCE_THRESHOLD = 0.7


# ── Parsing du ticket ─────────────────────────────────────────────────────────


def _parse_ticket(raw_ticket, step_id: str, cfg: dict) -> dict:
    """
    Normalise le ticket en dict compatible run_agent_spec().

    L'Orchestrateur envoie context["ticket"] comme une STRING brute
    (OrchestratorState.ticket: str).  Cette fonction la convertit en dict
    structuré avec les champs attendus par les phases BM25 / LLM.

    Si raw_ticket est déjà un dict (appel direct / tests), il est retourné
    tel quel.
    """
    if isinstance(raw_ticket, dict):
        return raw_ticket

    text = raw_ticket.strip() if isinstance(raw_ticket, str) else ""
    lines = text.splitlines()

    # Première ligne non vide → titre
    title = next((l.strip() for l in lines if l.strip()), text[:120])

    # Corps complet → description
    description = text

    return {
        "id":          step_id,
        "title":       title,
        "description": description,
        "severity":    cfg.get("severity",  ""),
        "component":   cfg.get("component", ""),
        "labels":      cfg.get("labels",    []),
    }


# ── Extraction des inputs depuis context ──────────────────────────────────────


def _extract_inputs(step_id: str, context: dict) -> dict:
    """
    Extrait et normalise tous les inputs depuis le context de l'Orchestrateur.

    Structure réelle de context["metadata"] (executor.py _prepare_context) :
        {
            "config": {          ← config dict passé à run_orchestrator_with_config
                "repo_path": str,
                "mr_diff":   str,
                "llm_model": str,   # optionnel
                "thread_id": str,   # optionnel
                "severity":  str,   # optionnel
                "component": str,   # optionnel
            },
            "history": [...],    ← ajouté par execute_step
        }

    Double fallback (metadata.get direct) pour compatibilité avec les appelants
    qui passent les clés à plat dans metadata.
    """
    metadata = context.get("metadata", {})
    cfg      = metadata.get("config", {})

    # repo_path — optionnel : Phase 0 (workspace) le résout si absent
    repo_path = (
        cfg.get("repo_path", "")
        or metadata.get("repo_path", "")
    )

    mr_diff = (
        cfg.get("mr_diff",   "")
        or metadata.get("mr_diff",   "")
    )

    # thread_id unique par ticket pour éviter les conflits de checkpoint LangGraph.
    # Priorité : fourni explicitement > hash du contenu ticket > step_id fallback.
    _raw_for_hash = context.get("ticket", "") if isinstance(context.get("ticket"), str) else str(context.get("ticket", ""))
    _ticket_hash  = hashlib.md5(_raw_for_hash.encode("utf-8", errors="replace")).hexdigest()[:12]
    thread_id = (
        cfg.get("thread_id")
        or metadata.get("thread_id")
        or _ticket_hash
        or step_id
    )

    llm_model = (
        cfg.get("llm_model")
        or metadata.get("llm_model")
        or None
    )

    ticket = _parse_ticket(context.get("ticket", ""), step_id, cfg)

    return {
        "ticket":    ticket,
        "mr_diff":   mr_diff,
        "repo_path": repo_path,
        "thread_id": thread_id,
        "llm_model": llm_model,
    }


# ── Point d'entrée public ─────────────────────────────────────────────────────


def spec_agent(step_id: str, context: dict) -> dict:
    """
    Point d'entrée de l'Agent Spec pour l'Orchestrateur.

    Args:
        step_id : Identifiant de l'étape (ex: "spec").
        context : Dict préparé par executor.py._prepare_context().

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
    inputs = _extract_inputs(step_id, context)

    ticket    = inputs["ticket"]
    mr_diff   = inputs["mr_diff"]
    repo_path = inputs["repo_path"]
    thread_id = inputs["thread_id"]
    llm_model = inputs["llm_model"]

    # ── Étape 2 : Valider les inputs obligatoires ──────────────────────────────

    # ticket doit avoir un titre exploitable
    if not ticket.get("title"):
        logger.error("[spec_agent] ticket sans titre : %r", ticket)
        return {
            "status":     "failed",
            "output":     None,
            "confidence": 0.0,
            "task_id":    step_id,
            "agent":      "spec",
            "error":      "ticket sans titre exploitable",
        }

    # repo_path : si absent, Phase 0 (workspace) tentera la découverte.
    # On laisse passer — run_agent_spec dégradera gracieusement si introuvable.
    if repo_path and not Path(repo_path).is_dir():
        logger.warning(
            "[spec_agent] repo_path fourni mais inexistant : %r — "
            "Phase 0 (workspace) tentera la découverte automatique.",
            repo_path,
        )
        repo_path = ""  # laisser Phase 0 résoudre

    if not mr_diff:
        logger.warning(
            "[spec_agent] mr_diff vide — pipeline continue sans diff "
            "(Phase 1 et Phase 2 fonctionneront en mode dégradé)."
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
            "status":     "failed",
            "output":     None,
            "confidence": 0.0,
            "task_id":    step_id,
            "agent":      "spec",
            "error":      str(exc),
        }

    # ── Étape 4 : Construire la réponse succès ─────────────────────────────────
    confidence = float(result.get("confidence", 0.0))

    response = {
        "status":     "success",
        "output":     result,
        "confidence": confidence,
        "task_id":    step_id,
        "agent":      "spec",
    }

    # ── Étape 5 : Vérifier le seuil de confiance ──────────────────────────────
    # Aligné sur router.py confidence_threshold (0.7).
    # En dessous : le Router retentera de toute façon ; on signale "failed"
    # uniquement pour les résultats vraiment inutilisables.
    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            "[spec_agent] confidence trop basse : %.2f (seuil : %.2f) — "
            "le Router gérera le retry.",
            confidence,
            CONFIDENCE_THRESHOLD,
        )
        response["status"] = "failed"
        response["reason"] = "low_confidence"

    return response
