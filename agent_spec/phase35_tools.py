"""
Phase 3.5 — Tools déterministes (read_file + search_in_repo)
=============================================================
Nœud LangGraph inséré entre Phase 3 (RAG) et Phase 4 (LLM).
Enrichit ast_functions avec le code source réel (source_real)
et produit des résultats de recherche dans le repo (tool_search_results).

0 appel LLM.  0 dépendance externe.  stdlib uniquement.
"""

import logging
import re
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from .constants import SKIP_DIRS as _SKIP_DIRS, SUPPORTED_EXTENSIONS as _EXTENSIONS
_STOP_WORDS: set = {
    "with", "from", "that", "this", "when", "where", "which", "what",
    "then", "than", "there", "their", "they", "been", "were", "will",
    "would", "could", "should", "about", "into", "does", "have", "error",
    "raise", "return", "false", "true", "null", "none", "class", "import",
}
_SEARCH_TIMEOUT = 30  # secondes


# ── Tool 1 — read_file ────────────────────────────────────────────────────────


def read_file(path: str, start_line: int, end_line: int, repo_path: str) -> str:
    """
    Lit les lignes [start_line-5 : end_line+5] d'un fichier source.

    Args:
        path      : Chemin relatif à repo_path (ou absolu).
        start_line: Première ligne d'intérêt (1-based).
        end_line  : Dernière ligne d'intérêt (1-based).
        repo_path : Chemin absolu du repo cloné.

    Returns:
        Snippet annoté "L<n>: <code>" ou "" si le fichier est introuvable.

    Note:
        Pathlib résout automatiquement les chemins absolus :
        Path(repo_path) / absolute_path == absolute_path (comportement stdlib).
        La fonction fonctionne donc avec des chemins relatifs ET absolus.
    """
    abs_path = Path(repo_path) / path
    if not abs_path.is_file():
        logger.warning("[phase35] read_file: fichier introuvable : %s", abs_path)
        return ""

    try:
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        logger.warning("[phase35] read_file: erreur lecture %s — %s", abs_path, exc)
        return ""

    n = len(lines)
    # ±5 lignes de contexte, clippé aux bornes du fichier.
    start_idx = max(0, (start_line - 1) - 5)
    end_idx   = min(n, end_line + 5)

    snippet = "\n".join(
        f"L{start_idx + i + 1}: {line}"
        for i, line in enumerate(lines[start_idx:end_idx])
    )
    return snippet


# ── Tool 2 — search_in_repo ───────────────────────────────────────────────────


def search_in_repo(
    pattern: str,
    repo_path: str,
    file_extensions: Optional[List[str]] = None,
) -> List[dict]:
    """
    Cherche pattern (regex) dans tous les fichiers sources du repo.

    Args:
        pattern        : Pattern regex (re.search).
        repo_path      : Chemin absolu du repo.
        file_extensions: Extensions à scanner (défaut: .py .js .ts .java .go).

    Returns:
        Liste de max 10 résultats : [{file, line, content, match}].
        Liste vide si aucun résultat ou timeout 30 s.
    """
    ext_set = set(file_extensions) if file_extensions else _EXTENSIONS

    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        logger.warning(
            "[phase35] search_in_repo: pattern regex invalide %r — %s", pattern, exc
        )
        return []

    timed_out = threading.Event()
    timer = threading.Timer(_SEARCH_TIMEOUT, timed_out.set)
    timer.start()

    results: List[dict] = []
    try:
        repo = Path(repo_path)
        for file_path in repo.rglob("*"):
            if timed_out.is_set():
                logger.warning(
                    "[phase35] search_in_repo: timeout après %ds", _SEARCH_TIMEOUT
                )
                break
            if len(results) >= 10:
                break

            # Ignorer les répertoires blacklistés.
            if any(skip in file_path.parts for skip in _SKIP_DIRS):
                continue
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in ext_set:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                for lineno, line in enumerate(content.splitlines(), start=1):
                    if timed_out.is_set() or len(results) >= 10:
                        break
                    m = compiled.search(line)
                    if m:
                        try:
                            rel = str(file_path.relative_to(repo)).replace("\\", "/")
                        except ValueError:
                            rel = str(file_path).replace("\\", "/")
                        results.append({
                            "file":    rel,
                            "line":    lineno,
                            "content": line.strip(),
                            "match":   m.group(0),
                        })
            except Exception:
                continue
    finally:
        timer.cancel()

    return results


# ── Extraction des termes de recherche ────────────────────────────────────────


def _extract_search_terms(ticket: dict, rag_contexts: List[dict]) -> List[str]:
    """
    Extrait jusqu'à 3 termes significatifs :
    - mots > 4 chars du ticket["title"] (hors stop-words)
    - root_cause du premier rag_context si disponible
    """
    terms: List[str] = []

    title_words = re.findall(r"\b[a-zA-Z_]\w*\b", ticket.get("title", ""))
    for word in title_words:
        if len(word) > 4 and word.lower() not in _STOP_WORDS:
            terms.append(word)
        if len(terms) >= 2:
            break

    if rag_contexts:
        # Chercher dans root_cause ou, à défaut, dans source (code parsé)
        root_cause = rag_contexts[0].get("root_cause", "") or rag_contexts[0].get("source", "")
        rc_words = re.findall(r"\b[a-zA-Z_]\w*\b", root_cause)
        for word in rc_words:
            if (
                len(word) > 4
                and word.lower() not in _STOP_WORDS
                and word not in terms
            ):
                terms.append(word)
                break

    return terms[:3]


# ── LangGraph node ─────────────────────────────────────────────────────────────


def phase_tools(state: dict) -> dict:
    """
    LangGraph node — Phase 3.5.

    Reads:  ast_functions (top-5), repo_path, ticket, rag_contexts
    Writes: ast_functions (enrichi avec source_real), tool_search_results

    Fallback : en cas d'exception, retourne l'état ast_functions intact
    et tool_search_results=[] pour ne pas bloquer le pipeline.
    """
    try:
        ast_functions: List[dict] = list(state.get("ast_functions", []))
        repo_path:     str        = state.get("repo_path", "")
        ticket:        dict       = state.get("ticket", {})
        rag_contexts:  List[dict] = state.get("rag_contexts", [])

        # ── Étape 1–2 : enrichir chaque fonction avec source_real ─────────────
        for fn in ast_functions:
            file_path = fn.get("file", "")
            source_real = read_file(
                path=file_path,
                start_line=fn.get("start_line", 1),
                end_line=fn.get("end_line", 1),
                repo_path=repo_path,
            )
            fn["source_real"] = source_real

        # ── Étapes 3–4 : termes de recherche + search_in_repo ─────────────────
        terms = _extract_search_terms(ticket, rag_contexts)

        seen: set = set()
        tool_search_results: List[dict] = []

        for term in terms:
            hits = search_in_repo(term, repo_path)
            for hit in hits:
                key = (hit["file"], hit["line"])
                if key not in seen:
                    seen.add(key)
                    tool_search_results.append(hit)

        return {
            "ast_functions":       ast_functions,
            "tool_search_results": tool_search_results,
        }

    except Exception:
        logger.exception("[phase35] Erreur dans phase_tools — pipeline non bloqué")
        return {
            "ast_functions":       state.get("ast_functions", []),
            "tool_search_results": [],
        }
