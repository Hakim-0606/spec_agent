from typing import TypedDict


class SpecState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    ticket: dict       # Structured GitLab ticket
    mr_diff: str       # Raw unified diff of the MR
    repo_path: str     # Absolute path to the locally cloned repo

    # ── LLM override ──────────────────────────────────────────────────────────
    llm_model: str   # Ollama model name — optionnel, défaut via $OLLAMA_MODEL

    # ── Phase 0 — Workspace & Project Structure ───────────────────────────────
    # Carte structurelle complète du projet extraite avant BM25.
    # {repo_path, project_name, summary:{total_files,functions,classes,languages},
    #  files:[{path,language,classes,functions,loc}], tree:{dir/:{...}}}
    # Vide ({}) si le projet est introuvable dans workspace/.
    project_structure: dict

    # ── Phase 1 — BM25 ────────────────────────────────────────────────────────
    keywords: list     # Tokens extracted from ticket + diff
    bm25_files: list   # Top-10 candidate files: [{"file": str, "score": float}]

    # ── Phase 2 — tree-sitter + NetworkX ─────────────────────────────────────
    # Stored as nx.node_link_data dict for JSON-serialisable checkpointing.
    repo_graph: dict
    # Top-5 scored functions: [{
    #   file, function, class, start_line, end_line,
    #   source, signature, callers, callees, score, language,
    #   diff_boost,   ← MR diff signal multiplier (1.0 = no signal)
    #   final_score   ← score × diff_boost (ranking key)
    # }]
    ast_functions: list
    # All functions parsed from bm25_files (used by Reflexion in phase 4).
    all_functions: list

    # ── Phase 3 — RAG ─────────────────────────────────────────────────────────
    rag_contexts: list  # Top-3 re-ranked by semantic similarity

    # ── Phase 3.5 — Tools déterministes ───────────────────────────────────────
    # Résultats de search_in_repo : [{file, line, content, match}, …]
    # Vide si tools non disponibles ou aucun résultat.
    tool_search_results: list

    # ── Phase 4 — LLM + Reflexion ─────────────────────────────────────────────
    # Full location document for the Agent Coder.  All 13 fields:
    #   Existing : file, function, line, root_cause, confidence,
    #              callers, callees, language
    #   New      : problem_summary   — structured 3-sentence bug description
    #              code_context      — annotated snippet with # BUG: comments
    #              patch_constraints — {scope, preserve_tests,
    #                                   forbidden_files, style_hint}
    #              expected_behavior — what the code should do after the patch
    #              fallback_locations — [{file, function, reason}, …]
    location: dict
    confidence: float

    # ── Phase 1 debug / traceability ──────────────────────────────────────────
    # Populated by the RRF fusion step; empty list when embedding is unavailable.
    # [{file, rrf_score, bm25_rank, embed_rank}, …]
    rrf_scores: list
