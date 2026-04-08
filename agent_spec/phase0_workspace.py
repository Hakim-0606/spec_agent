"""
Phase 0 — Workspace Discovery + Project Structure Extraction
=============================================================
Point d'entrée du pipeline.  Deux responsabilités :

1. DÉCOUVERTE — trouver le bon projet dans workspace/ à partir du ticket.
2. EXTRACTION  — parcourir l'arbre du projet et extraire la structure complète
   (dossiers, fichiers, langages, classes, fonctions + signature) via tree-sitter.
   Stockée dans state["project_structure"] pour enrichir les phases suivantes.

Pipeline :
    workspace_scan → bm25 → treesitter → rag → tools → llm → END

Aucun appel LLM.  Dépendances : pathlib, os, tree-sitter (déjà requis par Phase 2).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .constants import SKIP_DIRS, SUPPORTED_EXTENSIONS as SOURCE_EXTENSIONS

# Workspace racine — configurable via env var SPEC_WORKSPACE.
#
# IMPORTANT (déploiement pip) : quand le package est installé via
#   pip install git+https://github.com/Hakim-0606/spec_agent.git
# le fallback pointe vers le site-packages de l'environnement, pas le repo.
# Définir SPEC_WORKSPACE avec le chemin absolu du dossier partagé contenant
# les projets clonés :
#   export SPEC_WORKSPACE=/srv/shared/workspace
_DEFAULT_WORKSPACE = os.environ.get(
    "SPEC_WORKSPACE",
    str(Path(__file__).parent.parent / "workspace"),
)


# ── 1. Découverte du projet dans workspace/ ───────────────────────────────────


def find_project_path(ticket: dict, repo_path: str = "") -> str:
    """
    Retourne le chemin absolu du projet dans workspace/.

    Stratégie de recherche (ordre de priorité) :
    1. repo_path déjà fourni et valide              → retourner tel quel
    2. ticket["component"] matche un dossier         → workspace/<component>
    3. ticket["title"] contient le nom d'un dossier  → workspace/<match>
    4. ticket["id"] matche un dossier                → workspace/<id>
    5. Premier dossier trouvé dans workspace/        → fallback

    Returns:
        Chemin absolu du projet, ou "" si rien trouvé.
    """
    # Priorité 1 : repo_path déjà valide
    if repo_path and Path(repo_path).is_dir():
        logger.info("[phase0] repo_path fourni et valide : %s", repo_path)
        return repo_path

    workspace = Path(_DEFAULT_WORKSPACE)
    if not workspace.is_dir():
        logger.warning("[phase0] workspace introuvable : %s", workspace)
        return repo_path or ""

    projects = [p for p in workspace.iterdir() if p.is_dir() and p.name not in SKIP_DIRS]
    if not projects:
        logger.warning("[phase0] workspace vide : %s", workspace)
        return ""

    # Construire une liste de candidats depuis le ticket
    candidates: List[str] = []
    component = ticket.get("component", "").strip().rstrip("/")
    if component:
        candidates.append(component)

    title = ticket.get("title", "")
    # Extraire des tokens du titre (min 3 chars)
    title_tokens = [w for w in title.lower().split() if len(w) >= 3]
    candidates.extend(title_tokens)

    ticket_id = ticket.get("id", "")
    if ticket_id:
        candidates.append(ticket_id.lower())

    # Chercher le meilleur match parmi les projets
    project_names = {p.name.lower(): p for p in projects}
    for candidate in candidates:
        c = candidate.lower()
        # Match exact
        if c in project_names:
            found = project_names[c]
            logger.info("[phase0] Projet trouvé (exact) : %s", found)
            return str(found)
        # Match partiel (candidate est un sous-mot du nom de projet)
        for name, path in project_names.items():
            if c in name or name in c:
                logger.info("[phase0] Projet trouvé (partiel '%s' ~ '%s') : %s", c, name, path)
                return str(path)

    # Fallback : premier projet du workspace
    fallback = projects[0]
    logger.warning("[phase0] Aucun match — fallback sur le premier projet : %s", fallback)
    return str(fallback)


# ── 2. Extraction de la structure complète ────────────────────────────────────


def _get_lang(ext: str) -> str:
    """Retourne le nom de langage à partir d'une extension."""
    return {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".java": "java",
        ".go": "go", ".rs": "rust", ".cpp": "cpp", ".c": "c",
        ".cs": "c_sharp", ".rb": "ruby",
    }.get(ext, "unknown")


def _extract_symbols_regex(file_path: Path, lang: str) -> Tuple[List[str], List[str]]:
    """
    Fallback regex quand tree-sitter n'est pas disponible.
    Couvre Python, JS/TS, Java, Go, Rust, C#.
    """
    import re
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return [], []

    patterns = {
        "python":     (r"^class\s+(\w+)", r"^def\s+(\w+)"),
        "javascript": (r"class\s+(\w+)", r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()"),
        "typescript": (r"class\s+(\w+)", r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()"),
        "java":       (r"(?:class|interface)\s+(\w+)", r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\("),
        "go":         (r"type\s+(\w+)\s+struct", r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\("),
        "rust":       (r"(?:struct|enum|impl)\s+(\w+)", r"fn\s+(\w+)\s*[(<]"),
        "c":          (r"struct\s+(\w+)", r"^\w[\w\s\*]+\s+(\w+)\s*\("),
        "cpp":        (r"(?:class|struct)\s+(\w+)", r"^\w[\w\s\*:]+\s+(\w+)\s*\("),
        "c_sharp":    (r"(?:class|interface)\s+(\w+)", r"(?:public|private|protected|static|\s)+\w+\s+(\w+)\s*\("),
        "ruby":       (r"class\s+(\w+)", r"def\s+(\w+)"),
    }

    cls_pat, fn_pat = patterns.get(lang, (None, None))
    classes   = re.findall(cls_pat, content, re.MULTILINE) if cls_pat else []
    functions = []
    if fn_pat:
        for match in re.finditer(fn_pat, content, re.MULTILINE):
            name = next((g for g in match.groups() if g), None)
            if name:
                functions.append(name)

    # Flatten si groupes multiples (tuples)
    classes   = [c if isinstance(c, str) else c[0] for c in classes if c]
    return classes, functions


def _extract_symbols_treesitter(file_path: Path, lang: str) -> Tuple[List[str], List[str]]:
    """
    Extrait les noms de classes et de fonctions via tree-sitter (léger — noms uniquement).
    Fallback automatique sur regex si tree-sitter n'est pas installé.

    Returns:
        (classes, functions) — listes de noms de strings.
    """
    try:
        import importlib
        from tree_sitter import Language, Parser

        lang_modules = {
            "python":     "tree_sitter_python",
            "javascript": "tree_sitter_javascript",
            "typescript": "tree_sitter_typescript",
            "java":       "tree_sitter_java",
            "go":         "tree_sitter_go",
            "rust":       "tree_sitter_rust",
            "cpp":        "tree_sitter_cpp",
            "c":          "tree_sitter_c",
            "c_sharp":    "tree_sitter_c_sharp",
            "ruby":       "tree_sitter_ruby",
        }
        func_types = {
            "python":     ["function_definition"],
            "javascript": ["function_declaration", "arrow_function", "method_definition"],
            "typescript": ["function_declaration", "arrow_function", "method_definition"],
            "java":       ["method_declaration", "constructor_declaration"],
            "go":         ["function_declaration", "method_declaration"],
            "rust":       ["function_item"],
            "cpp":        ["function_definition"],
            "c":          ["function_definition"],
            "c_sharp":    ["method_declaration"],
            "ruby":       ["method", "singleton_method"],
        }
        class_types = {
            "python":     ["class_definition"],
            "javascript": ["class_declaration"],
            "typescript": ["class_declaration"],
            "java":       ["class_declaration", "interface_declaration"],
            "go":         ["type_declaration"],
            "rust":       ["struct_item", "impl_item", "enum_item"],
            "cpp":        ["class_specifier", "struct_specifier"],
            "c":          ["struct_specifier"],
            "c_sharp":    ["class_declaration", "interface_declaration"],
            "ruby":       ["class", "module"],
        }

        mod_name = lang_modules.get(lang)
        if not mod_name:
            return [], []

        mod = importlib.import_module(mod_name)
        ts_lang = Language(mod.language())
        parser = Parser(ts_lang)

        src_bytes = file_path.read_bytes()
        tree = parser.parse(src_bytes)

        def _collect_names(node, target_types: List[str]) -> List[str]:
            names = []
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur.type in target_types:
                    name_node = cur.child_by_field_name("name")
                    if name_node:
                        raw = name_node.text
                        name = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                        names.append(name.strip())
                stack.extend(reversed(cur.children))
            return names

        classes   = _collect_names(tree.root_node, class_types.get(lang, []))
        functions = _collect_names(tree.root_node, func_types.get(lang, []))
        return classes, functions

    except ImportError:
        # tree-sitter non installé → fallback regex
        return _extract_symbols_regex(file_path, lang)
    except Exception:
        return _extract_symbols_regex(file_path, lang)


def extract_project_structure(repo_path: str) -> Dict:
    """
    Parcourt récursivement repo_path et construit la carte structurelle du projet.

    Returns:
        {
            "repo_path":    str,
            "project_name": str,
            "summary": {
                "total_files":     int,
                "total_functions": int,
                "total_classes":   int,
                "languages":       {lang: file_count},
                "top_dirs":        [str],          # dossiers principaux (depth=1)
            },
            "files": [
                {
                    "path":      str,    # relatif à repo_path, séparateurs /
                    "language":  str,
                    "classes":   [str],
                    "functions": [str],
                    "loc":       int,    # nombre de lignes
                }
            ],
            "tree": {
                "dir/": {
                    "subdir/": {...},
                    "file.py": {"language": ..., "classes": [...], "functions": [...]}
                }
            }
        }
    """
    root = Path(repo_path)
    files_data: List[Dict] = []
    lang_count: Dict[str, int] = {}

    for file_path in sorted(root.rglob("*")):
        # Ignorer les dossiers blacklistés
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in SOURCE_EXTENSIONS:
            continue

        lang = _get_lang(ext)
        lang_count[lang] = lang_count.get(lang, 0) + 1

        try:
            rel = file_path.relative_to(root)
            rel_str = str(rel).replace("\\", "/")
            loc = len(file_path.read_bytes().splitlines())
        except Exception:
            continue

        classes, functions = _extract_symbols_treesitter(file_path, lang)

        files_data.append({
            "path":      rel_str,
            "language":  lang,
            "classes":   classes,
            "functions": functions,
            "loc":       loc,
        })

    # Arbre imbriqué
    tree: Dict = {}
    for f in files_data:
        parts = f["path"].split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = {
            "language":  f["language"],
            "classes":   f["classes"],
            "functions": f["functions"],
        }

    # Top-level dirs
    top_dirs = sorted({f["path"].split("/")[0] for f in files_data if "/" in f["path"]})

    total_functions = sum(len(f["functions"]) for f in files_data)
    total_classes   = sum(len(f["classes"])   for f in files_data)

    project_name = root.name

    structure = {
        "repo_path":    repo_path,
        "project_name": project_name,
        "summary": {
            "total_files":     len(files_data),
            "total_functions": total_functions,
            "total_classes":   total_classes,
            "languages":       lang_count,
            "top_dirs":        top_dirs,
        },
        "files":  files_data,
        "tree":   tree,
    }

    logger.info(
        "[phase0] Structure extraite — %d fichiers, %d fonctions, %d classes | langages: %s",
        len(files_data), total_functions, total_classes,
        ", ".join(f"{l}:{n}" for l, n in lang_count.items()),
    )
    return structure


def _format_structure_for_prompt(structure: Dict, max_files: int = 60) -> str:
    """
    Génère un résumé textuel compact de la structure projet pour le prompt LLM.

    Format :
        project: my-service  (42 files, 187 functions, 23 classes)
        languages: python:30, go:12
        dirs: auth/, api/, core/, tests/

        auth/token.py [python] — classes: TokenManager | functions: verify_token, generate_token
        api/routes.py [python] — functions: get_user, post_user
        ...
    """
    summary = structure.get("summary", {})
    lines = [
        f"project: {structure.get('project_name', '?')}  "
        f"({summary.get('total_files', 0)} files, "
        f"{summary.get('total_functions', 0)} functions, "
        f"{summary.get('total_classes', 0)} classes)",
        f"languages: {', '.join(f'{l}:{n}' for l, n in summary.get('languages', {}).items())}",
        f"dirs: {', '.join(summary.get('top_dirs', [])[:10])}",
        "",
    ]

    files = structure.get("files", [])[:max_files]
    for f in files:
        parts = []
        if f["classes"]:
            parts.append("classes: " + ", ".join(f["classes"][:5]))
        if f["functions"]:
            fn_list = ", ".join(f["functions"][:8])
            if len(f["functions"]) > 8:
                fn_list += f" (+{len(f['functions']) - 8})"
            parts.append("functions: " + fn_list)
        detail = " | ".join(parts) if parts else "(no symbols)"
        lines.append(f"{f['path']} [{f['language']}] — {detail}")

    if len(structure.get("files", [])) > max_files:
        lines.append(f"... ({len(structure['files']) - max_files} more files)")

    return "\n".join(lines)


# ── LangGraph node ─────────────────────────────────────────────────────────────


def phase_workspace(state: dict) -> dict:
    """
    LangGraph node — Phase 0.

    Reads:  ticket, repo_path (optionnel), metadata (optionnel)
    Writes: repo_path (confirmé/découvert), project_structure

    Fallback :
        Si le projet est introuvable, repo_path reste inchangé
        et project_structure est vide → le pipeline continue
        (les phases suivantes dégradent gracieusement).
    """
    ticket    = state.get("ticket", {})
    repo_path = state.get("repo_path", "")

    try:
        # Découverte du projet
        resolved_path = find_project_path(ticket, repo_path)

        if not resolved_path:
            logger.warning("[phase0] Projet introuvable — pipeline continue sans structure")
            return {
                "repo_path":         repo_path,
                "project_structure": {},
            }

        # Extraction de la structure complète
        structure = extract_project_structure(resolved_path)

        return {
            "repo_path":         resolved_path,   # mis à jour si découvert via workspace
            "project_structure": structure,
        }

    except Exception:
        logger.exception("[phase0] Erreur dans phase_workspace — pipeline non bloqué")
        return {
            "repo_path":         repo_path,
            "project_structure": {},
        }
