"""
Diff Signal Extractor — MR diff → function boost multipliers
=============================================================
Extracts three ranked signals from a unified diff and a ticket to compute
a boost multiplier for each function identified by tree-sitter.

Signals (non-cumulative — highest wins):
    Signal 1  BOOST_DIRECT_MODIFICATION  function body overlaps a + line in the diff
    Signal 2  BOOST_MODIFIED_FILE        function lives in a file touched by the diff
    Signal 3  BOOST_COMPONENT_MATCH      function file starts with ticket["component"]
    (none)    BOOST_NONE = 1.0           no signal — score unchanged

No LLM.  No external dependencies beyond stdlib.  Fully deterministic.
"""

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Boost constants (configure here) ──────────────────────────────────────────

BOOST_DIRECT_MODIFICATION: float = 3.0   # Signal 1 — function directly modified
BOOST_MODIFIED_FILE:       float = 1.5   # Signal 2 — file present in diff
BOOST_COMPONENT_MATCH:     float = 1.2   # Signal 3 — matches ticket component
BOOST_NONE:                float = 1.0   # No signal

# ── Diff parsing helpers ───────────────────────────────────────────────────────

# Matches hunk headers: @@ -L1[,N1] +L2[,N2] @@
_HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@")


def _normalise_diff_path(raw: str) -> str:
    """
    Strip the 'b/' prefix produced by unified diff and leading slashes.
    '+++ b/auth/token.py'  → 'auth/token.py'
    '+++ /dev/null'        → '/dev/null'
    """
    return raw.strip().lstrip("/")


def _normalise_file_path(fpath: str) -> str:
    """Convert backslashes to forward slashes for cross-platform comparison."""
    return fpath.replace("\\", "/")


def _file_matches_diff_path(abs_path: str, diff_path: str) -> bool:
    """
    True when the absolute repo path corresponds to the diff-relative path.
    Handles both suffix match and substring match for monorepo layouts.
    """
    norm = _normalise_file_path(abs_path)
    return norm.endswith(diff_path) or diff_path in norm


# ── DiffSignalExtractor ────────────────────────────────────────────────────────


class DiffSignalExtractor:
    """
    Parse a unified diff and a ticket to produce per-function boost multipliers.

    Usage
    -----
        extractor = DiffSignalExtractor(mr_diff, ticket)
        boosts = extractor.compute_function_boosts(functions)
        # boosts → {"auth/token.py::validate_token": 3.0, ...}
    """

    def __init__(self, mr_diff: str, ticket: dict):
        self._mr_diff = mr_diff or ""
        self._ticket  = ticket  or {}
        # Lazy caches — computed once on first access.
        self._modified_lines: Optional[Dict[str, Set[int]]] = None
        self._modified_files: Optional[Set[str]]            = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_modified_lines(self) -> Dict[str, Set[int]]:
        """
        Parse the diff and return the new-file line numbers of every + line,
        grouped by (normalised, b/-stripped) file path.

        Returns: {diff_path: {line_number, ...}}
        """
        if self._modified_lines is not None:
            return self._modified_lines

        result: Dict[str, Set[int]] = {}

        current_file: Optional[str] = None
        current_new_line: int       = 0

        for raw_line in self._mr_diff.splitlines():
            # File header: +++ b/path/to/file.py
            if raw_line.startswith("+++ "):
                suffix = raw_line[4:]
                if suffix.startswith("b/"):
                    suffix = suffix[2:]
                current_file = _normalise_diff_path(suffix)
                if current_file == "/dev/null":
                    current_file = None
                elif current_file not in result:
                    result[current_file] = set()
                current_new_line = 0
                continue

            # Old-file header — skip, does not affect new-file line counter.
            if raw_line.startswith("--- "):
                continue

            # Hunk header: @@ -L1,N1 +L2,N2 @@
            m = _HUNK_RE.match(raw_line)
            if m:
                current_new_line = int(m.group(1))
                continue

            if current_file is None:
                continue

            if raw_line.startswith("+"):
                # Added line — belongs to the new file at current_new_line.
                result[current_file].add(current_new_line)
                current_new_line += 1
            elif raw_line.startswith("-"):
                # Removed line — only in the old file; do NOT advance new counter.
                pass
            else:
                # Context line — present in both old and new file.
                current_new_line += 1

        self._modified_lines = result
        return result

    def get_modified_files(self) -> Set[str]:
        """
        Return the set of (normalised) file paths touched by the diff.
        Derived from '+++ b/…' lines; '/dev/null' is excluded.
        """
        if self._modified_files is not None:
            return self._modified_files

        files: Set[str] = set()
        for raw_line in self._mr_diff.splitlines():
            if raw_line.startswith("+++ "):
                suffix = raw_line[4:]
                if suffix.startswith("b/"):
                    suffix = suffix[2:]
                path = _normalise_diff_path(suffix)
                if path and path != "/dev/null":
                    files.add(path)

        self._modified_files = files
        return files

    def get_component_prefix(self) -> Optional[str]:
        """Return ticket['component'] stripped, or None if absent/empty."""
        comp = self._ticket.get("component", "")
        return comp.strip() or None

    def compute_function_boosts(
        self,
        functions: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute a boost multiplier for every function in *functions*.

        Args:
            functions: list of function dicts from tree-sitter
                       Each dict must have at least:
                           "file"       — absolute or repo-relative path
                           "function"   — function name
                           "start_line" — 1-based first line of the function body
                           "end_line"   — 1-based last line of the function body

        Returns:
            {function_id: boost}  where function_id = "file::function"
            Functions with no matching signal receive BOOST_NONE (1.0).

        Signal priority (non-cumulative, highest wins):
            Signal 1 > Signal 2 > Signal 3
        """
        if not self._mr_diff.strip():
            logger.warning("[DiffSignalExtractor] mr_diff is empty — all boosts = 1.0.")
            return {
                f"{fn.get('file','')}::{fn.get('function','')}": BOOST_NONE
                for fn in functions
            }

        try:
            modified_lines = self.get_modified_lines()
            modified_files = self.get_modified_files()
            component      = self.get_component_prefix()
        except Exception as exc:
            logger.warning(
                f"[DiffSignalExtractor] Diff parsing failed ({exc}) — all boosts = 1.0."
            )
            return {
                f"{fn.get('file','')}::{fn.get('function','')}": BOOST_NONE
                for fn in functions
            }

        boosts: Dict[str, float] = {}

        for fn in functions:
            fpath  = fn.get("file", "")
            fname  = fn.get("function", "")
            fn_id  = f"{fpath}::{fname}"
            start  = fn.get("start_line", 0)
            end    = fn.get("end_line",   0)

            signal = BOOST_NONE

            # ── Signal 1: function body overlaps a modified line ───────────────
            if signal < BOOST_DIRECT_MODIFICATION and start and end:
                for diff_path, lines in modified_lines.items():
                    if not _file_matches_diff_path(fpath, diff_path):
                        continue
                    # Efficient overlap check: iterate the (usually small) set
                    # of modified lines and short-circuit on first match.
                    if any(start <= ln <= end for ln in lines):
                        signal = BOOST_DIRECT_MODIFICATION
                        break

            # ── Signal 2: function is in a modified file ───────────────────────
            if signal < BOOST_MODIFIED_FILE:
                for diff_path in modified_files:
                    if _file_matches_diff_path(fpath, diff_path):
                        signal = BOOST_MODIFIED_FILE
                        break

            # ── Signal 3: function file matches the ticket component ───────────
            if signal < BOOST_MODIFIED_FILE and component:
                norm = _normalise_file_path(fpath)
                comp = component.rstrip("/")
                if norm.startswith(comp) or comp in norm:
                    signal = max(signal, BOOST_COMPONENT_MATCH)

            boosts[fn_id] = signal

        return boosts
