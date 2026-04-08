"""
agent_spec/constants.py — Shared constants across all pipeline phases.
"""

# Directories to skip when walking a repository.
# Shared by phase0_workspace, phase1_bm25, phase35_tools, embedding_indexer.
SKIP_DIRS: set = {
    ".git", "node_modules", "__pycache__", "vendor",
    ".venv", "venv", "env",
    "dist", "build", "target", "out",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
}

# Source-file extensions considered by the pipeline.
SUPPORTED_EXTENSIONS: set = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".cs", ".rb", ".php", ".kt", ".swift",
}
