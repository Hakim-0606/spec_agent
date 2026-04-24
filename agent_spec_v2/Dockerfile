FROM python:3.11-slim

# gcc  : compiles tree-sitter grammars at install time
# git  : used by get_recent_commits and git_diff tools at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so this layer is cached when only code changes.
COPY agent_spec_v2/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
# agent_spec (v1) is required: agent_spec_v2 tools delegate to its modules
# (phase1_bm25, phase2_treesitter, phase3_rag, phase35_tools, phase4_llm).
COPY agent_spec/   ./agent_spec/
COPY agent_spec_v2/ ./agent_spec_v2/

# ── Non-root user (security best practice) ───────────────────────────────────
RUN useradd -m -u 1000 agent
USER agent

# ── Runtime configuration ─────────────────────────────────────────────────────
# Override these with environment variables or a .env file at runtime:
#   LM_STUDIO_URL   e.g. http://host.docker.internal:1234
#   OLLAMA_HOST     e.g. http://host.docker.internal:11434
#   OLLAMA_MODEL    e.g. qwen3:8b
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "agent_spec_v2.main:app", "--host", "0.0.0.0", "--port", "8001"]
