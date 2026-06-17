# Backend image: the rca-agent FastAPI server.
# Uses uv for reproducible deps. The benchmark dataset is NOT baked in — mount
# it at runtime via RCA_CASES_DIR (see docker-compose.prod.yml).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install runtime deps first (cache layer). README.md is required by hatchling
# (pyproject declares readme = "README.md") to build the editable package, so it
# must be present alongside pyproject.toml at this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# App code + memory seed (small). Cases/infra are mounted, not baked.
COPY rca_agent ./rca_agent
COPY memory ./memory

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uv", "run", "uvicorn", "rca_agent.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
