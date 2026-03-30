FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY src/ src/
COPY README.md .

# Install dependencies (no dev/test/doc extras)
RUN PDM_BUILD_SCM_VERSION=1.0.0 uv sync --no-dev

# Copy entrypoint
COPY main.py .

ENV PYTHONUNBUFFERED=1

VOLUME ["/documents"]

CMD ["uv", "run", "python", "main.py"]
