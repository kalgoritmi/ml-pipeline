# Build stage: install dependencies and run tests
FROM python:3.11-slim AS builder

WORKDIR /app

# Install poetry
ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files first (layer caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --only main

# Copy source code and test configs
COPY src/ src/
COPY tests/ tests/
COPY config.yaml ./

# Install project
RUN poetry install --only main

# Run tests (fail build if tests fail)
RUN poetry run python -m unittest discover tests/ -v

# Production stage: minimal runtime image
FROM python:3.11-slim AS production

# Locale and timezone (override at runtime with -e TZ=Europe/London)
ARG TZ=UTC
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=${TZ}

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app/data /app/checkpoints && \
    chown -R app:app /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --chown=app:app src/ src/
COPY --chown=app:app config.yaml ./

# Set environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app

# Default config, can be overridden with -v /path/to/config.yaml:/app/config.yaml
ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["config.yaml"]
