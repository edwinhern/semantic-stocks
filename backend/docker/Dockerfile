# Build stage
FROM python:3.13-slim AS builder

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only dependency files first (for better layer caching)
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Runtime stage
FROM python:3.13-slim

# Copy only the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy only the application code (not dependencies files, README, etc.)
WORKDIR /app
COPY app/ ./app/

# Ensure the virtual environment is in PATH and Python can find the app package
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# Run the application.
CMD ["fastapi", "run", "app/main.py", "--port", "8080", "--host", "0.0.0.0"]