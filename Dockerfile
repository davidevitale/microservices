# Multi-stage build for optimal image size
FROM python:3.11-slim as builder

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtual env (we're in container)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root --only main

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 agent3 && \
    mkdir -p /app /app/logs && \
    chown -R agent3:agent3 /app

# Set work directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=agent3:agent3 app/ ./app/
COPY --chown=agent3:agent3 .env .env

# Switch to non-root user
USER agent3

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Labels for metadata
LABEL maintainer="AI Engineering Team <engineering@agent3.ai>"
LABEL version="1.0.0"
LABEL description="Agent 3 - Functional Specification Generator"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8003

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--timeout-keep-alive", "300"]