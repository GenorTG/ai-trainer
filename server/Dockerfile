FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python 3.13
RUN apt-get update && apt-get install -y \
    python3.13 python3.13-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN python3.13 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install -e ".[all]"

# Create directories
RUN mkdir -p rag_data/store rag_data/documents models

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run
CMD ["python3.13", "-m", "src.cli", "--config", "/app/config.yaml"]
