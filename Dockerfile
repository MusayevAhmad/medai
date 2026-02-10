# ---------------------------------------------------------------------------
# BioScholar FastAPI – Multi-stage Dockerfile
#
# Stage 1: Install Python dependencies (cached layer)
# Stage 2: Copy source code + model weights → run the API
#
# Build:
#   docker build -t bioscholar .
#
# Run (standalone, expects Qdrant on the host or via docker-compose):
#   docker run -p 8000:8000 \
#       -v $(pwd)/outputs/models:/app/outputs/models:ro \
#       -v $(pwd)/data/qdrant_db:/app/data/qdrant_db \
#       bioscholar
# ---------------------------------------------------------------------------

# ---- Stage 1: Dependencies ------------------------------------------------
FROM python:3.11-slim AS deps

WORKDIR /app

# System packages needed for PyMuPDF (mupdf) and building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first (keeps image ~2 GB smaller than the CUDA build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Application -------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ src/
COPY app/ app/
COPY config.yaml .

# Create directories that the app expects
RUN mkdir -p outputs/logs data/qdrant_db

# Model weights are mounted at runtime (see docker-compose.yml)
# so the image stays small and models can be swapped without rebuilding.

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Health check — hits the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
