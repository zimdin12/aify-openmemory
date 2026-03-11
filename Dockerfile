FROM python:3.12-slim

# System dependencies (extend as needed for your service)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with docker group access
# The GID should match the host's docker group - override via DOCKER_GID build arg
ARG DOCKER_GID=999
RUN groupadd -g ${DOCKER_GID} docker 2>/dev/null || true && \
    useradd -m -s /bin/bash service && \
    usermod -aG docker service && \
    mkdir -p /app /data && \
    chown -R service:service /app /data

WORKDIR /app

# Install Python dependencies
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY service/ ./service/
COPY mcp_local/ ./mcp_local/
COPY config/ ./config/

# The /data directory is where named volumes mount for persistence
VOLUME /data

EXPOSE 8800

# Run as root — needs Docker socket access to manage sub-containers
# USER service

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8800/health || exit 1

CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8800"]
