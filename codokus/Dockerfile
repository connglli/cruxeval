# Dockerfile for CRUXEval OpenCode benchmark sandbox
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

# Install base dependencies, Python 3, Node.js, and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      jq \
      nodejs \
      npm \
      python3 \
      python3-pip \
      python3-venv \
      ripgrep \
      wget \
    && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCode CLI globally
RUN npm install -g opencode-ai

# Create non-root user matching host UID/GID
RUN groupadd -g ${GID} runner 2>/dev/null || true \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash runner \
    && mkdir -p /workspace \
    && chown -R ${UID}:${GID} /workspace /home/runner

WORKDIR /workspace
USER runner

# Default entrypoint executes run_opencode.py with any passed arguments
ENTRYPOINT ["python3", "codokus/run_opencode.py"]
