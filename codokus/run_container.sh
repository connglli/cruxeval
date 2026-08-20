#!/usr/bin/env bash
# ==============================================================================
# run_container.sh — Build and run agent benchmark in a dedicated Docker image
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-cruxeval-agent:latest}"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

# Check for --rebuild flag
REBUILD=0
FORWARD_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--rebuild" ]]; then
    REBUILD=1
  else
    FORWARD_ARGS+=("$arg")
  fi
done

# Build Docker image if not present or if --rebuild specified
if [[ "$REBUILD" -eq 1 ]] || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "📦 Building Docker image '${IMAGE_NAME}' (Ubuntu 24.04 + OpenCode + Claude Code)..."
  docker build \
    -t "${IMAGE_NAME}" \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    -f "${DOCKERFILE}" \
    "${SCRIPT_DIR}"
  echo "✅ Docker image built successfully."
  echo ""
fi

# Detect relevant host environment variables (API keys and OpenCode settings)
CANDIDATE_VARS=(
  OPENROUTER_API_KEY
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  DEEPSEEK_API_KEY
  GEMINI_API_KEY
  OPENCODE_API_KEY
  OPENCODE_MODEL
  OPENCODE_BASE_URL
)

FOUND_VARS=()
for var in "${CANDIDATE_VARS[@]}"; do
  if [[ -n "${!var:-}" ]]; then
    FOUND_VARS+=("${var}")
  fi
done

ENV_FLAGS=()
if [[ ${#FOUND_VARS[@]} -gt 0 ]]; then
  echo "🔍 Detected the following environment variable(s) on host:"
  for var in "${FOUND_VARS[@]}"; do
    echo "   • ${var}"
  done
  echo ""

  read -r -p "Do you want to pass these environment variable(s) to the container? [y/N]: " user_choice
  if [[ "${user_choice,,}" =~ ^(y|yes)$ ]]; then
    echo "🔑 Passing detected environment variables to container..."
    for var in "${FOUND_VARS[@]}"; do
      ENV_FLAGS+=("-e" "${var}=${!var}")
    done
  else
    echo "🔒 Host environment variables will NOT be passed to the container."
  fi
else
  echo "ℹ️  No provider environment variables detected on host."
fi

echo "🚀 Launching container '${IMAGE_NAME}'..."

docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${ENV_FLAGS[@]}" \
  "${IMAGE_NAME}" \
  "${FORWARD_ARGS[@]}"
