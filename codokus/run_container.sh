#!/usr/bin/env bash
# ==============================================================================
# run_container.sh — Run OpenCode benchmark inside default ubuntu:24.04 container
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UBUNTU_IMAGE="${UBUNTU_IMAGE:-ubuntu:24.04}"

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

echo "🚀 Launching container with ${UBUNTU_IMAGE}..."

docker run --rm -i \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${ENV_FLAGS[@]}" \
  "${UBUNTU_IMAGE}" \
  bash -c '
    set -euo pipefail
    echo "📦 Installing Python, Node.js, and OpenCode (npm install -g opencode-ai)..."
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      python3 nodejs npm ca-certificates curl
    npm install -g opencode-ai
    echo "✅ OpenCode installed. Running benchmark..."
    python3 codokus/run_opencode.py "$@"
  ' -- "$@"
