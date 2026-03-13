#!/usr/bin/env bash
# Step 1: Build SWE-Smith Docker execution environment for the target repo.
#
# For private repos, uses build_private_env.py which:
#   1. Registers a custom profile at runtime
#   2. Builds Docker image with HTTPS + gh auth token
#   3. Names the image to match swesmith's conventions
#
# Prerequisites:
#   - Docker running (verify with `docker ps`)
#   - gh CLI authenticated (gh auth login)
#
# Usage:
#   ./scripts/01_build_env.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load repo config
source "$SCRIPT_DIR/../repo.conf"

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

echo "=== Step 1: Building SWE-Smith execution environment ==="
echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "Repo root:  $REPO_ROOT"
echo ""

# Ensure uv is installed (needed for all Python tooling)
if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "ERROR: uv installation failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    echo "uv installed: $(uv --version)"
    echo ""
fi

# Ensure the swe-gym venv and deps are installed
SWE_GYM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -d "$SWE_GYM_DIR/.venv" ]; then
    echo "Setting up swe-gym Python environment..."
    (cd "$SWE_GYM_DIR" && uv sync)
    echo ""
fi

# Check wandb auth (needed for SFT training logging)
if command -v wandb &> /dev/null; then
    if ! wandb verify 2>/dev/null; then
        echo "TIP: Run 'wandb login' to enable training dashboards (free account)"
        echo ""
    fi
else
    echo "TIP: wandb not installed yet — will be available after 'make setup'"
    echo "     Run 'wandb login' before SFT training for live dashboards"
    echo ""
fi

# Verify Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker and retry."
    exit 1
fi

# Verify gh CLI is authenticated
if ! gh auth status > /dev/null 2>&1; then
    echo "WARNING: gh CLI not authenticated. Run: gh auth login"
    echo ""
fi

# Use build_private_env.py for private repo support
uv run python "$SCRIPT_DIR/build_private_env.py" \
    --owner "$REPO_OWNER" \
    --repo "$REPO_NAME" \
    --python-version "$PYTHON_VERSION" \
    $FORCE_FLAG

echo ""
echo "=== Environment built successfully ==="
REPO_LOWER=$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]')
echo "Verify with:"
echo "  docker images | grep swesmith | grep $REPO_LOWER"
echo ""
echo "Test interactively with:"
echo "  docker run -it --rm \$(docker images --format '{{.Repository}}:{{.Tag}}' | grep $REPO_LOWER | head -1) bash"
