#!/usr/bin/env bash
# Step 2: Synthesize bug-introducing task instances from the repo.
#
# Uses 5 strategies to generate diverse bugs:
#   1. LM Modify     - LLM introduces subtle bugs into functions (randomized prompts)
#   2. Class Basic   - LLM introduces bugs at class level (multi-method bugs)
#   3. Func Fun      - "tired developer" persona for realistic function bugs
#   4. LM Rewrite    - LLM rewrites functions from scratch (natural errors)
#   5. Procedural    - AST rule-based transformations (no LLM needed)
#
# Prerequisites:
#   - Step 1 completed (Docker image exists)
#   - OPENAI_API_KEY set
#
# Usage:
#   ./scripts/02_generate_tasks.sh              # run all steps
#   ./scripts/02_generate_tasks.sh --from 2d    # resume from step 2d

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load repo config
source "$SCRIPT_DIR/../repo.conf"

# Parse --from flag for resume support
START_STEP="2a"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) START_STEP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Helper: returns 0 (true) if current step should run
should_run() {
    local steps=("2a" "2b" "2c" "2d" "2e")
    local start_idx=-1 cur_idx=-1
    for i in "${!steps[@]}"; do
        [[ "${steps[$i]}" == "$START_STEP" ]] && start_idx=$i
        [[ "${steps[$i]}" == "$1" ]] && cur_idx=$i
    done
    [[ $cur_idx -ge $start_idx ]]
}

# All commands go through swesmith_run.py which registers the private repo profile
RUN="uv run python $SCRIPT_DIR/swesmith_run.py"

echo "=== Step 2: Synthesizing task instances ==="
[[ "$START_STEP" != "2a" ]] && echo "(resuming from step $START_STEP)"
echo ""

# Config dir (upstream SWE-smith bug_gen configs)
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs/bug_gen" && pwd)"

# --- LM Modify (function-level, randomized prompts) ---
if should_run "2a"; then
echo "--- [2a] LM Modify: ~$BUGGEN_LM_MODIFY bugs ($BUGGEN_N_PER_FUNC per function) ---"
$RUN swesmith.bug_gen.llm.modify "$REPO_KEY" \
    --config_file "$CONFIG_DIR/lm_modify.yml" \
    --n_bugs "$BUGGEN_N_PER_FUNC" \
    --max_bugs "$BUGGEN_LM_MODIFY" \
    --model "$BUGGEN_MODEL"
fi

# --- Class Basic (class-level bugs) ---
if should_run "2b"; then
echo ""
echo "--- [2b] Class Basic: ~$BUGGEN_CLASS_BASIC bugs ($BUGGEN_N_PER_FUNC per class) ---"
$RUN swesmith.bug_gen.llm.modify "$REPO_KEY" \
    --config_file "$CONFIG_DIR/class_basic.yml" \
    --n_bugs "$BUGGEN_N_PER_FUNC" \
    --max_bugs "$BUGGEN_CLASS_BASIC" \
    --model "$BUGGEN_MODEL"
fi

# --- Func Fun (tired developer persona) ---
if should_run "2c"; then
echo ""
echo "--- [2c] Func Fun: ~$BUGGEN_FUNC_FUN bugs ($BUGGEN_N_PER_FUNC per function) ---"
$RUN swesmith.bug_gen.llm.modify "$REPO_KEY" \
    --config_file "$CONFIG_DIR/func_fun.yml" \
    --n_bugs "$BUGGEN_N_PER_FUNC" \
    --max_bugs "$BUGGEN_FUNC_FUN" \
    --model "$BUGGEN_MODEL"
fi

# --- LM Rewrite (reimplementation from scratch) ---
# Note: rewrite.py does NOT accept --n_bugs (always 1 rewrite per candidate)
if should_run "2d"; then
echo ""
echo "--- [2d] LM Rewrite: ~$BUGGEN_LM_REWRITE bugs (1 per function) ---"
$RUN swesmith.bug_gen.llm.rewrite "$REPO_KEY" \
    --config_file "$CONFIG_DIR/lm_rewrite.yml" \
    --max_bugs "$BUGGEN_LM_REWRITE" \
    --model "$BUGGEN_MODEL"
fi

# --- Procedural (AST transforms, no LLM) ---
if should_run "2e"; then
echo ""
echo "--- [2e] Procedural: ~$BUGGEN_PROCEDURAL bugs (AST rules, no LLM) ---"
$RUN swesmith.bug_gen.procedural.generate "$REPO_KEY" \
    --max_bugs "$BUGGEN_PROCEDURAL"
fi

echo ""
echo "=== Task synthesis complete ==="
echo "Generated tasks in: logs/bug_gen/$REPO_KEY/"
echo ""
echo "Next: run ./scripts/03_validate_tasks.sh to filter valid instances."
