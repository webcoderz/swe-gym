#!/usr/bin/env bash
# Step 8: Fine-tune a model on expert trajectories using torchtune.
#
# Trains locally (no Modal/cloud). Supports:
#   - LoRA on 7B  (default, single GPU 16-24GB)
#   - Full FT 7B  (--full, 40GB+ or multi-GPU)
#   - Full FT 32B (--32b, multi-GPU with 80GB each)
#   - DPO 7B      (--dpo, after SFT, needs preference data)
#
# Configs (from SWE-smith upstream):
#   configs/sft_lora_7b.yaml     — LoRA 7B (default)
#   configs/sft_torchtune.yaml   — Full FT 7B
#   configs/sft_qwen_32b.yaml   — Full FT 32B
#   configs/dpo_qwen_7b.yaml    — DPO 7B
#
# Prerequisites:
#   - Step 7 completed (SFT JSONL data)
#   - torchtune installed (auto-installed if missing)
#   - CUDA GPU available
#
# Usage:
#   ./scripts/08_train_sft.sh                    # LoRA 7B, single GPU
#   ./scripts/08_train_sft.sh --full             # Full FT 7B
#   ./scripts/08_train_sft.sh --full --gpus 2    # Full FT 7B, multi-GPU
#   ./scripts/08_train_sft.sh --32b --gpus 2     # Full FT 32B
#   ./scripts/08_train_sft.sh --dpo              # DPO (after SFT)
#   ./scripts/08_train_sft.sh --config path.yaml # Custom config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_GYM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load repo config
source "$SCRIPT_DIR/../repo.conf"

# Defaults
FT_MODE="lora"
NUM_GPUS=1
CUSTOM_CONFIG=""
SFT_DIR="$SWE_GYM_DIR/logs/sft_data/${REPO_KEY}"
OUTPUT_DIR="$SWE_GYM_DIR/sft_output"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --full) FT_MODE="full"; shift ;;
        --32b) FT_MODE="full_32b"; shift ;;
        --dpo) FT_MODE="dpo"; shift ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --config) CUSTOM_CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Select config and recipe based on mode
case "$FT_MODE" in
    lora)
        CONFIG="${CUSTOM_CONFIG:-$SWE_GYM_DIR/configs/sft_lora_7b.yaml}"
        MODEL_ID="${SFT_BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
        if [ "$NUM_GPUS" -gt 1 ]; then
            RECIPE="lora_finetune_distributed"
        else
            RECIPE="lora_finetune_single_device"
        fi
        ;;
    full)
        CONFIG="${CUSTOM_CONFIG:-$SWE_GYM_DIR/configs/sft_torchtune.yaml}"
        MODEL_ID="${SFT_BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
        if [ "$NUM_GPUS" -gt 1 ]; then
            RECIPE="full_finetune_distributed"
        else
            RECIPE="full_finetune_single_device"
        fi
        ;;
    full_32b)
        CONFIG="${CUSTOM_CONFIG:-$SWE_GYM_DIR/configs/sft_qwen_32b.yaml}"
        MODEL_ID="Qwen/Qwen2.5-Coder-32B-Instruct"
        RECIPE="full_finetune_distributed"
        if [ "$NUM_GPUS" -lt 2 ]; then
            NUM_GPUS=2
            echo "32B requires multi-GPU, setting --gpus 2"
        fi
        ;;
    dpo)
        CONFIG="${CUSTOM_CONFIG:-$SWE_GYM_DIR/configs/dpo_qwen_7b.yaml}"
        MODEL_ID="${SFT_BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
        RECIPE="full_finetune_distributed"
        if [ "$NUM_GPUS" -lt 2 ]; then
            NUM_GPUS=2
        fi
        ;;
esac

RUN_PREFIX=""
if [ "$NUM_GPUS" -gt 1 ]; then
    RUN_PREFIX="--nproc_per_node $NUM_GPUS"
fi

echo "=== Step 8: Local SFT with torchtune ==="
echo "  Mode:     $FT_MODE"
echo "  Recipe:   $RECIPE"
echo "  Config:   $CONFIG"
echo "  Model:    $MODEL_ID"
echo "  GPUs:     $NUM_GPUS"
echo "  Data:     $SFT_DIR"
echo "  Output:   $OUTPUT_DIR"
echo ""

# Check for SFT data
SFT_FILES=$(find "$SFT_DIR" -name "ft_*.jsonl" 2>/dev/null | head -5)
if [ -z "$SFT_FILES" ]; then
    echo "ERROR: No SFT data found in $SFT_DIR"
    echo "Run ./scripts/07_prepare_sft.sh first."
    exit 1
fi

EXAMPLE_COUNT=$(cat $SFT_DIR/ft_*.jsonl 2>/dev/null | wc -l)
echo "Training examples: $EXAMPLE_COUNT"
echo ""

# Ensure torchtune + wandb are installed
if ! uv run python -c "import torchtune" 2>/dev/null; then
    echo "Installing torchtune..."
    uv pip install torchtune torch torchao wandb
    echo ""
fi

# Check wandb auth
if ! uv run wandb verify 2>/dev/null; then
    echo "WARNING: wandb not logged in. Run 'wandb login' for training dashboards."
    echo "         (Training will still work, metrics go to $OUTPUT_DIR/logs/ instead)"
    echo ""
fi

# Download base model if not cached
MODEL_CACHE="/tmp/$(basename $MODEL_ID)"
if [ ! -d "$MODEL_CACHE" ]; then
    echo "--- Downloading base model: $MODEL_ID ---"
    uv run tune download "$MODEL_ID" --output-dir "$MODEL_CACHE"
    echo ""
fi

mkdir -p "$OUTPUT_DIR"

# Combine all SFT JSONL files into one
COMBINED_SFT="$SFT_DIR/combined_train.jsonl"
cat $SFT_DIR/ft_*.jsonl > "$COMBINED_SFT"
echo "Combined SFT data: $COMBINED_SFT ($EXAMPLE_COUNT examples)"
echo ""

echo "--- Running torchtune: $RECIPE ---"
echo ""

# Run torchtune with config + path overrides
uv run tune run $RUN_PREFIX $RECIPE \
    --config "$CONFIG" \
    checkpointer.checkpoint_dir="$MODEL_CACHE" \
    checkpointer.output_dir="$OUTPUT_DIR/checkpoints/" \
    tokenizer.path="$MODEL_CACHE/vocab.json" \
    tokenizer.merges_file="$MODEL_CACHE/merges.txt" \
    dataset.data_files="$COMBINED_SFT" \
    output_dir="$OUTPUT_DIR"

echo ""
echo "=== SFT training complete ==="
echo "Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "WandB:       https://wandb.ai (project: swe-gym)"
echo ""
echo "To serve for inference:"
echo "  python -m sglang.launch_server --model-path $OUTPUT_DIR/checkpoints/ --port 8080"
