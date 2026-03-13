#!/usr/bin/env python3
"""Step 9: GRPO reinforcement learning on gym tasks using Unsloth.

Instead of SFT (imitating expert traces), GRPO lets the model learn by
actually trying to solve bugs and getting rewarded for correct patches.
The gym tasks from step 3 become the reward signal — if the patch passes
the test suite, reward=1; otherwise reward=0.

This is the "true gym" path: the model does reps, gets feedback, gets stronger.

Requires:
    - Step 3 completed (validated task instances)
    - A base model (or SFT checkpoint from step 8)
    - GPU with enough VRAM (4-bit: ~8GB for 4B, ~16GB for 8B, ~24GB for 30B MoE)
    - Docker running (for test harness evaluation)

Single GPU:
    python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml
    python scripts/09_train_grpo.py --model Qwen/Qwen3-4B

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_8b.yaml

Multi-GPU (model splitting for large models):
    python scripts/09_train_grpo.py --recipe configs/unsloth/grpo_qwen3_coder_next.yaml --device-map balanced

From SFT checkpoint:
    python scripts/09_train_grpo.py --from-sft ./sft_output/merged

Reward modes:
    python scripts/09_train_grpo.py --reward-mode test     # Full Docker test suite
    python scripts/09_train_grpo.py --reward-mode format   # Fast format check
    python scripts/09_train_grpo.py --reward-mode hybrid   # 30% format + 70% test (default)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


# ── Full YAML recipe schema (all Unsloth GRPO capabilities) ──────
# Every field below can be set in a recipe YAML or overridden via CLI.
RECIPE_DEFAULTS = {
    # ── Model ──
    "model": "Qwen/Qwen3-8B",
    "mode": "grpo",
    "dtype": None,              # None (auto), "bfloat16", "float16"
    "token": None,              # HF token for gated models (Llama, Gemma); or set HF_TOKEN env var
    "trust_remote_code": False, # Required for some newer model architectures
    "revision": None,           # Pin specific model revision from Hub (e.g. "main", commit hash)
    "resize_model_vocab": None, # Resize vocab (int) — for adding custom special tokens

    # ── Quantization ──
    "four_bit": True,           # QLoRA 4-bit (minimal VRAM)
    "eight_bit": False,         # 8-bit quantization
    "fp8": False,               # FP8 quantization (RTX 40/50, H100+; 60% less VRAM, 1.4x faster)
    "load_in_16bit": False,     # Force 16-bit loading (required for QAT)
    "offload_embedding": False, # Saves ~1GB VRAM for large models (e.g. GPT-OSS)

    # ── QAT (Quantization-Aware Training) ──
    "qat_scheme": None,         # None = no QAT; "int4", "int8-int4", "fp8-int4", "fp8-fp8"
    "save_torchao": False,      # Save QAT model via save_pretrained_torchao()

    # ── LoRA config ──
    "lora_rank": 64,            # r: 8, 16, 32, 64, 128, 256
    "lora_alpha": 64,           # Usually = rank, or rank*2 for aggressive learning
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0,          # 0 recommended; Unsloth optimizes this
    "bias": "none",             # "none" (faster) or "all"
    "use_rslora": False,        # Rank-Stabilized LoRA: scales alpha by 1/sqrt(r)
    "loftq_config": None,       # LoftQ initialization config
    "init_lora_weights": True,  # True (default), False, or "gaussian"
    "layers_to_transform": None,  # List of layer indices to apply LoRA to (e.g. [24,25,26,27])
    "modules_to_save": None,    # Extra modules to save

    # ── Gradient checkpointing ──
    "gradient_checkpointing": "unsloth",  # "unsloth" (30% extra savings), True, False

    # ── Training hyperparams ──
    "lr": 5e-6,
    "steps": 200,               # GRPO uses steps, not epochs
    "batch_size": 4,            # Problems per batch
    "num_generations": 4,       # Completions per problem
    "max_seq_len": 10240,
    "max_completion_len": 4096,
    "optim": "adamw_8bit",      # "adamw_8bit", "adamw_torch", "sgd", etc.
    "warmup_steps": 10,
    "warmup_ratio": None,       # Alternative to warmup_steps (e.g. 0.1 = 10% of steps)
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",   # "cosine", "linear", "constant", etc.
    "seed": 42,

    # ── GRPO loss & clipping ──
    "loss_type": "grpo",        # "grpo", "bnpo", "dr_grpo", "dapo"
    "beta": 0.0,                # KL penalty coefficient; 0.0 = no reference model (saves memory)
    "num_iterations": 1,        # PPO epochs per batch (replays within grad accum)
    "epsilon": 0.2,             # PPO clipping epsilon (standard)
    "epsilon_high": None,       # One-sided high clip (e.g. 0.28) — None disables; DAPO uses 0.28
    "delta": None,              # Two-sided clip (e.g. 1.5) — None disables; recommended > 1+epsilon
    "mask_truncated_completions": True,  # Ignore truncated completions in loss
    "importance_sampling_level": "token",  # "token" (GRPO) or "sequence" (GSPO/Qwen-style)
    "scale_rewards": "group",   # "group" (per-group norm), "batch", or "none"
    "steps_per_generation": None,  # Microbatches per generation; None = auto

    # ── Off-policy corrections (vLLM importance sampling) ──
    "vllm_importance_sampling_correction": True,  # Truncated Importance Sampling when using vLLM
    "vllm_importance_sampling_cap": 2.0,  # TIS truncation parameter C

    # ── Reward ──
    "reward_mode": "hybrid",    # "test", "format", "hybrid"
    "reward_test_weight": 0.7,  # Weight for test reward in hybrid mode
    "reward_format_weight": 0.3,  # Weight for format reward in hybrid mode
    "reward_weights": None,     # Per-reward-function weights (list[float])

    # ── vLLM inference ──
    "temperature": 1.5,
    "min_p": 0.1,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": None, # Token reuse penalty (1.0 = no penalty)
    "gpu_memory_utilization": 0.6,  # 0.6 default; set 0.95 with vllm_standby
    "vllm_standby": False,      # Share weight memory between train/inference (~9GB savings)
    "float8_kv_cache": False,   # 2x KV cache reduction (RTX 3090+/A100+)

    # ── GRPO long-context memory optimization ──
    "unsloth_grpo_mini_batch": None,  # Batch dim chunking for hidden states; None = auto
    "unsloth_logit_chunk_multiplier": None,  # Seq dim chunking; None = auto (max(4, ctx//4096))

    # ── MoE backend ──
    "moe_backend": None,        # "grouped_mm" (default), "unsloth_triton", "native_torch"

    # ── Reward hacking countermeasures ──
    "reward_stdlib_only": False,  # Penalize non-stdlib imports (-20)
    "reward_strip_globals": False,  # Strip global-scope mutations
    "reward_max_cache_mb": None,  # Cache thrashing limit (e.g. 2048 for 2GB)
    "reward_timeout": None,     # Per-evaluation timeout in seconds (e.g. 10)

    # ── Checkpointing ──
    "save_steps": 50,
    "save_total_limit": None,   # Max checkpoints to keep

    # ── Save formats ──
    "save_method": "merged_16bit",  # "merged_16bit", "merged_4bit", "lora", "mxfp4"
    "save_gguf": None,          # GGUF quant: "q4_k_m", "q8_0", "f16", or list; XL: "Q2_K_XL", "Q3_K_XL", "Q4_K_XL"
    # save_torchao is defined in QAT section above
    "maximum_memory_usage": 0.75,  # Memory cap during export

    # ── Long context (500k+) ──
    "unsloth_tiled_mlp": False,  # Tiled MLP: 60% less VRAM, enables 500k+ context on 80GB GPUs

    # ── Multi-GPU ──
    "device_map": None,         # "balanced", "auto", or None (single GPU)

    # ── Logging ──
    "wandb_project": "swe-gym",
    "no_wandb": False,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO RL training — all Unsloth features configurable via YAML recipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Recipe & model
    p.add_argument("--recipe", default=None, help="YAML recipe config (e.g. configs/unsloth/grpo_qwen3_8b.yaml)")
    p.add_argument("--model", default=None, help="HuggingFace model ID or local path (overrides recipe)")
    p.add_argument("--from-sft", default=None, help="Start from an SFT checkpoint instead of base model")
    p.add_argument("--output-dir", default="./grpo_output", help="Output directory")
    p.add_argument("--task-dir", default=None, help="Directory with validated task instances (default: auto-detect)")
    p.add_argument("--token", default=None, help="HF token for gated models (Llama, Gemma); or set HF_TOKEN env var")
    p.add_argument("--trust-remote-code", action="store_true", help="Trust remote code (needed for some architectures)")
    p.add_argument("--revision", default=None, help="Model revision to download (commit hash, branch, tag)")

    # Quantization
    p.add_argument("--four-bit", action="store_true", default=None, help="4-bit quantization")
    p.add_argument("--no-four-bit", action="store_true", help="Disable 4-bit, use full precision")
    p.add_argument("--eight-bit", action="store_true", help="8-bit quantization")
    p.add_argument("--fp8", action="store_true", help="FP8 quantization (RTX 40/50, H100+; 60%% less VRAM)")

    # Multi-GPU
    p.add_argument("--device-map", default=None,
                   help="Device map for model splitting across GPUs (e.g. 'balanced', 'auto')")

    # Hyperparameter overrides
    p.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (overrides recipe)")
    p.add_argument("--lr", type=float, default=None, help="Learning rate (overrides recipe)")
    p.add_argument("--steps", type=int, default=None, help="Training steps (overrides recipe)")
    p.add_argument("--batch-size", type=int, default=None, help="Number of problems per batch (overrides recipe)")
    p.add_argument("--num-generations", type=int, default=None, help="Completions per problem (overrides recipe)")
    p.add_argument("--max-seq-len", type=int, default=None, help="Maximum sequence length (overrides recipe)")
    p.add_argument("--max-completion-len", type=int, default=None, help="Max tokens for completion (overrides recipe)")

    # Reward
    p.add_argument("--reward-mode", choices=["test", "format", "hybrid"], default=None,
                   help="Reward function (overrides recipe)")

    # GRPO loss variants
    p.add_argument("--loss-type", choices=["grpo", "bnpo", "dr_grpo", "dapo", "gspo"], default=None,
                   help="GRPO loss variant (overrides recipe)")

    # Training features
    p.add_argument("--use-rslora", action="store_true", help="Rank-Stabilized LoRA")
    p.add_argument("--offload-embedding", action="store_true", help="Offload embeddings (save ~1GB)")
    p.add_argument("--vllm-standby", action="store_true", help="vLLM standby mode (~9GB savings)")
    p.add_argument("--float8-kv-cache", action="store_true", help="2x KV cache reduction")
    p.add_argument("--tiled-mlp", action="store_true", help="Tiled MLP for 500k+ context (60%% less VRAM)")

    # QAT (Quantization-Aware Training)
    p.add_argument("--qat-scheme", default=None,
                   choices=["int4", "int8-int4", "fp8-int4", "fp8-fp8"],
                   help="QAT scheme — recovers ~70%% accuracy lost in quantization")

    # Save options
    p.add_argument("--save-gguf", default=None, help="Also save GGUF quantization (e.g. q4_k_m, Q4_K_XL)")
    p.add_argument("--save-lora-only", action="store_true", help="Save only LoRA adapter, skip merged model")
    p.add_argument("--save-mxfp4", action="store_true", help="Save as MXFP4 (75%% less disk)")
    p.add_argument("--save-torchao", action="store_true", help="Save via TorchAO (for QAT models)")
    p.add_argument("--push-to-hub", default=None, help="Push merged model to HuggingFace Hub")
    p.add_argument("--push-to-hub-gguf", default=None, help="Push GGUF to HuggingFace Hub (e.g. user/model-gguf)")

    # Logging
    p.add_argument("--wandb-project", default=None, help="WandB project name")
    p.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    p.add_argument("--resume", default=None, nargs="?", const=True,
                   help="Resume from checkpoint (True=latest, or path to checkpoint dir)")
    return p.parse_args()


def load_recipe(recipe_path: str | None) -> dict:
    """Load a YAML recipe config, merged onto full defaults."""
    recipe = dict(RECIPE_DEFAULTS)
    if recipe_path and Path(recipe_path).exists():
        with open(recipe_path) as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            recipe.update(yaml_data)
    return recipe


def apply_overrides(recipe: dict, args: argparse.Namespace) -> dict:
    """CLI args override recipe values when explicitly provided."""
    overrides = {
        "model": args.model or args.from_sft,
        "four_bit": True if args.four_bit else (False if args.no_four_bit else None),
        "eight_bit": True if args.eight_bit else None,
        "fp8": True if args.fp8 else None,
        "lora_rank": args.lora_rank,
        "lr": args.lr,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_seq_len": args.max_seq_len,
        "max_completion_len": args.max_completion_len,
        "reward_mode": args.reward_mode,
        "loss_type": args.loss_type,
        "device_map": args.device_map,
        "wandb_project": args.wandb_project,
    }
    for key, val in overrides.items():
        if val is not None:
            recipe[key] = val
    if args.lora_rank is not None:
        recipe["lora_alpha"] = args.lora_rank
    if args.use_rslora:
        recipe["use_rslora"] = True
    if args.offload_embedding:
        recipe["offload_embedding"] = True
    if args.vllm_standby:
        recipe["vllm_standby"] = True
    if args.float8_kv_cache:
        recipe["float8_kv_cache"] = True
    if args.tiled_mlp:
        recipe["unsloth_tiled_mlp"] = True
    if args.no_wandb:
        recipe["no_wandb"] = True
    if args.save_gguf:
        recipe["save_gguf"] = args.save_gguf
    if args.save_mxfp4:
        recipe["save_method"] = "mxfp4"
    if args.save_torchao:
        recipe["save_torchao"] = True
    if args.qat_scheme:
        recipe["qat_scheme"] = args.qat_scheme
        recipe["load_in_16bit"] = True  # QAT requires 16-bit loading
    if args.token:
        recipe["token"] = args.token
    if args.trust_remote_code:
        recipe["trust_remote_code"] = True
    if args.revision:
        recipe["revision"] = args.revision
    return recipe


def load_repo_conf() -> dict[str, str]:
    """Read repo.conf key=value pairs."""
    conf = {}
    conf_path = Path(__file__).resolve().parent.parent / "repo.conf"
    if conf_path.exists():
        for line in conf_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                for ref_key, ref_val in conf.items():
                    val = val.replace(f"${{{ref_key}}}", ref_val)
                conf[key.strip()] = val
    return conf


def load_tasks(task_dir: str) -> list[dict]:
    """Load validated task instances from JSONL/JSON files."""
    task_path = Path(task_dir)
    tasks = []
    for f in sorted(task_path.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    for f in sorted(task_path.glob("validated_*.json")):
        data = json.loads(f.read_text())
        if isinstance(data, list):
            tasks.extend(data)
    return tasks


def make_prompt(task: dict) -> str:
    """Convert a gym task into a prompt for the model."""
    problem_stmt = task.get("problem_statement", "")
    repo = task.get("repo", "")
    hints = task.get("hints_text", "")

    prompt = f"""You are a software engineer fixing a bug in the repository `{repo}`.

## Problem Description
{problem_stmt}
"""
    if hints:
        prompt += f"""
## Hints
{hints}
"""

    prompt += """
## Instructions
Analyze the bug and provide a fix as a unified diff patch. Your response must include
a code block with the patch in unified diff format that can be applied with `git apply`.

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ ... @@
 context line
-old line
+new line
 context line
```
"""
    return prompt


def extract_patch(completion: str) -> str | None:
    """Extract a unified diff patch from model output."""
    diff_blocks = re.findall(r"```diff\n(.*?)```", completion, re.DOTALL)
    if diff_blocks:
        return diff_blocks[0].strip()
    lines = completion.split("\n")
    patch_lines = []
    in_patch = False
    for line in lines:
        if line.startswith("--- a/") or line.startswith("diff --git"):
            in_patch = True
        if in_patch:
            patch_lines.append(line)
    if patch_lines:
        return "\n".join(patch_lines)
    return None


# ── Reward functions ─────────────────────────────────────────────

def _check_stdlib_only(patch: str) -> bool:
    """Check if a patch only imports from Python stdlib."""
    import_lines = re.findall(r"^\+\s*(?:import|from)\s+(\w+)", patch, re.MULTILINE)
    stdlib_modules = {
        "os", "sys", "re", "json", "pathlib", "collections", "itertools",
        "functools", "typing", "dataclasses", "abc", "io", "math", "copy",
        "datetime", "time", "hashlib", "base64", "urllib", "http", "logging",
        "unittest", "textwrap", "string", "enum", "contextlib", "operator",
        "shutil", "tempfile", "glob", "fnmatch", "struct", "csv", "configparser",
        "argparse", "subprocess", "threading", "multiprocessing", "socket",
        "warnings", "traceback", "inspect", "ast", "dis", "pdb", "profile",
    }
    for mod in import_lines:
        if mod not in stdlib_modules:
            return False
    return True


def _strip_globals(patch: str) -> str:
    """Strip global-scope mutations from a patch (reward hacking countermeasure)."""
    filtered = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            stripped = line[1:].strip()
            # Skip lines that are global assignments or exec/eval
            if re.match(r"^[A-Z_][A-Z_0-9]*\s*=", stripped):
                continue
            if stripped.startswith(("exec(", "eval(", "__import__")):
                continue
        filtered.append(line)
    return "\n".join(filtered)


def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Fast reward: check if the completion contains a valid-looking patch."""
    rewards = []
    for completion in completions:
        patch = extract_patch(completion)
        if patch is None:
            rewards.append(0.0)
            continue
        has_minus = any(ln.startswith("-") for ln in patch.split("\n"))
        has_plus = any(ln.startswith("+") for ln in patch.split("\n"))
        has_header = "---" in patch and "+++" in patch
        if has_minus and has_plus and has_header:
            rewards.append(0.5)
        else:
            rewards.append(0.1)
    return rewards


def test_reward_fn(completions: list[str], tasks: list[dict] | None = None,
                   recipe: dict | None = None, **kwargs) -> list[float]:
    """Full reward: run the patch against the test suite in Docker.

    Supports reward hacking countermeasures from recipe:
    - reward_stdlib_only: Penalize non-stdlib imports
    - reward_strip_globals: Strip global-scope mutations
    - reward_timeout: Per-evaluation timeout
    """
    timeout = (recipe or {}).get("reward_timeout") or 180
    stdlib_only = (recipe or {}).get("reward_stdlib_only", False)
    strip_globals = (recipe or {}).get("reward_strip_globals", False)

    rewards = []
    for i, completion in enumerate(completions):
        patch = extract_patch(completion)
        if patch is None:
            rewards.append(0.0)
            continue

        # Reward hacking countermeasures
        if stdlib_only and not _check_stdlib_only(patch):
            rewards.append(-20.0)
            continue

        if strip_globals:
            patch = _strip_globals(patch)

        task = tasks[i % len(tasks)] if tasks else None
        if task is None:
            rewards.append(format_reward_fn([completion])[0])
            continue

        try:
            instance_id = task.get("instance_id", "unknown")
            result = subprocess.run(
                [
                    "python", "-m", "swebench.harness.run_evaluation",
                    "--predictions_path", "/dev/stdin",
                    "--swe_bench_tasks", task.get("task_file", ""),
                    "--log_dir", "/tmp/grpo_eval",
                    "--testbed", "/tmp/grpo_testbed",
                    "--timeout", str(timeout),
                ],
                input=json.dumps({"instance_id": instance_id, "model_patch": patch}),
                capture_output=True,
                text=True,
                timeout=timeout + 60,
            )
            if "PASS" in result.stdout:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except (subprocess.TimeoutExpired, Exception):
            rewards.append(format_reward_fn([completion])[0])

    return rewards


# ── Model loading (shared helpers) ──────────────────────────────

def _load_model(recipe: dict):
    """Load model with all recipe-configured options."""
    from unsloth import FastLanguageModel

    model_name = recipe["model"]
    lora_rank = recipe["lora_rank"]

    load_kwargs = dict(
        model_name=model_name,
        max_seq_length=recipe["max_seq_len"],
        load_in_4bit=recipe["four_bit"],
        load_in_8bit=recipe.get("eight_bit", False),
        fast_inference=True,  # Enables vLLM-backed generation for GRPO
        max_lora_rank=lora_rank,
        gpu_memory_utilization=recipe["gpu_memory_utilization"],
        dtype=recipe.get("dtype"),
        random_state=recipe.get("seed", 42),
    )
    # Auth token for gated models (Llama, Gemma, etc.)
    if recipe.get("token"):
        load_kwargs["token"] = recipe["token"]
    if recipe.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True
    if recipe.get("revision"):
        load_kwargs["revision"] = recipe["revision"]
    if recipe.get("resize_model_vocab"):
        load_kwargs["resize_model_vocab"] = recipe["resize_model_vocab"]
    # FP8 quantization (RTX 40/50, H100+; 60% less VRAM, 1.4x faster inference)
    if recipe.get("fp8"):
        load_kwargs["load_in_4bit"] = False
        load_kwargs["load_in_8bit"] = False
        load_kwargs["load_in_fp8"] = True
    # QAT requires 16-bit loading
    elif recipe.get("load_in_16bit") or recipe.get("qat_scheme"):
        load_kwargs["load_in_4bit"] = False
        load_kwargs["load_in_8bit"] = False
        load_kwargs["load_in_16bit"] = True
    if recipe.get("device_map"):
        load_kwargs["device_map"] = recipe["device_map"]
    if recipe.get("offload_embedding"):
        load_kwargs["offload_embedding"] = True
    if recipe.get("float8_kv_cache"):
        load_kwargs["float8_kv_cache"] = True
    if recipe.get("unsloth_tiled_mlp"):
        load_kwargs["unsloth_tiled_mlp"] = True

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    return model, tokenizer


def _apply_lora(model, recipe: dict):
    """Apply LoRA with all recipe-configured options."""
    from unsloth import FastLanguageModel

    peft_kwargs = dict(
        r=recipe["lora_rank"],
        lora_alpha=recipe["lora_alpha"],
        target_modules=recipe["lora_targets"],
        lora_dropout=recipe.get("lora_dropout", 0),
        bias=recipe.get("bias", "none"),
        use_gradient_checkpointing=recipe.get("gradient_checkpointing", "unsloth"),
        use_rslora=recipe.get("use_rslora", False),
        random_state=recipe.get("seed", 42),
    )
    if recipe.get("loftq_config"):
        peft_kwargs["loftq_config"] = recipe["loftq_config"]
    if recipe.get("init_lora_weights") is not True and recipe.get("init_lora_weights") is not None:
        peft_kwargs["init_lora_weights"] = recipe["init_lora_weights"]
    if recipe.get("layers_to_transform"):
        peft_kwargs["layers_to_transform"] = recipe["layers_to_transform"]
    if recipe.get("modules_to_save"):
        peft_kwargs["modules_to_save"] = recipe["modules_to_save"]
    if recipe.get("qat_scheme"):
        peft_kwargs["qat_scheme"] = recipe["qat_scheme"]

    return FastLanguageModel.get_peft_model(model, **peft_kwargs)


def _save_model(model, tokenizer, out: Path, recipe: dict, args: argparse.Namespace):
    """Save model with all recipe-configured formats."""
    print(f"\n=== Saving model to {out} ===")

    # Always save LoRA adapter
    model.save_pretrained(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))

    if not args.save_lora_only:
        save_method = recipe.get("save_method", "merged_16bit")
        print(f"Saving merged model ({save_method}) for inference...")
        save_kwargs = dict(save_method=save_method)
        if recipe.get("maximum_memory_usage"):
            save_kwargs["maximum_memory_usage"] = recipe["maximum_memory_usage"]
        model.save_pretrained_merged(str(out / "merged"), tokenizer, **save_kwargs)

    # TorchAO export (for QAT models — Int4/Int8/FP8 inference)
    if recipe.get("save_torchao") or recipe.get("qat_scheme"):
        torchao_dir = out / "torchao"
        torchao_dir.mkdir(exist_ok=True)
        print("Converting QAT fake-quantize ops to real quantize ops...")
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        quantize_(model, QATConfig(step="convert"))
        print(f"Saving TorchAO model to {torchao_dir}...")
        torchao_config = getattr(model, "_torchao_config", None)
        base_config = torchao_config.base_config if torchao_config else None
        model.save_pretrained_torchao(
            str(torchao_dir), tokenizer,
            torchao_config=base_config,
        )
        print(f"  TorchAO: {torchao_dir}/")

    # GGUF export
    gguf = recipe.get("save_gguf") or getattr(args, "save_gguf", None)
    quant_methods = (gguf if isinstance(gguf, list) else [gguf]) if gguf else []
    if quant_methods:
        gguf_dir = out / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        print(f"Saving GGUF ({gguf})...")
        for method in quant_methods:
            model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=method)
        print(f"  GGUF: {gguf_dir}/")

    # Push to Hub (merged)
    hub_repo = getattr(args, "push_to_hub", None)
    token = recipe.get("token")
    if hub_repo:
        save_method = recipe.get("save_method", "merged_16bit")
        print(f"Pushing to HuggingFace Hub: {hub_repo}")
        hub_kwargs = dict(save_method=save_method)
        if token:
            hub_kwargs["token"] = token
        model.push_to_hub_merged(hub_repo, tokenizer, **hub_kwargs)

    # Push GGUF to Hub
    hub_gguf_repo = getattr(args, "push_to_hub_gguf", None)
    if hub_gguf_repo and quant_methods:
        print(f"Pushing GGUF to HuggingFace Hub: {hub_gguf_repo}")
        gguf_kwargs = dict(quantization_method=quant_methods[0])
        if token:
            gguf_kwargs["token"] = token
        model.push_to_hub_gguf(hub_gguf_repo, tokenizer, **gguf_kwargs)


def main():
    args = parse_args()
    conf = load_repo_conf()

    # Load recipe (YAML config) and apply CLI overrides
    recipe = load_recipe(args.recipe)
    recipe = apply_overrides(recipe, args)

    # vLLM standby mode — set env var before imports if enabled
    if recipe.get("vllm_standby"):
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
        print("vLLM standby mode enabled (shares weight memory, ~9GB savings)")

    # MoE backend selection
    if recipe.get("moe_backend"):
        os.environ["UNSLOTH_MOE_BACKEND"] = recipe["moe_backend"]
        print(f"MoE backend: {recipe['moe_backend']}")

    # repo.conf model as fallback
    if not args.model and not args.from_sft and not args.recipe:
        recipe["model"] = conf.get("SFT_BASE_MODEL", recipe["model"])

    model_name = recipe["model"]
    repo_key = conf.get("REPO_KEY", "unknown__repo")

    # Find task instances
    if args.task_dir:
        task_dir = args.task_dir
    else:
        candidates = [
            f"./logs/validated_tasks/{repo_key}",
            f"./logs/tasks/{repo_key}",
            "./logs/validated_tasks",
        ]
        task_dir = next((d for d in candidates if Path(d).exists()), candidates[0])

    print("=== Step 9: GRPO RL Training ===")
    print(f"  Model:        {model_name}")
    print(f"  Tasks:        {task_dir}")
    print(f"  Reward mode:  {recipe['reward_mode']}")
    print(f"  Loss type:    {recipe['loss_type']}")
    print(f"  4-bit:        {recipe['four_bit']}")
    print(f"  Steps:        {recipe['steps']}")
    print(f"  Generations:  {recipe['num_generations']} per problem")
    print(f"  IS level:     {recipe.get('importance_sampling_level', 'token')}")
    print(f"  vLLM standby: {recipe.get('vllm_standby', False)}")
    print(f"  float8 KV:    {recipe.get('float8_kv_cache', False)}")
    if recipe.get("epsilon_high"):
        print(f"  Clipping:     epsilon={recipe['epsilon']}, epsilon_high={recipe['epsilon_high']}")
    if recipe.get("delta"):
        print(f"  Delta clip:   {recipe['delta']}")
    if recipe.get("qat_scheme"):
        print(f"  QAT:          {recipe['qat_scheme']}")
    if recipe.get("unsloth_tiled_mlp"):
        print("  Tiled MLP:    enabled (long context)")
    if recipe.get("unsloth_grpo_mini_batch") or recipe.get("unsloth_logit_chunk_multiplier"):
        print(f"  GRPO chunks:  mini_batch={recipe.get('unsloth_grpo_mini_batch', 'auto')}, logit_mult={recipe.get('unsloth_logit_chunk_multiplier', 'auto')}")
    print()

    # Load tasks for reward computation
    tasks = load_tasks(task_dir)
    if not tasks:
        print(f"ERROR: No task instances found in {task_dir}")
        print("Run steps 2-3 first (make generate-tasks && make validate-tasks)")
        sys.exit(1)
    print(f"Loaded {len(tasks)} task instances")

    # Build dataset of prompts from tasks
    from datasets import Dataset

    prompts = [{"prompt": make_prompt(task), "task_idx": i} for i, task in enumerate(tasks)]
    dataset = Dataset.from_list(prompts)

    # Load model with unsloth
    from unsloth import is_bfloat16_supported

    use_bf16 = is_bfloat16_supported()

    print(f"\n=== Loading model: {model_name} ===")
    if recipe.get("device_map"):
        print(f"  Device map: {recipe['device_map']} (model splitting across GPUs)")

    model, tokenizer = _load_model(recipe)
    model = _apply_lora(model, recipe)

    # Set up reward function based on mode
    reward_mode = recipe["reward_mode"]
    if reward_mode == "test":
        def reward_fn_test(completions, **kw):
            return test_reward_fn(completions, tasks=tasks, recipe=recipe, **kw)
        reward_fn = reward_fn_test
    elif reward_mode == "format":
        reward_fn = format_reward_fn
    else:  # hybrid
        fmt_w = recipe.get("reward_format_weight", 0.3)
        test_w = recipe.get("reward_test_weight", 0.7)

        def hybrid_reward(completions, **kw):
            format_rewards = format_reward_fn(completions, **kw)
            test_rewards = test_reward_fn(completions, tasks=tasks, recipe=recipe, **kw)
            return [fmt_w * f + test_w * t for f, t in zip(format_rewards, test_rewards)]
        reward_fn = hybrid_reward

    # GRPO training config with vLLM sampling params
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import vLLMSamplingParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build vLLM sampling params from recipe
    sampling_kwargs = dict(min_p=recipe.get("min_p", 0.1), seed=recipe.get("seed", 42))
    if recipe.get("top_p"):
        sampling_kwargs["top_p"] = recipe["top_p"]
    if recipe.get("top_k"):
        sampling_kwargs["top_k"] = recipe["top_k"]
    if recipe.get("repetition_penalty"):
        sampling_kwargs["repetition_penalty"] = recipe["repetition_penalty"]
    vllm_sampling_params = vLLMSamplingParams(**sampling_kwargs)

    # Build GRPOConfig with all recipe params
    grpo_kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=recipe["lr"],
        per_device_train_batch_size=recipe["batch_size"],
        num_generations=recipe["num_generations"],
        max_completion_length=recipe["max_completion_len"],
        max_steps=recipe["steps"],
        logging_steps=1,
        save_steps=recipe.get("save_steps", 50),
        warmup_steps=recipe["warmup_steps"],
        weight_decay=recipe["weight_decay"],
        lr_scheduler_type=recipe["lr_scheduler"],
        optim=recipe.get("optim", "adamw_8bit"),
        seed=recipe.get("seed", 42),
        fp16=not use_bf16,
        bf16=use_bf16,
        report_to="none" if recipe.get("no_wandb") else "wandb",
        run_name=f"swe-gym-grpo-{Path(model_name).name}",
        # GRPO-specific
        temperature=recipe["temperature"],
        loss_type=recipe["loss_type"],
        vllm_sampling_params=vllm_sampling_params,
        mask_truncated_completions=recipe.get("mask_truncated_completions", True),
        # Advanced GRPO
        beta=recipe.get("beta", 0.0),
        num_iterations=recipe.get("num_iterations", 1),
        importance_sampling_level=recipe.get("importance_sampling_level", "token"),
        scale_rewards=recipe.get("scale_rewards", "group"),
    )

    # Clipping params (epsilon, epsilon_high, delta)
    if recipe.get("epsilon") is not None:
        grpo_kwargs["epsilon"] = recipe["epsilon"]
    if recipe.get("epsilon_high") is not None:
        grpo_kwargs["epsilon_high"] = recipe["epsilon_high"]
    if recipe.get("delta") is not None:
        grpo_kwargs["delta"] = recipe["delta"]

    # Off-policy corrections
    if recipe.get("vllm_importance_sampling_correction") is not None:
        grpo_kwargs["vllm_importance_sampling_correction"] = recipe["vllm_importance_sampling_correction"]
    if recipe.get("vllm_importance_sampling_cap") is not None:
        grpo_kwargs["vllm_importance_sampling_cap"] = recipe["vllm_importance_sampling_cap"]

    # Reward weights (per-reward-function)
    if recipe.get("reward_weights"):
        grpo_kwargs["reward_weights"] = recipe["reward_weights"]

    # Steps per generation
    if recipe.get("steps_per_generation"):
        grpo_kwargs["steps_per_generation"] = recipe["steps_per_generation"]

    # Warmup ratio alternative
    if recipe.get("warmup_ratio"):
        grpo_kwargs["warmup_ratio"] = recipe["warmup_ratio"]

    # Save total limit
    if recipe.get("save_total_limit"):
        grpo_kwargs["save_total_limit"] = recipe["save_total_limit"]

    # GRPO long-context memory optimization (7x longer context)
    if recipe.get("unsloth_grpo_mini_batch") is not None:
        grpo_kwargs["unsloth_grpo_mini_batch"] = recipe["unsloth_grpo_mini_batch"]
    if recipe.get("unsloth_logit_chunk_multiplier") is not None:
        grpo_kwargs["unsloth_logit_chunk_multiplier"] = recipe["unsloth_logit_chunk_multiplier"]

    training_args = GRPOConfig(**grpo_kwargs)

    # Save config for reproducibility
    config_log = {k: v for k, v in recipe.items() if not callable(v)}
    (output_dir / "grpo_config.json").write_text(json.dumps(config_log, indent=2, default=str))

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=training_args,
    )

    print("\n=== Starting GRPO training ===")
    print(f"  The model will generate {recipe['num_generations']} patches per problem,")
    print(f"  get rewarded based on {reward_mode} results, and learn from the best ones.")
    print("  Expect 300+ steps before rewards start increasing.")
    print()

    trainer.train(resume_from_checkpoint=args.resume)

    # Save
    _save_model(model, tokenizer, output_dir, recipe, args)

    print("\n=== GRPO training complete ===")
    print(f"  LoRA adapter: {output_dir}/final/")
    if not args.save_lora_only:
        print(f"  Merged model: {output_dir}/merged/")
    print("\nTo serve:")
    print(f"  vllm serve {output_dir}/merged/")


if __name__ == "__main__":
    main()
