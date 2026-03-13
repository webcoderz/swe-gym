#!/usr/bin/env python3
"""Step 8 (alt): Fine-tune a model using Unsloth.

Local training — runs on whatever GPU(s) are available.
Supports LoRA SFT (default), full SFT, DPO, and continued pretraining.
ALL Unsloth capabilities are configurable via YAML recipe or CLI flags.

Single GPU:
    python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml
    python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

Multi-GPU (DDP — one model copy per GPU, distinct samples, aggregated gradients):
    torchrun --nproc_per_node=2 scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml
    accelerate launch scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml

Multi-GPU (model splitting — for models too large for one GPU, e.g. 70B+):
    python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_coder_next_lora.yaml --device-map balanced

Full fine-tune:
    python scripts/08_train_sft_unsloth.py --full

DPO (after SFT):
    python scripts/08_train_sft_unsloth.py --dpo --sft-checkpoint ./sft_output/

Continued pretraining:
    python scripts/08_train_sft_unsloth.py --cpt --data-dir ./corpus/

GGUF export:
    python scripts/08_train_sft_unsloth.py --recipe ... --save-gguf q4_k_m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


# ── Full YAML recipe schema (all Unsloth capabilities) ───────────
# Every field below can be set in a recipe YAML or overridden via CLI.
RECIPE_DEFAULTS = {
    # ── Model ──
    "model": "Qwen/Qwen3-8B",
    "mode": "lora",           # lora | full | cpt
    "dtype": None,            # None (auto), "bfloat16", "float16"
    "token": None,            # HF token for gated models (Llama, Gemma); or set HF_TOKEN env var
    "trust_remote_code": False,  # Required for some newer model architectures
    "revision": None,         # Pin specific model revision from Hub (e.g. "main", commit hash)
    "resize_model_vocab": None,  # Resize vocab (int) — for adding custom special tokens

    # ── Quantization ──
    "four_bit": False,        # QLoRA 4-bit (minimal VRAM)
    "eight_bit": False,       # 8-bit quantization
    "fp8": False,             # FP8 quantization (RTX 40/50, H100+; 60% less VRAM, 1.4x faster)
    "load_in_16bit": False,   # Force 16-bit loading (required for QAT)
    "offload_embedding": False,  # Saves ~1GB VRAM for large models (e.g. GPT-OSS)

    # ── QAT (Quantization-Aware Training) ──
    # Recovers ~70% of accuracy lost during quantization via TorchAO.
    # Schemes: "int4", "int8-int4", "fp8-int4", "fp8-fp8"
    "qat_scheme": None,       # None = no QAT; set to enable (e.g. "int4")
    "save_torchao": False,    # Save QAT model via save_pretrained_torchao()

    # ── LoRA config ──
    "lora_rank": 64,          # r: 8, 16, 32, 64, 128, 256
    "lora_alpha": 64,         # Usually = rank, or rank*2 for aggressive learning
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0,        # 0 recommended; Unsloth optimizes this
    "bias": "none",           # "none" (faster) or "all"
    "use_rslora": False,      # Rank-Stabilized LoRA: scales alpha by 1/sqrt(r)
    "loftq_config": None,     # LoftQ initialization config
    "init_lora_weights": True,  # True (default), False, or "gaussian"
    "layers_to_transform": None,  # List of layer indices to apply LoRA to (e.g. [24,25,26,27] for last 4)
    "modules_to_save": None,  # Extra modules to save (e.g. ["lm_head", "embed_tokens"] for CPT)

    # ── Gradient checkpointing ──
    "gradient_checkpointing": "unsloth",  # "unsloth" (30% extra savings), True, False

    # ── Training hyperparams ──
    "lr": 2e-4,
    "epochs": 2,
    "max_steps": None,        # If set, overrides epochs
    "batch_size": 1,
    "grad_accum": 8,
    "max_seq_len": 10240,
    "optim": "adamw_8bit",    # "adamw_8bit", "adamw_torch", "sgd", etc.
    "warmup_steps": 10,
    "warmup_ratio": None,     # Alternative to warmup_steps (e.g. 0.1 = 10% of steps)
    "weight_decay": 0.01,
    "lr_scheduler": "cosine", # "cosine", "linear", "constant", etc.
    "seed": 42,

    # ── Data ──
    "packing": False,         # Pack multiple examples per sequence
    "train_on_completions": False,  # Only compute loss on assistant responses (~1% boost)
    "instruction_part": "<|im_start|>user\n",    # For train_on_completions
    "response_part": "<|im_start|>assistant\n",  # For train_on_completions
    "reasoning_effort": None,  # GPT-OSS: "low", "medium", "high"

    # ── Eval & early stopping ──
    "eval_split": 0.0,        # Fraction held out for eval (0 = no eval)
    "eval_steps": 10,         # Eval every N steps
    "early_stopping_patience": None,  # Stop after N evals without improvement

    # ── Checkpointing ──
    "save_strategy": "epoch",  # "epoch", "steps"
    "save_steps": 500,        # If save_strategy="steps"
    "save_total_limit": None,  # Max checkpoints to keep

    # ── DPO-specific ──
    "dpo_lr": 2e-5,
    "dpo_beta": 0.05,
    "dpo_label_smoothing": 0.0,

    # ── CPT-specific (continued pretraining) ──
    "cpt_lr": 5e-5,
    "cpt_embed_lr": None,     # Auto: cpt_lr / 5

    # ── Save formats ──
    "save_method": "merged_16bit",  # "merged_16bit", "merged_4bit", "lora", "mxfp4"
    "save_gguf": None,        # GGUF quant: "q4_k_m", "q8_0", "f16", or list; XL: "Q2_K_XL", "Q3_K_XL", "Q4_K_XL"
    # save_torchao is defined in QAT section above
    "maximum_memory_usage": 0.75,  # Memory cap during export (0.5 if OOM)

    # ── Long context (500k+) ──
    "unsloth_tiled_mlp": False,  # Tiled MLP: 60% less VRAM, enables 500k+ context on 80GB GPUs

    # ── Multi-GPU ──
    "device_map": None,       # "balanced", "auto", or None (single GPU)

    # ── Logging ──
    "wandb_project": "swe-gym",
    "no_wandb": False,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unsloth SFT/DPO/CPT training — all features configurable via YAML recipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Recipe & model
    p.add_argument("--recipe", default=None, help="YAML recipe config (e.g. configs/unsloth/qwen3_8b_lora.yaml)")
    p.add_argument("--model", default=None, help="HuggingFace model ID (overrides recipe)")
    p.add_argument("--data-dir", default=None, help="Directory with ft_*.jsonl files (default: auto-detect)")
    p.add_argument("--output-dir", default="./sft_output", help="Output directory")
    p.add_argument("--token", default=None, help="HF token for gated models (Llama, Gemma); or set HF_TOKEN env var")
    p.add_argument("--trust-remote-code", action="store_true", help="Trust remote code (needed for some architectures)")
    p.add_argument("--revision", default=None, help="Model revision to download (commit hash, branch, tag)")

    # Training mode
    p.add_argument("--full", action="store_true", help="Full fine-tune instead of LoRA")
    p.add_argument("--dpo", action="store_true", help="DPO training (requires --sft-checkpoint)")
    p.add_argument("--cpt", action="store_true", help="Continued pretraining (domain adaptation)")
    p.add_argument("--sft-checkpoint", default=None, help="Path to SFT checkpoint for DPO")

    # Quantization
    p.add_argument("--four-bit", action="store_true", help="QLoRA 4-bit quantization")
    p.add_argument("--eight-bit", action="store_true", help="8-bit quantization")
    p.add_argument("--fp8", action="store_true", help="FP8 quantization (RTX 40/50, H100+; 60%% less VRAM)")

    # Multi-GPU
    p.add_argument("--device-map", default=None, help="Model splitting: 'balanced', 'auto'")

    # Hyperparameter overrides (all also settable in YAML)
    p.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--epochs", type=int, default=None, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    p.add_argument("--max-seq-len", type=int, default=None, help="Maximum sequence length")
    p.add_argument("--max-steps", type=int, default=None, help="Max steps (overrides epochs)")

    # Save options
    p.add_argument("--save-gguf", default=None, help="GGUF quantization (q4_k_m, q8_0, f16, Q4_K_XL)")
    p.add_argument("--save-lora-only", action="store_true", help="Save only LoRA adapter (~100MB)")
    p.add_argument("--save-mxfp4", action="store_true", help="Save as MXFP4 (75%% less disk)")
    p.add_argument("--save-torchao", action="store_true", help="Save via TorchAO (for QAT models)")
    p.add_argument("--push-to-hub", default=None, help="Push merged model to HuggingFace Hub")
    p.add_argument("--push-to-hub-gguf", default=None, help="Push GGUF to HuggingFace Hub (e.g. user/model-gguf)")

    # QAT (Quantization-Aware Training)
    p.add_argument("--qat-scheme", default=None,
                   choices=["int4", "int8-int4", "fp8-int4", "fp8-fp8"],
                   help="QAT scheme — recovers ~70%% accuracy lost in quantization")

    # Training features (all also settable in YAML)
    p.add_argument("--train-on-completions", action="store_true", help="Loss on assistant only (~1%% boost)")
    p.add_argument("--packing", action="store_true", help="Pack multiple examples per sequence")
    p.add_argument("--use-rslora", action="store_true", help="Rank-Stabilized LoRA")
    p.add_argument("--offload-embedding", action="store_true", help="Offload embeddings (save ~1GB)")
    p.add_argument("--tiled-mlp", action="store_true", help="Tiled MLP for 500k+ context (60%% less VRAM)")
    p.add_argument("--early-stopping", type=int, default=None, help="Patience for early stopping")
    p.add_argument("--eval-split", type=float, default=None, help="Eval holdout fraction")
    p.add_argument("--resume", default=None, nargs="?", const=True,
                   help="Resume from checkpoint (True=latest, or path to checkpoint dir)")

    # Logging
    p.add_argument("--wandb-project", default=None, help="WandB project name")
    p.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
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
        "model": args.model,
        "four_bit": True if args.four_bit else None,
        "eight_bit": True if args.eight_bit else None,
        "fp8": True if args.fp8 else None,
        "lora_rank": args.lora_rank,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "max_steps": args.max_steps,
        "device_map": args.device_map,
        "wandb_project": args.wandb_project,
    }
    for key, val in overrides.items():
        if val is not None:
            recipe[key] = val
    if args.lora_rank is not None:
        recipe["lora_alpha"] = args.lora_rank
    if args.full:
        recipe["mode"] = "full"
    if args.cpt:
        recipe["mode"] = "cpt"
    if args.packing:
        recipe["packing"] = True
    if args.use_rslora:
        recipe["use_rslora"] = True
    if args.offload_embedding:
        recipe["offload_embedding"] = True
    if args.tiled_mlp:
        recipe["unsloth_tiled_mlp"] = True
    if args.train_on_completions:
        recipe["train_on_completions"] = True
    if args.no_wandb:
        recipe["no_wandb"] = True
    if args.early_stopping is not None:
        recipe["early_stopping_patience"] = args.early_stopping
    if args.eval_split is not None:
        recipe["eval_split"] = args.eval_split
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


def load_sft_data(data_dir: str) -> list[dict]:
    """Load and combine all ft_*.jsonl files into a list of conversations."""
    data_path = Path(data_dir)
    examples = []
    for f in sorted(data_path.glob("ft_*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _load_model(recipe: dict):
    """Load model with all recipe-configured options."""
    from unsloth import FastLanguageModel

    model_name = recipe["model"]
    is_full = recipe["mode"] == "full"

    load_kwargs = dict(
        model_name=model_name,
        max_seq_length=recipe["max_seq_len"],
        load_in_4bit=recipe["four_bit"],
        load_in_8bit=recipe.get("eight_bit", False),
        full_finetuning=is_full,
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


def train_sft(recipe: dict, data_dir: str, output_dir: str, args: argparse.Namespace):
    """Run LoRA or full SFT with Unsloth."""
    from unsloth import is_bfloat16_supported
    from trl import SFTTrainer, SFTConfig

    model_name = recipe["model"]
    is_full = recipe["mode"] == "full"
    max_seq_len = recipe["max_seq_len"]
    use_bf16 = is_bfloat16_supported()

    print(f"\n=== Loading model: {model_name} ===")
    model, tokenizer = _load_model(recipe)

    if not is_full:
        print(f"Applying LoRA (rank={recipe['lora_rank']}, rslora={recipe.get('use_rslora', False)})...")
        model = _apply_lora(model, recipe)

    # Load data
    print(f"\n=== Loading training data from {data_dir} ===")
    examples = load_sft_data(data_dir)
    print(f"Training examples: {len(examples)}")

    if not examples:
        print("ERROR: No training data found. Run step 7 first.")
        sys.exit(1)

    from datasets import Dataset

    def format_example(example):
        messages = example.get("messages", [])
        kwargs = dict(tokenize=False, add_generation_prompt=False)
        if recipe.get("reasoning_effort"):
            kwargs["reasoning_effort"] = recipe["reasoning_effort"]
        text = tokenizer.apply_chat_template(messages, **kwargs)
        return {"text": text}

    dataset = Dataset.from_list(examples).map(format_example)

    # Optional train/eval split
    eval_split = recipe.get("eval_split", 0.0)
    if eval_split > 0:
        split = dataset.train_test_split(test_size=eval_split, seed=recipe["seed"])
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build training args from recipe
    patience = recipe.get("early_stopping_patience")
    train_kwargs = {}
    if recipe.get("max_steps"):
        train_kwargs["max_steps"] = recipe["max_steps"]
    else:
        train_kwargs["num_train_epochs"] = recipe["epochs"]

    sft_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=recipe["batch_size"],
        gradient_accumulation_steps=recipe["grad_accum"],
        learning_rate=recipe["lr"],
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=1,
        save_strategy=recipe.get("save_strategy", "steps" if patience else "epoch"),
        save_steps=recipe.get("save_steps", 50 if patience else 500),
        warmup_steps=recipe["warmup_steps"],
        weight_decay=recipe["weight_decay"],
        lr_scheduler_type=recipe["lr_scheduler"],
        optim=recipe.get("optim", "adamw_8bit"),
        seed=recipe["seed"],
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        packing=recipe["packing"],
        report_to="none" if recipe.get("no_wandb") else "wandb",
        run_name=f"swe-gym-sft-{'full' if is_full else 'lora'}-{Path(model_name).name}",
        **train_kwargs,
    )
    if recipe.get("warmup_ratio"):
        sft_kwargs["warmup_ratio"] = recipe["warmup_ratio"]
    if eval_dataset:
        sft_kwargs.update(
            eval_strategy="steps",
            eval_steps=recipe.get("eval_steps", 10),
            load_best_model_at_end=bool(patience),
            metric_for_best_model="eval_loss",
        )
    if patience:
        sft_kwargs["save_total_limit"] = recipe.get("save_total_limit", 3)

    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Train on completions only — ~1% accuracy boost per QLoRA paper
    if recipe.get("train_on_completions"):
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part=recipe.get("instruction_part", "<|im_start|>user\n"),
            response_part=recipe.get("response_part", "<|im_start|>assistant\n"),
        )
        print("  Training on completions only (assistant responses)")

    # Early stopping callback
    if patience and eval_dataset:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=0.0,
        ))
        print(f"  Early stopping: patience={patience}")

    mode_str = "full" if is_full else "LoRA"
    print(f"\n=== Starting {mode_str} SFT ===")
    print(f"  Model:      {model_name}")
    print(f"  Epochs:     {recipe['epochs']}")
    print(f"  Batch size: {recipe['batch_size']} x {recipe['grad_accum']} grad accum")
    print(f"  LR:         {recipe['lr']}")
    print(f"  4-bit:      {recipe['four_bit']}")
    print(f"  bf16:       {use_bf16}")
    print(f"  Optim:      {recipe.get('optim', 'adamw_8bit')}")
    print(f"  Output:     {out}")
    print()

    trainer.train(resume_from_checkpoint=args.resume)
    _save_model(model, tokenizer, out, recipe, args)

    print("\n=== SFT complete ===")
    print(f"  LoRA adapter: {out}/final/")
    if not args.save_lora_only:
        print(f"  Merged model: {out}/merged/")
    print("\nTo serve:")
    print(f"  vllm serve {out}/merged/")


def train_cpt(recipe: dict, data_dir: str, output_dir: str, args: argparse.Namespace):
    """Continued pretraining — domain adaptation before SFT.

    Uses higher rank LoRA with trainable lm_head + embed_tokens.
    Requires UnslothTrainer with separate embedding_learning_rate.
    """
    from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

    model_name = recipe["model"]
    max_seq_len = recipe["max_seq_len"]
    use_bf16 = is_bfloat16_supported()

    print(f"\n=== Loading model for CPT: {model_name} ===")
    model, tokenizer = _load_model(recipe)

    # CPT needs higher rank and trainable embeddings
    lora_rank = max(recipe["lora_rank"], 256)
    cpt_recipe = dict(recipe)
    cpt_recipe["lora_rank"] = lora_rank
    cpt_recipe["lora_alpha"] = lora_rank
    cpt_recipe["use_rslora"] = True  # Important for high rank
    cpt_recipe["modules_to_save"] = recipe.get("modules_to_save") or ["lm_head", "embed_tokens"]
    print(f"Applying CPT LoRA (rank={lora_rank}, RSLoRA, trainable embeddings)...")
    model = _apply_lora(model, cpt_recipe)

    # Load corpus data
    from datasets import Dataset
    data_path = Path(data_dir)
    if data_path.is_file() and data_path.suffix == ".txt":
        texts = data_path.read_text().split("\n\n")
        dataset = Dataset.from_dict({"text": [t.strip() for t in texts if t.strip()]})
    elif data_path.is_dir():
        texts = []
        for f in sorted(data_path.glob("*.txt")):
            texts.extend([t.strip() for t in f.read_text().split("\n\n") if t.strip()])
        for f in sorted(data_path.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    obj = json.loads(line)
                    texts.append(obj.get("text", ""))
        dataset = Dataset.from_dict({"text": texts})
    else:
        print(f"ERROR: No corpus data found at {data_dir}")
        sys.exit(1)

    print(f"CPT corpus: {len(dataset)} documents")

    out = Path(output_dir) / "cpt"
    out.mkdir(parents=True, exist_ok=True)

    lr = recipe.get("cpt_lr", 5e-5)
    embed_lr = recipe.get("cpt_embed_lr") or lr / 5

    training_args = UnslothTrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=recipe["batch_size"],
        gradient_accumulation_steps=recipe["grad_accum"],
        num_train_epochs=recipe["epochs"],
        learning_rate=lr,
        embedding_learning_rate=embed_lr,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=1,
        save_strategy="epoch",
        warmup_steps=recipe["warmup_steps"],
        weight_decay=recipe["weight_decay"],
        lr_scheduler_type=recipe["lr_scheduler"],
        optim=recipe.get("optim", "adamw_8bit"),
        seed=recipe["seed"],
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        report_to="none" if recipe.get("no_wandb") else "wandb",
        run_name=f"swe-gym-cpt-{Path(model_name).name}",
    )

    trainer = UnslothTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset, args=training_args,
    )

    print("\n=== Starting Continued Pretraining ===")
    print(f"  Model:    {model_name}")
    print(f"  LR:       {lr} (embed: {embed_lr})")
    print(f"  Rank:     {lora_rank} (RSLoRA)")
    print()

    trainer.train(resume_from_checkpoint=args.resume)
    _save_model(model, tokenizer, out, recipe, args)
    print("\n=== CPT complete. Now run SFT on this checkpoint. ===")
    print(f"  python scripts/08_train_sft_unsloth.py --model {out}/merged/ --recipe ...")


def train_dpo(recipe: dict, data_dir: str, output_dir: str, sft_checkpoint: str, args: argparse.Namespace):
    """DPO alignment after SFT."""
    from unsloth import PatchDPOTrainer, is_bfloat16_supported
    from trl import DPOTrainer, DPOConfig

    PatchDPOTrainer()
    use_bf16 = is_bfloat16_supported()

    sft_path = Path(sft_checkpoint)
    merged_path = sft_path / "merged"
    load_path = str(merged_path) if merged_path.exists() else str(sft_path)

    print(f"\n=== Loading SFT model from {load_path} ===")
    dpo_recipe = dict(recipe)
    dpo_recipe["model"] = load_path
    model, tokenizer = _load_model(dpo_recipe)
    model = _apply_lora(model, recipe)

    dpo_file = Path(data_dir) / "dpo_preferences.json"
    if not dpo_file.exists():
        print(f"ERROR: No DPO preference data at {dpo_file}")
        sys.exit(1)

    from datasets import Dataset
    with open(dpo_file) as f:
        dataset = Dataset.from_list(json.load(f))

    out = Path(output_dir) / "dpo"
    out.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=recipe["batch_size"],
        gradient_accumulation_steps=recipe["grad_accum"],
        num_train_epochs=recipe["epochs"],
        learning_rate=recipe.get("dpo_lr", 2e-5),
        beta=recipe.get("dpo_beta", 0.05),
        label_smoothing=recipe.get("dpo_label_smoothing", 0.0),
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=1,
        save_strategy="epoch",
        warmup_steps=5,
        weight_decay=recipe["weight_decay"],
        lr_scheduler_type=recipe["lr_scheduler"],
        optim=recipe.get("optim", "adamw_8bit"),
        seed=recipe["seed"],
        max_length=recipe["max_seq_len"],
        report_to="none" if recipe.get("no_wandb") else "wandb",
        run_name="swe-gym-dpo",
    )

    trainer = DPOTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=training_args)
    print("\n=== Starting DPO training ===")
    trainer.train(resume_from_checkpoint=args.resume)
    _save_model(model, tokenizer, out, recipe, args)
    print("\n=== DPO complete ===")


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

    recipe = load_recipe(args.recipe)
    recipe = apply_overrides(recipe, args)

    # repo.conf model as fallback
    if not args.model and not args.recipe:
        recipe["model"] = conf.get("SFT_BASE_MODEL", recipe["model"])

    repo_key = conf.get("REPO_KEY", "unknown__repo")
    data_dir = args.data_dir or f"./logs/sft_data/{repo_key}"
    output_dir = args.output_dir

    # Save resolved config for reproducibility
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in recipe.items() if v is not None}
    (out / "train_config.json").write_text(json.dumps(serializable, indent=2, default=str))

    print(f"Recipe: {args.recipe or '(defaults)'}")
    print(f"Model:  {recipe['model']}")
    print(f"Mode:   {recipe['mode']}")
    print()

    if args.dpo:
        if not args.sft_checkpoint:
            print("ERROR: DPO requires --sft-checkpoint.")
            sys.exit(1)
        train_dpo(recipe, data_dir, output_dir, args.sft_checkpoint, args)
    elif recipe["mode"] == "cpt" or args.cpt:
        train_cpt(recipe, data_dir, output_dir, args)
    else:
        train_sft(recipe, data_dir, output_dir, args)


if __name__ == "__main__":
    main()
