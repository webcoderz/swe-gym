#!/usr/bin/env python3
"""Step 8b: Hyperparameter optimization for LoRA fine-tuning using Optuna.

Runs short training trials with different LoRA hyperparameters and reports
the best configuration as a YAML recipe you can use for full training.

Searches over: rank, alpha, learning rate, weight decay, warmup, scheduler.

Usage:
    # Quick HPO (10 trials, 50 steps each) — ~30 min on single GPU
    python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml

    # Thorough search (30 trials, 100 steps each)
    python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --n-trials 30 --steps-per-trial 100

    # Custom search space via CLI
    python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --ranks 16,32,64 --lrs 1e-4,2e-4,5e-4

    # With 4-bit quantization
    python scripts/08b_hpo.py --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

Requires:
    pip install optuna  (or: uv pip install optuna)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA hyperparameter optimization via Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--recipe", required=True, help="Base YAML recipe to optimize")
    p.add_argument("--data-dir", default=None, help="Training data directory (default: auto-detect)")
    p.add_argument("--output-dir", default="./hpo_output", help="Output directory for trials + best recipe")

    # Search budget
    p.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials (default: 10)")
    p.add_argument("--steps-per-trial", type=int, default=50, help="Training steps per trial (default: 50)")
    p.add_argument("--eval-split", type=float, default=0.15, help="Fraction held out for eval (default: 0.15)")

    # Custom search space (comma-separated values)
    p.add_argument("--ranks", default=None, help="LoRA ranks to try (e.g. '16,32,64,128')")
    p.add_argument("--lrs", default=None, help="Learning rates to try (e.g. '1e-4,2e-4,5e-4')")
    p.add_argument("--alphas", default=None, help="Alpha strategies: 'equal' (=rank), 'double' (=2*rank), or values")

    # Quantization
    p.add_argument("--four-bit", action="store_true", help="Use 4-bit quantization")
    p.add_argument("--fp8", action="store_true", help="Use FP8 quantization")
    p.add_argument("--device-map", default=None, help="Device map for model splitting")

    # Misc
    p.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--pruning", action="store_true", help="Enable median pruning (stop bad trials early)")
    return p.parse_args()


def load_base_recipe(recipe_path: str) -> dict:
    """Load the base recipe YAML."""
    with open(recipe_path) as f:
        recipe = yaml.safe_load(f) or {}
    return recipe


def build_search_space(trial, args):
    """Define the Optuna search space."""
    # LoRA rank
    if args.ranks:
        rank_choices = [int(r) for r in args.ranks.split(",")]
        rank = trial.suggest_categorical("lora_rank", rank_choices)
    else:
        rank = trial.suggest_categorical("lora_rank", [16, 32, 64, 128])

    # Alpha strategy
    if args.alphas:
        if args.alphas in ("equal", "double"):
            alpha_strategy = args.alphas
        else:
            alpha_choices = [int(a) for a in args.alphas.split(",")]
            alpha = trial.suggest_categorical("lora_alpha", alpha_choices)
            return {"lora_rank": rank, "lora_alpha": alpha}
    else:
        alpha_strategy = trial.suggest_categorical("alpha_strategy", ["equal", "double"])

    alpha = rank if alpha_strategy == "equal" else rank * 2

    # Learning rate
    if args.lrs:
        lr_choices = [float(lr) for lr in args.lrs.split(",")]
        lr = trial.suggest_categorical("lr", lr_choices)
    else:
        lr = trial.suggest_float("lr", 5e-6, 5e-4, log=True)

    # Weight decay
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)

    # Warmup ratio (fraction of total steps)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15, step=0.03)

    # Scheduler
    lr_scheduler = trial.suggest_categorical("lr_scheduler", ["cosine", "linear"])

    # RSLoRA (recommended for rank >= 64)
    use_rslora = trial.suggest_categorical("use_rslora", [True, False])

    # Gradient accumulation (affects effective batch size)
    grad_accum = trial.suggest_categorical("grad_accum", [4, 8, 16])

    return {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler": lr_scheduler,
        "use_rslora": use_rslora,
        "grad_accum": grad_accum,
    }


def run_trial(base_recipe: dict, hp: dict, data_dir: str, trial_dir: str,
              steps: int, eval_split: float, args) -> float:
    """Run a single short SFT trial and return eval loss."""
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Merge base recipe with trial hyperparameters
    recipe = dict(base_recipe)
    recipe.update(hp)

    if args.four_bit:
        recipe["four_bit"] = True
    if args.fp8:
        recipe["fp8"] = True

    model_name = recipe["model"]
    max_seq_len = recipe.get("max_seq_len", 4096)
    use_bf16 = is_bfloat16_supported()

    # Load model
    load_kwargs = dict(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=recipe.get("four_bit", False),
        load_in_8bit=recipe.get("eight_bit", False),
        dtype=recipe.get("dtype"),
    )
    if recipe.get("fp8"):
        load_kwargs["load_in_4bit"] = False
        load_kwargs["load_in_8bit"] = False
        load_kwargs["load_in_fp8"] = True
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
        target_modules=recipe.get("lora_targets", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=hp.get("use_rslora", False),
        random_state=args.seed,
    )

    # Load data (inline — can't import from 08_train_sft_unsloth due to numeric prefix)
    examples = []
    for f in sorted(Path(data_dir).glob("ft_*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if not examples:
        raise RuntimeError(f"No training data in {data_dir}")

    def format_example(example, _tok=tokenizer):
        messages = example.get("messages", [])
        text = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = Dataset.from_list(examples).map(format_example)
    split = dataset.train_test_split(test_size=eval_split, seed=args.seed)

    # Short training run
    training_args = SFTConfig(
        output_dir=trial_dir,
        max_steps=steps,
        per_device_train_batch_size=recipe.get("batch_size", 1),
        gradient_accumulation_steps=hp.get("grad_accum", 8),
        learning_rate=hp["lr"],
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=max(steps // 5, 5),
        warmup_ratio=hp.get("warmup_ratio", 0.05),
        weight_decay=hp.get("weight_decay", 0.01),
        lr_scheduler_type=hp.get("lr_scheduler", "cosine"),
        optim=recipe.get("optim", "adamw_8bit"),
        seed=args.seed,
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        packing=recipe.get("packing", False),
        report_to="none",
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
    )

    trainer.train()

    # Get final eval loss
    eval_result = trainer.evaluate()
    eval_loss = eval_result.get("eval_loss", float("inf"))

    # Cleanup to free VRAM for next trial
    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eval_loss


def main():
    args = parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    base_recipe = load_base_recipe(args.recipe)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect data directory
    data_dir = args.data_dir
    if not data_dir:
        conf_path = Path(__file__).resolve().parent.parent / "repo.conf"
        repo_key = "unknown__repo"
        if conf_path.exists():
            for line in conf_path.read_text().splitlines():
                if line.strip().startswith("REPO_KEY="):
                    repo_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
        candidates = [
            f"./logs/sft_data/{repo_key}",
            "./logs/sft_data",
            "./sft_data",
        ]
        data_dir = next((d for d in candidates if Path(d).exists()), candidates[0])

    print("=== Hyperparameter Optimization ===")
    print(f"  Base recipe:    {args.recipe}")
    print(f"  Model:          {base_recipe.get('model', 'Qwen/Qwen3-8B')}")
    print(f"  Data:           {data_dir}")
    print(f"  Trials:         {args.n_trials}")
    print(f"  Steps/trial:    {args.steps_per_trial}")
    print(f"  Eval split:     {args.eval_split}")
    if args.ranks:
        print(f"  Ranks:          {args.ranks}")
    if args.lrs:
        print(f"  LRs:            {args.lrs}")
    print()

    # Set up Optuna study
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="minimize",
        study_name="swe-gym-hpo",
        pruner=pruner,
    )

    def objective(trial):
        hp = build_search_space(trial, args)
        trial_dir = str(out_dir / f"trial_{trial.number:03d}")

        print(f"\n--- Trial {trial.number} ---")
        for k, v in hp.items():
            print(f"  {k}: {v}")

        try:
            eval_loss = run_trial(
                base_recipe, hp, data_dir, trial_dir,
                args.steps_per_trial, args.eval_split, args,
            )
            print(f"  eval_loss: {eval_loss:.4f}")
            return eval_loss
        except Exception as e:
            print(f"  FAILED: {e}")
            return float("inf")

    study.optimize(objective, n_trials=args.n_trials)

    # Report results
    print("\n" + "=" * 60)
    print("=== HPO Complete ===")
    print(f"  Best trial:     {study.best_trial.number}")
    print(f"  Best eval_loss: {study.best_trial.value:.4f}")
    print("  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # Build best recipe
    best_params = study.best_trial.params
    best_recipe = dict(base_recipe)

    # Map Optuna params back to recipe format
    if "lora_rank" in best_params:
        best_recipe["lora_rank"] = best_params["lora_rank"]
    if "lora_alpha" in best_params:
        best_recipe["lora_alpha"] = best_params["lora_alpha"]
    elif "alpha_strategy" in best_params:
        rank = best_params["lora_rank"]
        best_recipe["lora_alpha"] = rank if best_params["alpha_strategy"] == "equal" else rank * 2
    if "lr" in best_params:
        best_recipe["lr"] = best_params["lr"]
    if "weight_decay" in best_params:
        best_recipe["weight_decay"] = best_params["weight_decay"]
    if "warmup_ratio" in best_params:
        # Convert ratio to approximate steps (user can adjust)
        best_recipe["warmup_ratio"] = best_params["warmup_ratio"]
        best_recipe.pop("warmup_steps", None)
    if "lr_scheduler" in best_params:
        best_recipe["lr_scheduler"] = best_params["lr_scheduler"]
    if "use_rslora" in best_params:
        best_recipe["use_rslora"] = best_params["use_rslora"]
    if "grad_accum" in best_params:
        best_recipe["grad_accum"] = best_params["grad_accum"]

    # Save best recipe
    best_recipe_path = out_dir / "best_recipe.yaml"
    with open(best_recipe_path, "w") as f:
        f.write(f"# Best recipe from HPO ({args.n_trials} trials, {args.steps_per_trial} steps/trial)\n")
        f.write(f"# Best eval_loss: {study.best_trial.value:.4f}\n")
        f.write(f"# Base recipe: {args.recipe}\n\n")
        yaml.dump(best_recipe, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Best recipe saved to: {best_recipe_path}")

    # Save full study results
    results = []
    for trial in study.trials:
        results.append({
            "trial": trial.number,
            "eval_loss": trial.value if trial.value != float("inf") else None,
            "params": trial.params,
            "state": trial.state.name,
        })
    results_path = out_dir / "hpo_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  All results saved to: {results_path}")

    print("\nTo train with best params:")
    print(f"  uv run python scripts/08_train_sft_unsloth.py --recipe {best_recipe_path}")


if __name__ == "__main__":
    main()
