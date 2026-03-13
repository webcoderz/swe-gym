# SWE-Gym 🏋️

### Get your agents BUFF 💪💪

> *Your repo is the gym. Bugs are the weights. Skills are the gains.*

Turn any Python repository into an RL gym for training coding agents. Synthesize bugs, run agents against them, and evolve optimized skill files — all on autopilot. 🤖🔁

Built on [SWE-Smith](https://github.com/SWE-bench/SWE-smith) for task generation and [GEPA gskill](https://gepa-ai.github.io/gepa/guides/gskill/) for skill optimization.

**No reps skipped. No bugs left standing. Train cheap, deploy expensive.** 🚀

## What This Does

1. **SWE-Smith** turns a target repo into an RL gym by mining real commits, introducing bugs, and producing verifiable task instances with test suites
2. **GEPA gskill** runs agents on batches of tasks in parallel Docker containers, then uses a reflection model to iteratively optimize a skill file
3. The output `best_skills.txt` is deployed as `.claude/skills/<REPO_NAME>/SKILL.md` and injected into Claude Code's prompt to improve its resolve rate on that codebase

## Prerequisites

- **Docker** running (`docker ps` to verify)
- **gh CLI** authenticated (`gh auth login`) — used for private repo access
- **Python 3.10+** and **uv** (auto-installed by `01_build_env.sh` if missing)
- **Ubuntu/Linux or WSL** (SWE-Smith Docker images are x86_64 Linux)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/webcoderz/swe-gym.git
cd swe-gym
make setup
```

### 2. Create your `.env` file

Copy the example and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** | Used by SWE-Smith for bug generation and by GEPA for agent training/reflection |
| `GITHUB_TOKEN` | **Yes** | GitHub personal access token with `repo` scope. Needed for API calls and Docker builds for private repos. [Generate one here](https://github.com/settings/tokens) |
| `ANTHROPIC_API_KEY` | No | Only needed if evaluating with Claude Code (step 5) |
| `WANDB_API_KEY` | No | Enables live training dashboards for SFT. [Sign up free](https://wandb.ai) then run `wandb login` |

### 3. Configure `repo.conf` for your target repository

All repo-specific values live in **one file**: `repo.conf`. Edit it to point at the repo you want to build a gym for:

```bash
# ── Target Repository ────────────────────────────────────────
# The GitHub owner (org or user) and repo name.
# Example: for https://github.com/pallets/flask
REPO_OWNER=pallets
REPO_NAME=flask
PYTHON_VERSION=3.12

# ── Environment Build ────────────────────────────────────────
# Install command run inside Docker after cloning. Must install
# all deps needed to run the test suite.
INSTALL_CMD="pip install uv && uv sync --extra test"
# Test command run inside Docker containers during validation
# Default PythonProfile assumes conda activate testbed; override for pip-based repos
TEST_CMD="cd /testbed && uv run python -m pytest tests/ -m 'not integration and not eval' --disable-warnings --color=no --tb=short --verbose -p no:cacheprovider --continue-on-collection-errors"


# ── GEPA gskill Training (Step 4) ───────────────────────────
TRAIN_MODEL=gpt-5-mini           # Agent model (cheaper = faster training)
REFLECTION_MODEL=gpt-5.2-pro     # Reflection model (smarter = better skills)
TRAIN_WORKERS=6                   # Parallel Docker containers
MAX_METRIC_CALLS=600              # Evaluation budget
TRAIN_SIZE=200                    # Training task instances
VAL_SIZE=50                       # Validation instances
TEST_SIZE=100                     # Held-out test instances

# ── Bug Generation (Step 2) ─────────────────────────────────
BUGGEN_MODEL=openai/gpt-4o       # LiteLLM format: provider/model
BUGGEN_N_PER_FUNC=1              # Variants per function (1 = broad coverage)
BUGGEN_LM_MODIFY=400             # Max bugs: LM Modify strategy
BUGGEN_CLASS_BASIC=200            # Max bugs: Class Basic strategy
BUGGEN_FUNC_FUN=200               # Max bugs: Func Fun strategy
BUGGEN_LM_REWRITE=300            # Max bugs: LM Rewrite strategy
BUGGEN_PROCEDURAL=1000           # Max bugs: Procedural (AST) strategy

# ── SFT / GRPO Pipeline (Steps 6-9) ─────────────────────────
SFT_AGENT_MODEL=gpt-5-mini       # Model for trajectory generation
SFT_WORKERS=4                    # Workers for SFT pipeline
SFT_BASE_MODEL=Qwen/Qwen3-8B    # Base model for LoRA/GRPO (native tool calling)
```

All shell scripts (`source repo.conf`), Python scripts (`from conf import get`), and the Makefile (`include repo.conf`) read from this single file. **No other files need editing** when targeting a different repo.

#### Common `INSTALL_CMD` examples

| Toolchain | `INSTALL_CMD` |
|-----------|---------------|
| pip | `"python -m pip install -e '.[dev]'"` |
| uv | `"curl -LsSf https://astral.sh/uv/install.sh \| sh && uv sync"` |
| poetry | `"pip install poetry && poetry install"` |
| make | `"make install"` |
| requirements.txt | `"python -m pip install -r requirements-dev.txt"` |

## Quick Start

```bash
# 1. Smoke test (3 tasks, ~5 min) to verify everything works
make smoke-test

# 2. Run the full pipeline
make build-env          # Build Docker image (~10-30 min)
make generate-tasks     # Generate bug tasks (~20-60 min)
make validate-tasks     # Filter to valid tasks (~30-60 min)
make train              # GEPA optimization loop (~2-12 hrs)
make evaluate           # Test skills on held-out instances

# 3. Deploy learned skills
make deploy-skills      # Copies to .claude/skills/<REPO_NAME>/SKILL.md

# 4. (Optional) SFT — fine-tune a model on your codebase
make gen-trajectories     # Generate expert solve traces via SWE-agent
make prepare-sft          # Evaluate & convert to JSONL training data
make train-sft-unsloth    # Unsloth LoRA fine-tune (recommended)

# 5. (Optional) GRPO — RL training using gym tasks as reward
make train-grpo           # Model learns by solving bugs, not imitating
```

## Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  01 Build   │────>│  02 Generate │────>│  03 Validate │
│  Docker Env │     │  Bug Tasks   │     │  via Tests   │
└─────────────┘     └──────────────┘     └──────────────┘
                                               │
                              ┌────────────────┼────────────────┐
                              v                                 v
                    ┌──────────────────┐              ┌──────────────────┐
                    │  04 GEPA gskill  │              │  06 Generate     │
                    │  Optimization    │              │  Trajectories    │
                    └────────┬─────────┘              └────────┬─────────┘
                             v                                 v
                    ┌──────────────────┐              ┌──────────────────┐
                    │  05 Evaluate     │              │  07 Prepare SFT  │
                    │  ±skills         │              │  JSONL data      │
                    └────────┬─────────┘              └────────┬─────────┘
                             v                                 v
                    ┌──────────────────┐              ┌──────────────────┐
                    │  best_skills.txt │              │  08 Train SFT    │
                    │  → SKILL.md      │              │  (unsloth)       │
                    └──────────────────┘              └────────┬─────────┘
                                                              v
                                                    ┌──────────────────┐
                                                    │  09 GRPO RL      │
                                                    │  (unsloth)       │
                                                    └──────────────────┘
```

Three independent paths after step 3:
- **Left (steps 4-5)**: Prompt optimization — produces a skill file (no GPU needed)
- **Right (steps 6-8)**: SFT fine-tuning — produces custom model weights via imitation learning (GPU needed)
- **Step 9**: GRPO RL — model learns by solving bugs directly, rewarded by test suite (GPU needed)

### Step 1: Build Environment (`01_build_env.sh`)
Creates a Docker image containing the repo with all Python dependencies installed. Auto-installs `uv` if missing and sets up the swe-gym venv. The image is named `swesmith.x86_64.<owner>__<repo>.<commit>`.

### Step 2: Generate Tasks (`02_generate_tasks.sh`)
Synthesizes bug-introducing task instances using multiple strategies:
- **LM Modify**: LLM introduces subtle bugs into functions
- **LM Rewrite**: LLM rewrites functions from scratch
- **Procedural**: AST-based rule transformations (swap operators, remove if-blocks, etc.)

### Step 3: Validate Tasks (`03_validate_tasks.sh`)
Runs each candidate against the test suite inside Docker. Keeps only tasks that break 1+ existing tests while leaving other tests passing.

### Step 4: Train Skills (`04_train_skills.sh`)
The core optimization loop:
1. Splits validated tasks into train/val/test
2. Generates an initial skill from repo analysis
3. Runs mini-SWE-agent on training tasks in parallel Docker containers
4. Reflection model analyzes pass/fail traces
5. Proposes improved skills, repeats until budget exhausted

### Step 5: Evaluate (`05_evaluate.sh`)
Compares baseline vs skills-enhanced performance on held-out test instances. Supports both mini-SWE-agent and Claude Code.

### Step 6: Generate Trajectories (`06_gen_trajectories.sh`)
Runs SWE-agent on validated task instances to produce expert solve traces.

### Step 7: Prepare SFT Data (`07_prepare_sft.sh`)
Evaluates which trajectories actually resolved their task, then converts successful ones to chat-format JSONL suitable for fine-tuning.

### Step 8: Train SFT (`08_train_sft_unsloth.py`)
Fine-tune a model on expert trajectories using [Unsloth](https://unsloth.ai). Supports LoRA, QLoRA (4-bit, runs on 3GB VRAM!), full fine-tune, DPO, and continued pretraining. Default model is Qwen3-8B. Uses YAML recipe configs in `configs/unsloth/`.

```bash
make train-sft-unsloth           # Qwen3-8B LoRA (default)
make train-sft-unsloth-4bit      # QLoRA — minimal VRAM
make train-sft-unsloth-32b       # Qwen3-32B LoRA (4-bit, A100/4090)
make train-sft-unsloth-gptoss    # GPT-OSS 20B LoRA (long context)
make train-sft-nemotron          # Nemotron-3-Nano 30B MoE (SOTA agentic)
make train-sft-qwen3-coder       # Qwen3-Coder-Next 80B MoE (SOTA coding)
make train-sft-unsloth-full      # Full fine-tune
make train-dpo-unsloth           # DPO (after SFT)
make train-cpt-unsloth           # Continued pretraining (domain adaptation)
make train-recipe RECIPE=configs/unsloth/my_recipe.yaml  # Custom recipe
```

#### Available recipes (`configs/unsloth/`)

**SFT recipes:**

| Recipe | Model | Active Params | VRAM (4-bit) | Notes |
|--------|-------|---------------|--------------|-------|
| `qwen3_4b_lora.yaml` | Qwen3-4B | 4B | ~4GB | Fast iteration, testing |
| `qwen3_8b_lora.yaml` | Qwen3-8B | 8B | ~8GB | **Default.** Native tool calling |
| `qwen3_32b_lora.yaml` | Qwen3-32B | 32B | ~24GB | Strong reasoning, A100/4090 |
| `gpt_oss_20b_lora.yaml` | GPT-OSS 20B | 20B | ~16GB | Long context (16K) |
| `llama3_8b_lora.yaml` | Llama 3.1 8B | 8B | ~8GB | General purpose |
| `nemotron3_nano_lora.yaml` | Nemotron-3-Nano 30B | ~3.6B (MoE) | ~24GB | SOTA agentic, hybrid reasoning |
| `nemotron3_super_lora.yaml` | Nemotron-3-Super 120B | ~12B (MoE) | ~64-72GB | SOTA reasoning, 1M context |
| `qwen3_coder_next_lora.yaml` | Qwen3-Coder-Next 80B | ~3B (MoE) | ~46GB | **SOTA coding.** 70.6% SWE-Bench |

**GRPO recipes:**

| Recipe | Model | VRAM (4-bit) | Notes |
|--------|-------|--------------|-------|
| `grpo_qwen3_4b.yaml` | Qwen3-4B | ~8GB | Fast GRPO iteration |
| `grpo_qwen3_8b.yaml` | Qwen3-8B | ~16GB | Default GRPO |
| `grpo_nemotron3_nano.yaml` | Nemotron-3-Nano 30B | ~24GB | MoE GRPO |
| `grpo_qwen3_coder_next.yaml` | Qwen3-Coder-Next 80B | ~46GB | SOTA coding GRPO (GSPO) |
| `grpo_nemotron3_super.yaml` | Nemotron-3-Super 120B | ~64-72GB | SOTA reasoning GRPO, multi-GPU |
| `grpo_gpt_oss_120b.yaml` | GPT-OSS 120B | ~65GB | Multi-GPU required (4x A100) |

Recipes are YAML files — copy one and customize for your needs. CLI args override recipe values.

#### Multi-GPU Training

Unsloth supports two multi-GPU strategies:

**DDP (Distributed Data Parallel)** — one model copy per GPU, scales throughput linearly:
```bash
# Via Make (NGPUS defaults to 2):
make train-sft-multigpu                        # SFT with 2 GPUs
make train-sft-multigpu NGPUS=4                # SFT with 4 GPUs
make train-grpo-multigpu                       # GRPO with 2 GPUs

# Via torchrun directly:
torchrun --nproc_per_node=2 scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml

# Via accelerate:
accelerate launch scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_8b_lora.yaml
```

**Model Splitting** — for models too large for one GPU (e.g. 80B Qwen3-Coder-Next):
```bash
# Distributes model layers across GPUs:
make train-sft-split                           # Qwen3-Coder-Next across GPUs
make train-grpo-split                          # GRPO across GPUs

# Or directly:
python scripts/08_train_sft_unsloth.py --recipe configs/unsloth/qwen3_coder_next_lora.yaml --device-map balanced
```

#### Additional Training Features

| Feature | Flag | Notes |
|---------|------|-------|
| Train on completions only | `--train-on-completions` | ~1% accuracy boost (QLoRA paper) |
| Rank-Stabilized LoRA | `--use-rslora` | Better for high ranks (64+) |
| FP8 quantization | `--fp8` | 60% less VRAM, 1.4x faster (RTX 40/50, H100+) |
| QAT (Quantization-Aware) | `--qat-scheme int4` | Recovers ~70% accuracy lost in quantization |
| 500k+ context | `--tiled-mlp` | Tiled MLP: 60% less VRAM for long sequences |
| vLLM standby (GRPO) | `--vllm-standby` | Share weight memory (~9GB savings) |
| FP8 KV cache (GRPO) | `--float8-kv-cache` | 2x KV cache reduction (RTX 3090+/A100+) |
| Early stopping | `--early-stopping 3 --eval-split 0.1` | Stops when eval loss plateaus |
| Resume training | `--resume` | Continue from last checkpoint |
| Save GGUF | `--save-gguf q4_k_m` | For llama.cpp / Ollama deployment |
| Save MXFP4 | `--save-mxfp4` | 75% less disk space |
| Push to Hub | `--push-to-hub user/model` | Upload merged model to HuggingFace |
| 8-bit quantization | `--eight-bit` | Middle ground between 4-bit and 16-bit |
| Continued pretraining | `--cpt --data-dir ./corpus/` | Domain adaptation before SFT |

> **Legacy:** `make train-sft` still works via [torchtune](https://github.com/meta-pytorch/torchtune) (configs in `configs/torchtune/`), but torchtune has [stopped active development](https://github.com/meta-pytorch/torchtune/issues/2883). [torchforge](https://github.com/meta-pytorch/torchforge) is its successor but only supports full SFT and GRPO currently (no LoRA/DPO).

### Step 9: GRPO RL (`09_train_grpo.py`) 🏋️
Instead of imitating expert traces (SFT), the model **learns by doing** — it generates patches for gym tasks and gets rewarded when they pass the test suite. This is the true "gym workout" path.

```bash
make train-grpo              # Hybrid reward (format + test suite)
make train-grpo-test         # Full test suite reward (slower, better signal)
make train-grpo-fast         # Format-only reward (fast iteration)
make train-grpo-nemotron     # Nemotron-3-Nano MoE
make train-grpo-qwen3-coder   # Qwen3-Coder-Next (multi-GPU)
make train-grpo-nemotron-super # Nemotron-3-Super 120B (multi-GPU)
make train-grpo-gptoss-120b   # GPT-OSS 120B (4x A100)
make train-grpo-fp8          # FP8 + vLLM standby (RTX 40/50, H100+)
make train-grpo-multigpu     # DDP with 2 GPUs
```

Three reward modes:
- **test**: Run patches in Docker against the test suite. Ground truth signal, but slow.
- **format**: Check if output contains a valid unified diff. Fast but shallow.
- **hybrid** (default): 30% format + 70% test. Best of both worlds.

Loss variants:
- **grpo**: Standard Group Relative Policy Optimization (default)
- **dr_grpo**: DR-GRPO variant (removes KL divergence term)
- **dapo**: DAPO (one-sided clipping, set `epsilon_high: 0.28`)
- **bnpo**: BNPO variant

Advanced RL options (set in YAML recipe or GRPOConfig):
- **GSPO**: Set `importance_sampling_level: sequence` for Qwen-style sequence-level IS (vs token-level)
- **Off-policy corrections**: `vllm_importance_sampling_correction: true` with truncation cap
- **Long-context GRPO**: `unsloth_grpo_mini_batch` + `unsloth_logit_chunk_multiplier` for 7x longer context
- **MoE acceleration**: `moe_backend: grouped_mm` for 12x faster MoE training

> **Tip:** Expect 300+ training steps before rewards start increasing. Use 500+ task instances for best results. Models under 1.5B params may not reliably generate reasoning tokens.

## Key Tuning Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `TRAIN_WORKERS` | 6 | Parallel Docker containers. More = faster but more RAM |
| `MAX_METRIC_CALLS` | 600 | Total evaluation budget. Higher = better skills but costlier |
| `TRAIN_MODEL` | gpt-5-mini | Agent model. Cheaper models train faster, skills transfer up |
| `REFLECTION_MODEL` | gpt-5.2-pro | Smarter = better skill proposals |

## Output Structure

After training, results are in `gepa_results/logs/run_<timestamp>/`:

```
run_<timestamp>/
├── prompts/
│   └── best_skills.txt     # <-- The main deliverable
├── config.json              # Experiment settings
├── iterations.jsonl         # Per-batch evaluation metrics
├── proposer_calls/          # Full reflection call logs
├── cost_summary.txt         # Cost breakdown
├── gepa_state.bin           # Checkpoint for resuming
└── terminal.log             # Complete execution output
```

## Resuming Interrupted Training

```bash
./scripts/04_train_skills.sh --resume gepa_results/logs/run_XXXXXXXX
# or
make resume   # auto-picks the latest run
```

## Costs

Rough estimates (vary by repo size and task count):
- **Smoke test**: ~$0.50
- **Task generation**: ~$5-15 (LM modify/rewrite)
- **Full training (600 evals)**: ~$30-80
- **Evaluation**: ~$5-15
- **Trajectory generation** (SFT): ~$10-30
- **SFT training** (local): GPU time only — ~1-4 hrs on a single GPU for LoRA

Skills learned on cheaper models (gpt-5-mini) transfer to expensive models (Claude Opus) without retraining — train cheap, deploy expensive.

## Deploying Skills

The learned skill file is deployed to `.claude/skills/<REPO_NAME>/SKILL.md`. Claude Code automatically picks up files in `.claude/skills/` and injects them into the agent prompt.

The skill file typically contains:
- Repository-specific patterns and conventions
- Common bug patterns and how to fix them
- Test suite structure and how to run tests
- Architecture-specific guidance for navigation

## VRAM Requirements

Approximate VRAM needed per model size (minimums — actual usage may be higher):

| Model Params | QLoRA (4-bit) | LoRA (16-bit) | FP8 |
|-------------|---------------|---------------|-----|
| 3B | 3.5 GB | 8 GB | ~2.5 GB* |
| 7-8B | 5-6 GB | 19-22 GB | ~5 GB* |
| 14B | 8.5 GB | 33 GB | ~8 GB* |
| 32B | 26 GB | 76 GB | ~20 GB* |
| 70B | 41 GB | 164 GB | ~30 GB* |
| 120B (MoE) | ~65 GB | N/A | ~48 GB* |

*FP8 estimates with `--vllm-standby`. Requires RTX 40/50 series, H100, or newer.

**GPU compatibility:** NVIDIA GPUs with CUDA Capability 7.0+ (V100, T4, A100, H100, RTX series). FP8 requires Capability 8.9+ (RTX 40xx/50xx, H100, L4).

**If you hit OOM:** Reduce `batch_size` to 1-2, enable `--four-bit`, use `--offload-embedding`, or try `--fp8` on supported hardware.

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Code generation** | Qwen3-Coder-Next 80B | SOTA coding (70.6% SWE-Bench), MoE ~3B active |
| **Reasoning + agentic** | Nemotron-3-Super 120B | SOTA reasoning, 1M context, MoE ~12B active, multi-GPU |
| **Agentic tasks (small)** | Nemotron-3-Nano 30B | SOTA agentic, hybrid reasoning, MoE ~3.6B active |
| **General purpose (large)** | Qwen3-32B | Strong reasoning, fits on A100/4090 |
| **General purpose (small)** | Qwen3-8B | Native tool calling, good balance |
| **Fast iteration / testing** | Qwen3-4B | Quick experiments, low VRAM |
| **Long context** | GPT-OSS 20B/120B | OpenAI's open models, 16K+ context |

**Base vs Instruct models:**
- **Large datasets (1000+ rows):** Use base models for maximum customization
- **Small datasets (<300 rows):** Use instruct models — they preserve built-in capabilities
- **Medium (300-1000 rows):** Either works; instruct is easier

## LoRA Hyperparameter Guide

Quick reference for tuning LoRA fine-tuning runs:

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **Rank (r)** | 16-64 | Higher = more capacity; 64 is a good default |
| **Alpha** | = rank | Or `rank * 2` for more aggressive learning |
| **Learning rate** | 2e-4 (SFT), 5e-6 (RL/DPO) | Reduce if overfitting |
| **Epochs** | 1-3 | Beyond 3 risks overfitting |
| **Batch size** | 2 with grad_accum=8 | Effective batch ~16 |
| **Dropout** | 0 | Unsloth optimizes this internally |
| **Weight decay** | 0.01-0.1 | Increase to combat overfitting |
| **Warmup** | 5-10% of total steps | Stabilizes early training |
| **Scheduler** | cosine | Or linear |

**Signs of overfitting:** Training loss < 0.2, eval loss diverging. Fix: reduce LR, reduce epochs, increase weight decay, or add more data.

**Signs of underfitting:** Loss not decreasing. Fix: increase LR, increase rank, increase epochs, decrease batch size.

**RSLoRA** (`--use-rslora`): Scales alpha by `1/sqrt(r)` — recommended for high ranks (64+) to stabilize training.

**Don't want to guess?** Use automated HPO:
```bash
make hpo                    # Quick search (10 trials, ~30 min)
make hpo-thorough           # Thorough search (30 trials, ~2 hrs)

# Then train with the best params:
uv run python scripts/08_train_sft_unsloth.py --recipe ./hpo_output/best_recipe.yaml
```

The HPO script (`08b_hpo.py`) uses [Optuna](https://optuna.org) to search over rank, alpha, LR, weight decay, warmup, scheduler, RSLoRA, and gradient accumulation. Each trial runs a short training loop (50 steps by default) and picks the config with lowest eval loss.

## Dataset Format

The SFT pipeline (steps 6-7) auto-generates training data, but you can also bring your own. Supported formats:

**ChatML (default, recommended):**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**ShareGPT:**
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

**Alpaca instruction format:**
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**Raw text (for continued pretraining):**
```json
{"text": "Continuous text from documents, code, etc."}
```

Place your JSONL files in the data directory as `ft_*.jsonl` and the training scripts will auto-discover them.

**Data quality tips:**
- 100 rows minimum for basic results; 1000+ recommended
- Quality > quantity — clean noisy/incorrect examples
- For code tasks, include chain-of-thought reasoning in responses
- Mix domain-specific data with general conversational data to avoid catastrophic forgetting

## Continued Pretraining (CPT)

Domain adaptation before SFT — teach the model your codebase's patterns, APIs, and conventions:

```bash
make train-cpt-unsloth  # Requires raw text corpus in ./corpus/
```

CPT uses `UnslothTrainer` with separate embedding learning rates. It adds `lm_head` and `embed_tokens` to LoRA targets for better adaptation. After CPT, run SFT on the checkpoint:

```bash
python scripts/08_train_sft_unsloth.py --model ./cpt_output/merged/ --recipe configs/unsloth/qwen3_8b_lora.yaml
```

## Project Structure

```
swe-gym/
├── .claude/
│   └── skills/                    # Claude Code skills (auto-discovered)
│       ├── swe-gym-pipeline/      #   /swe-gym-pipeline — full gym session
│       ├── unsloth-sft/           #   /unsloth-sft — SFT/DPO/CPT training
│       ├── unsloth-grpo/          #   /unsloth-grpo — RL training
│       ├── unsloth-hpo/           #   /unsloth-hpo — hyperparameter search
│       └── unsloth-train/         #   /unsloth-train — combined training hub
├── .claude-plugin/
│   └── plugin.json                # Plugin manifest for distribution
├── configs/
│   ├── bug_gen/                   # Bug generation prompts
│   ├── pipeline/                  # Pipeline step configs
│   ├── torchtune/                 # Legacy torchtune configs
│   └── unsloth/                   # Unsloth training recipes (recommended)
├── scripts/
│   ├── 01_build_env.sh            # Step 1: Docker environment
│   ├── 02_generate_tasks.sh       # Step 2: Synthesize bugs
│   ├── 03_validate_tasks.sh       # Step 3: Filter valid tasks
│   ├── 04_train_skills.sh         # Step 4: GEPA skill optimization
│   ├── 05_evaluate.sh             # Step 5: Evaluate skills
│   ├── 06_gen_trajectories.sh     # Step 6: Generate solve traces
│   ├── 07_prepare_sft.sh          # Step 7: Prepare SFT data
│   ├── 08_train_sft_unsloth.py    # Step 8: SFT/DPO/CPT training
│   ├── 08b_hpo.py                 # Step 8b: Hyperparameter optimization
│   └── 09_train_grpo.py           # Step 9: GRPO RL training
├── Makefile                       # All make targets
├── repo.conf                      # Target repo configuration
└── pyproject.toml                 # Python dependencies
```

## Claude Code Skills 🏋️‍♂️ — Your Personal Trainer

SWE-Gym ships with built-in Claude Code skills — slash commands that act like a personal trainer spotting you through every lift. No memorizing CLI flags. Just tell your spotter what you want.

### Available Skills

| Skill | Command | What It Does |
|-------|---------|-------------|
| **Pipeline** | `/swe-gym-pipeline <step>` | Full gym session — build env, generate tasks, train, evaluate, deploy |
| **SFT Training** | `/unsloth-sft` | Supervised fine-tuning — LoRA, full FT, DPO, CPT, QAT, export |
| **GRPO Training** | `/unsloth-grpo` | RL training — reward modes, loss types, vLLM, memory optimization |
| **HPO** | `/unsloth-hpo` | Find optimal hyperparameters before committing to a full training run |
| **Training Hub** | `/unsloth-train <mode>` | Combined coach — decides which training approach to use, chains pipelines |

### Usage Examples

```bash
# "I want to run the full pipeline"
/swe-gym-pipeline full

# "Fine-tune Qwen3-8B with 4-bit"
/unsloth-sft --recipe configs/unsloth/qwen3_8b_lora.yaml --four-bit

# "Do GRPO from my SFT checkpoint"
/unsloth-grpo --from-sft ./sft_output/merged --reward-mode hybrid

# "Find the best hyperparameters"
/unsloth-hpo --recipe configs/unsloth/qwen3_8b_lora.yaml --n-trials 20

# "What training approach should I use?"
/unsloth-train compare
```

Skills work instantly when you clone the repo — they live in `.claude/skills/` and Claude Code auto-discovers them.

### Installing as a Plugin 🔌

Other users can install SWE-Gym's skills without cloning the full repo:

```bash
# Add the marketplace
/plugin marketplace add webcoderz/swe-gym

# Install the plugin
/plugin install swe-gym
```

Skills get namespaced: `/swe-gym:unsloth-sft`, `/swe-gym:unsloth-grpo`, etc.

The plugin manifest lives at `.claude-plugin/plugin.json`. See the [Claude Code plugins docs](https://docs.anthropic.com/en/docs/claude-code/plugins) for details on distribution, versioning, and private marketplaces.

## References

- [SWE-Smith](https://github.com/SWE-bench/SWE-smith) — NeurIPS 2025 D&B Spotlight
- [GEPA gskill guide](https://gepa-ai.github.io/gepa/guides/gskill/)
- [GEPA paper](https://arxiv.org/abs/2507.19457) — Reflective Prompt Evolution
- [GEPA blog: Learning Skills for Coding Agents](https://gepa-ai.github.io/gepa/blog/2026/02/18/automatically-learning-skills-for-coding-agents/)
- [Unsloth](https://unsloth.ai) — Fast fine-tuning (LoRA, QLoRA, FP8, GRPO, DPO, QAT)
- [Unsloth GRPO blog](https://unsloth.ai/blog/grpo) — GRPO training guide
- [Unsloth FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) — 60% less VRAM
- [Unsloth advanced RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) — GSPO, DAPO, clipping
- [Unsloth multi-GPU docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) — DDP and model splitting
- [Unsloth LoRA guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) — Hyperparameter tuning
- [Unsloth datasets guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) — Data format reference
