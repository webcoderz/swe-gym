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
INSTALL_CMD="python -m pip install -e '.[dev]'"

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

# ── SFT Pipeline (Steps 6-8) ────────────────────────────────
SFT_AGENT_MODEL=gpt-5-mini       # Model for trajectory generation
SFT_WORKERS=4                    # Workers for SFT pipeline
SFT_BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct  # Base model for LoRA
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
make gen-trajectories   # Generate expert solve traces via SWE-agent
make prepare-sft        # Evaluate & convert to JSONL training data
make train-sft          # Local LoRA fine-tune with torchtune
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
                    │  → SKILL.md      │              │  (torchtune)     │
                    └──────────────────┘              └──────────────────┘
```

Two independent paths after step 3:
- **Left (steps 4-5)**: Prompt optimization — produces a skill file (no GPU needed)
- **Right (steps 6-8)**: Model fine-tuning — produces custom model weights (GPU needed)

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

### Step 8: Train SFT (`08_train_sft.sh`)
Local fine-tuning with [torchtune](https://github.com/meta-pytorch/torchtune). Default: LoRA on Qwen2.5-Coder-7B (fits on a single 16-24GB GPU). Pass `--full` for full fine-tune (40GB+ VRAM or multi-GPU).

> **Note on torchtune → torchforge transition:** torchtune has [stopped active development](https://github.com/meta-pytorch/torchtune/issues/2883) in favor of [torchforge](https://github.com/meta-pytorch/torchforge), a new PyTorch-native platform for post-training with scale and agentic RL (GRPO) as first-class citizens. torchforge is currently experimental and only supports full SFT and GRPO (no LoRA or DPO yet). We'll migrate once torchforge reaches feature parity. In the meantime, torchtune continues to receive critical bug fixes and works fine for our use case.

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

## References

- [SWE-Smith](https://github.com/SWE-bench/SWE-smith) — NeurIPS 2025 D&B Spotlight
- [GEPA gskill guide](https://gepa-ai.github.io/gepa/guides/gskill/)
- [GEPA paper](https://arxiv.org/abs/2507.19457) — Reflective Prompt Evolution
- [GEPA blog: Learning Skills for Coding Agents](https://gepa-ai.github.io/gepa/blog/2026/02/18/automatically-learning-skills-for-coding-agents/)
