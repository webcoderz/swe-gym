"""
Wrapper that registers the private repo profile, then runs a swesmith module.

Usage:
    python scripts/swesmith_run.py swesmith.bug_gen.llm.modify REPO --n_bugs 50
    python scripts/swesmith_run.py swesmith.harness.run_validation REPO
"""

import runpy
import sys
from pathlib import Path

# Register the profile before any swesmith imports
sys.path.insert(0, str(Path(__file__).parent))
from register_profile import ensure_profile_registered

ensure_profile_registered()

# The first arg is the module to run, rest are passed through
if len(sys.argv) < 2:
    print("Usage: python scripts/swesmith_run.py <module> [args...]")
    sys.exit(1)

module = sys.argv[1]
sys.argv = sys.argv[1:]  # Shift so the module sees its own args

runpy.run_module(module, run_name="__main__", alter_sys=True)
