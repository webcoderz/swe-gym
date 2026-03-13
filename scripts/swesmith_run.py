"""
Wrapper that registers the private repo profile, then runs a swesmith module.

Usage:
    python scripts/swesmith_run.py swesmith.bug_gen.llm.modify REPO --n_bugs 50
    python scripts/swesmith_run.py swesmith.harness.run_validation REPO
"""

import os
import runpy
import sys
from pathlib import Path

# Register the profile before any swesmith imports
sys.path.insert(0, str(Path(__file__).parent))
from register_profile import ensure_profile_registered

ensure_profile_registered()


# ---------------------------------------------------------------------------
# Monkey-patch: use HTTPS + GITHUB_TOKEN instead of SSH for private repos.
#
# Upstream swesmith's harness requires an SSH key to git-fetch inside Docker
# containers.  We bypass this by:
#   1. Generating a temporary ed25519 SSH key (satisfies the _find_ssh_key
#      check + the copy-to-container step)
#   2. Hooking container.start() to rewrite the git remote from SSH → HTTPS
#      and configure a credential helper with GITHUB_TOKEN
#
# Result: git fetch inside the container uses HTTPS+PAT, the SSH key is
# never actually used but its presence prevents the ValueError.
# ---------------------------------------------------------------------------
def _patch_harness_for_https_auth():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return

    from swesmith.profiles.base import _find_ssh_key
    if _find_ssh_key() is not None:
        return  # SSH key exists → no patch needed

    import subprocess
    import swesmith.harness.utils as harness_utils
    from swebench.harness.constants import DOCKER_USER, DOCKER_WORKDIR

    # Generate a real (but throwaway) SSH key to satisfy the harness
    dummy_key = Path("/tmp/_swesmith_dummy_ssh_key")
    if not dummy_key.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(dummy_key),
             "-N", "", "-C", "swesmith-dummy"],
            capture_output=True,
        )

    # Make _find_ssh_key return our dummy
    import swesmith.profiles.base as _base_mod
    _base_mod._find_ssh_key = lambda: dummy_key
    harness_utils._find_ssh_key = lambda: dummy_key

    # Hook container starts to inject HTTPS auth
    import docker as _docker
    _orig_create = _docker.models.containers.ContainerCollection.create

    def _create_with_https(self, *args, **kwargs):
        container = _orig_create(self, *args, **kwargs)
        _orig_start = container.start

        def _start_then_configure(*a, **kw):
            result = _orig_start(*a, **kw)
            # Set up HTTPS credential helper
            container.exec_run(
                "git config --global credential.helper "
                "'!f() { echo username=x-access-token; "
                f"echo password={token}; "
                "};f'",
                user=DOCKER_USER,
            )
            # Rewrite SSH remote → HTTPS
            rv = container.exec_run(
                "git remote get-url origin",
                workdir=DOCKER_WORKDIR, user=DOCKER_USER,
            )
            if rv.exit_code == 0:
                url = rv.output.decode("utf-8").strip()
                if url.startswith("git@"):
                    https_url = url.replace(
                        "git@github.com:", "https://github.com/"
                    )
                    container.exec_run(
                        f"git remote set-url origin {https_url}",
                        workdir=DOCKER_WORKDIR, user=DOCKER_USER,
                    )
            return result

        container.start = _start_then_configure
        return container

    _docker.models.containers.ContainerCollection.create = _create_with_https
    print("[patch] HTTPS+token auth enabled for Docker containers (no SSH needed)")


_patch_harness_for_https_auth()


# The first arg is the module to run, rest are passed through
if len(sys.argv) < 2:
    print("Usage: python scripts/swesmith_run.py <module> [args...]")
    sys.exit(1)

module = sys.argv[1]
sys.argv = sys.argv[1:]  # Shift so the module sees its own args

runpy.run_module(module, run_name="__main__", alter_sys=True)
