"""
Register a custom PythonProfile for the private repo at runtime.

Import this module before using any swesmith functionality that requires
the profile to be in the registry (bug generation, validation, etc.).

Usage:
    from register_profile import ensure_profile_registered
    ensure_profile_registered()

Or as a sitecustomize-style auto-import:
    python -c "import register_profile;
    register_profile.ensure_profile_registered()"
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field

# Read config from repo.conf (single source of truth)
from conf import get as _conf

REPO_OWNER = _conf("REPO_OWNER")
REPO_NAME = _conf("REPO_NAME")
PYTHON_VERSION = _conf("PYTHON_VERSION", "3.12")

_registered = False


def _ensure_github_token():
    """Set GITHUB_TOKEN from gh CLI if not already set.

    swesmith uses ghapi which reads GITHUB_TOKEN from env,
    not the gh CLI auth store.
    """
    if os.environ.get("GITHUB_TOKEN"):
        return
    try:
        result = subprocess.run(
            "gh auth token",
            shell=True, capture_output=True, text=True, check=True,
        )
        token = result.stdout.strip()
        if token:
            os.environ["GITHUB_TOKEN"] = token
    except Exception:
        pass  # Will run unauthenticated; ghapi will warn


def ensure_profile_registered(commit: str | None = None):
    """Register the private repo profile if not already done."""
    global _registered
    if _registered:
        return

    # Bridge gh CLI auth → GITHUB_TOKEN env var for ghapi
    _ensure_github_token()

    from swesmith.profiles.base import registry
    from swesmith.profiles.python import PythonProfile

    # Check if already registered
    repo_key = f"{REPO_OWNER}__{REPO_NAME}"
    try:
        registry.get(repo_key)
        _registered = True
        return
    except KeyError:
        pass

    # Resolve commit if not provided
    if commit is None:
        commit = os.getenv("SWESMITH_COMMIT")

    if commit is None:
        # Use gh CLI to resolve latest commit
        try:
            result = subprocess.run(
                f"gh api repos/{REPO_OWNER}/{REPO_NAME}/commits/main --jq .sha",
                shell=True, capture_output=True, text=True, check=True,
            )
            commit = result.stdout.strip()
        except Exception:
            # Fallback: use a known commit
            commit = "566cf8e0"

    # Capture resolved values as defaults so registry can do cls() with no args
    _owner = REPO_OWNER
    _repo = REPO_NAME
    _commit = commit
    _pyver = PYTHON_VERSION
    _install = _conf("INSTALL_CMD", "python -m pip install -e '.[dev]'")
    _test_cmd = _conf(
        "TEST_CMD",
        "source /opt/miniconda3/bin/activate; conda activate testbed; "
        "pytest --disable-warnings --color=no --tb=no --verbose",
    )

    @dataclass
    class PrivateRepoProfile(PythonProfile):
        owner: str = _owner
        repo: str = _repo
        commit: str = _commit
        python_version: str = _pyver
        install_cmds: list[str] = field(
            default_factory=lambda: [_install],
        )
        test_cmd: str = _test_cmd

        @property
        def repo_name(self):
            """Return short key (owner__repo) so clone dir matches
            the CLI arg passed to swesmith commands."""
            return f"{self.owner}__{self.repo}"

        def _mirror_exists(self):
            """For private repos, check the owner's org, not swesmith org."""
            if self._cache_mirror_exists is not True:
                mirror = f"{self.owner}__{self.repo}.{self.commit[:8]}"
                try:
                    self.api.repos.get(
                        owner=self.owner, repo=mirror,
                    )
                    self._cache_mirror_exists = True
                except Exception:
                    self._cache_mirror_exists = False
            return self._cache_mirror_exists

        @property
        def mirror_name(self):
            """Mirror lives in owner's org, not swesmith org."""
            return (
                f"{self.owner}/"
                f"{self.owner}__{self.repo}.{self.commit[:8]}"
            )

        @property
        def mirror_url(self):
            """Use HTTPS for cloning (gh token handles auth)."""
            return f"https://github.com/{self.mirror_name}"

        @property
        def _mirror_ssh_url(self):
            """HTTPS push URL (no SSH needed)."""
            return f"https://github.com/{self.mirror_name}.git"

        def create_mirror(self):
            """Push repo content to mirror using HTTPS + gh token."""
            if self._mirror_exists():
                # Check if mirror has content (size > 0)
                try:
                    info = self.api.repos.get(
                        owner=self.owner,
                        repo=f"{self.owner}__{self.repo}"
                             f".{self.commit[:8]}",
                    )
                    if info.get("size", 0) > 0:
                        return
                except Exception:
                    pass

            token = os.environ.get("GITHUB_TOKEN", "")
            mirror_repo = (
                f"{self.owner}__{self.repo}.{self.commit[:8]}"
            )
            source_url = (
                f"https://x-access-token:{token}@github.com/"
                f"{self.owner}/{self.repo}.git"
            )
            push_url = (
                f"https://x-access-token:{token}@github.com/"
                f"{self.owner}/{mirror_repo}.git"
            )

            # Create mirror repo if it doesn't exist
            try:
                self.api.repos.get(
                    owner=self.owner, repo=mirror_repo,
                )
            except Exception:
                subprocess.run(
                    f"gh repo create {self.owner}/{mirror_repo}"
                    f" --private --confirm",
                    shell=True, check=True,
                    capture_output=True,
                )

            # Clone source, checkout commit, push to mirror
            tmp_dir = f"/tmp/_mirror_{mirror_repo}"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            cmds = " && ".join([
                f"git clone {source_url} {tmp_dir}",
                f"cd {tmp_dir}",
                f"git checkout {self.commit}",
                "rm -rf .git",
                "git init",
                'git config user.name "swesmith"',
                'git config user.email "swesmith@anon.com"',
                "rm -rf .github/workflows .github/dependabot.y*",
                "git add --force .",
                "git commit --no-gpg-sign -m 'Initial commit'",
                "git branch -M main",
                f"git remote add origin {push_url}",
                "git push -u origin main --force",
            ])
            print(f"[mirror] Pushing code to {self.owner}/"
                  f"{mirror_repo}...")
            subprocess.run(
                cmds, shell=True, check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Cleanup
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

            self._cache_mirror_exists = True
            print(f"[mirror] Done.")

    registry.register_profile(PrivateRepoProfile)

    # Ensure mirror has content (create_mirror is idempotent)
    profile = PrivateRepoProfile()
    profile.create_mirror()

    # Also register under the short key (owner__repo) so lookups
    # without the commit suffix work (e.g., REPO_KEY from repo.conf)
    short_key = f"{_owner}__{_repo}"
    if short_key not in registry.data:
        registry.data[short_key] = PrivateRepoProfile

    _registered = True
    print(f"[profile] Registered {_owner}/{_repo} @ {_commit[:8]}")
