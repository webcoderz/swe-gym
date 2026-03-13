"""Read repo.conf — the single source of truth for repo configuration."""

from pathlib import Path

_CONF_PATH = Path(__file__).resolve().parent.parent / "repo.conf"
_cache: dict[str, str] = {}


def _load() -> dict[str, str]:
    if _cache:
        return _cache
    if not _CONF_PATH.exists():
        raise FileNotFoundError(f"repo.conf not found at {_CONF_PATH}")
    for line in _CONF_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Expand simple ${VAR} references
        while "${" in val:
            start = val.index("${")
            end = val.index("}", start)
            ref = val[start + 2 : end]
            val = val[:start] + _cache.get(ref, "") + val[end + 1 :]
        _cache[key] = val
    return _cache


def get(key: str, default: str | None = None) -> str:
    """Get a config value by key."""
    conf = _load()
    if key in conf:
        return conf[key]
    if default is not None:
        return default
    raise KeyError(f"{key} not found in repo.conf")


# Convenience accessors
REPO_OWNER = property(lambda self: get("REPO_OWNER"))
REPO_NAME = property(lambda self: get("REPO_NAME"))
