"""Hermes trading worker — paper-mode, self-improving strategy runner."""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"


def state_dir() -> Path:
    """Locate the state/ directory. /app/state on Railway, ~/hermes-trading/state
    locally, overridable with HERMES_STATE_DIR."""
    env = os.getenv("HERMES_STATE_DIR")
    if env:
        return Path(env)
    for candidate in (Path("/app/state"), Path.home() / "hermes-trading" / "state"):
        if candidate.exists():
            return candidate
    return Path("state")
