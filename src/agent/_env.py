"""Load .env once, from the project root, for anything started outside uv."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    import os
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        # Never adopt a placeholder from a copied .env.example.
        if k not in os.environ and v and not v.endswith("..."):
            os.environ[k] = v


load()
