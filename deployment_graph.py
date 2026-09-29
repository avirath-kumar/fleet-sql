"""Expose the existing agent module to LangSmith's graph loader."""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent / "src" / "agent"
sys.path.insert(0, str(AGENT_DIR))

from agent import agent  # noqa: E402
