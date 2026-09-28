"""Portable paths for the paper evaluation tools."""
import os
from pathlib import Path
REMOTE_PROJECT_ROOT = Path(os.environ.get("ORPG_ROOT", os.environ.get("CW_GRPO_ROOT", Path(__file__).resolve().parents[2]))).resolve()
MODEL_PATH = Path(os.environ.get("ORPG_MODEL", os.environ.get("CW_GRPO_MODEL_PATH", REMOTE_PROJECT_ROOT / "models/Qwen3-4B-Instruct-2507"))).resolve()
