"""Portable model and artifact paths for ORPG evaluation."""
import os
from pathlib import Path
REMOTE_PROJECT_ROOT = Path(os.environ.get("ORPG_ROOT", Path(__file__).resolve().parents[2])).resolve()
MODEL_PATH = Path(os.environ.get("ORPG_MODEL", REMOTE_PROJECT_ROOT / "models/Qwen3-4B-Instruct-2507")).resolve()
