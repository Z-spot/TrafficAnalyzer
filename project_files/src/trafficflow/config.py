"""Load YAML config from config/."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(name: str = "pipeline.yaml") -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)
