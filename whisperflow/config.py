"""Configuration management for WhisperFlow."""

import os
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(os.environ.get("WHISPERFLOW_CONFIG_DIR", "~/.config/whisperflow")).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.yaml"
DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "default_config.yaml"


def ensure_config_dir() -> None:
    """Create config directory and default files if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        shutil.copy2(DEFAULT_CONFIG, CONFIG_FILE)
    # Create empty dictionary and snippets files
    for fname in ("dictionary.json", "snippets.json", "style_examples.json"):
        fpath = CONFIG_DIR / fname
        if not fpath.exists():
            fpath.write_text("{}")


def load_config() -> dict[str, Any]:
    """Load configuration, merging defaults with user overrides."""
    ensure_config_dir()
    with open(DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    # Expand paths
    for section in ("dictionary", "snippets", "style", "history"):
        if "path" in config.get(section, {}):
            config[section]["path"] = str(Path(config[section]["path"]).expanduser())
    return config


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to user config file."""
    ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json_file(path: str) -> dict:
    """Load a JSON config file, returning empty dict if missing."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_json_file(path: str, data: dict) -> None:
    """Save data to a JSON config file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
