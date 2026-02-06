"""Configuration management for WhisperFlow."""

import logging
import os
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("WHISPERFLOW_CONFIG_DIR", "~/.config/whisperflow")).expanduser()
CONFIG_FILE = CONFIG_DIR / "config.yaml"
DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "default_config.yaml"

# Valid values for config validation
VALID_BACKENDS = {"local", "remote"}
VALID_DEVICES = {"cpu", "cuda", "auto"}
VALID_COMPUTE_TYPES = {"float32", "float16", "int8"}
VALID_LLM_PROVIDERS = {"ollama", "openai", "anthropic", "none"}
VALID_INJECTION_METHODS = {"auto", "xdotool", "xclip", "wl-copy", "wayland"}
VALID_OVERLAY_POSITIONS = {"top-right", "top-left", "bottom-right", "bottom-left", "center"}


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


def validate_config(config: dict) -> list[str]:
    """Validate configuration values. Returns list of warnings."""
    warnings = []

    # Transcription
    trans = config.get("transcription", {})
    if trans.get("backend") not in VALID_BACKENDS:
        warnings.append(f"Invalid transcription.backend: '{trans.get('backend')}' (valid: {VALID_BACKENDS})")
    if trans.get("device") not in VALID_DEVICES:
        warnings.append(f"Invalid transcription.device: '{trans.get('device')}' (valid: {VALID_DEVICES})")
    if trans.get("compute_type") not in VALID_COMPUTE_TYPES:
        warnings.append(f"Invalid transcription.compute_type: '{trans.get('compute_type')}' (valid: {VALID_COMPUTE_TYPES})")

    # LLM
    llm = config.get("llm", {})
    if llm.get("provider") not in VALID_LLM_PROVIDERS:
        warnings.append(f"Invalid llm.provider: '{llm.get('provider')}' (valid: {VALID_LLM_PROVIDERS})")

    # Injection
    inj = config.get("injection", {})
    if inj.get("method") not in VALID_INJECTION_METHODS:
        warnings.append(f"Invalid injection.method: '{inj.get('method')}' (valid: {VALID_INJECTION_METHODS})")

    # Audio
    audio = config.get("audio", {})
    vad_thresh = audio.get("vad_threshold", 0.5)
    if not isinstance(vad_thresh, (int, float)) or not 0 <= vad_thresh <= 1:
        warnings.append(f"Invalid audio.vad_threshold: {vad_thresh} (must be 0.0-1.0)")
    sr = audio.get("sample_rate", 16000)
    if sr not in (8000, 16000, 22050, 44100, 48000):
        warnings.append(f"Unusual audio.sample_rate: {sr} (expected 16000)")

    # UI
    ui = config.get("ui", {})
    if ui.get("overlay_position") not in VALID_OVERLAY_POSITIONS:
        warnings.append(f"Invalid ui.overlay_position: '{ui.get('overlay_position')}' (valid: {VALID_OVERLAY_POSITIONS})")

    # Hotkeys — basic check that they contain '+'
    hotkey = config.get("hotkey", {})
    for key_name in ("hold_key", "command_key", "whisper_mode_key"):
        val = hotkey.get(key_name, "")
        if not val or not isinstance(val, str):
            warnings.append(f"Invalid hotkey.{key_name}: '{val}'")

    return warnings


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

    # Validate and warn
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(f"Config: {w}")

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
