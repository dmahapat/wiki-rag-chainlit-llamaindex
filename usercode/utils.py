import os
import sys
from typing import Any, Dict, Optional

import yaml


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEYS_PATH = os.path.join(THIS_DIR, "apikeys.yml")


def load_api_keys(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML file containing API keys.

    Returns an empty dict if the file is missing or empty.
    """
    p = path or API_KEYS_PATH
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("apikeys.yml should contain a top-level mapping (dictionary)")
    return data


def get_openrouter_api_key(path: Optional[str] = None) -> Optional[str]:
    """Return the OpenRouter API key.

    Priority: environment variable `OPENROUTER_API_KEY` then YAML file value
    under key `openrouter_api_key` in `apikeys.yml`.
    """
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip() or None

    keys = load_api_keys(path)
    val = keys.get("openrouter_api_key") if isinstance(keys, dict) else None
    if isinstance(val, str):
        return val.strip() or None
    return None


def get_apikey(path: Optional[str] = None) -> Optional[str]:
    """Compatibility wrapper expected by other modules.

    Returns the OpenRouter API key using the same resolution
    as `get_openrouter_api_key`.
    """
    return get_openrouter_api_key(path)


def _mask(key: str) -> str:
    if not key:
        return ""
    visible = 4
    return ("*" * max(0, len(key) - visible)) + key[-visible:]


if __name__ == "__main__":
    key = get_openrouter_api_key()
    if key:
        print("OpenRouter API key loaded:", _mask(key))
        sys.exit(0)
    else:
        print(
            "No OpenRouter API key found. Set env var OPENROUTER_API_KEY or update usercode/apikeys.yml",
            file=sys.stderr,
        )
        sys.exit(1)
