"""Config loading. configs/env.yaml is the single source of default values."""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_CONFIG_PATH = ROOT / "configs" / "env.yaml"


def deep_update(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in recursively."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_env_config(path=None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_ENV_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_ENV_CONFIG_PATH:
        with open(path) as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    return deep_update(cfg, overrides or {})
