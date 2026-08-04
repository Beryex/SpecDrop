"""Config loading utilities."""

import yaml
import os


def load_config(path):
    """Load a YAML config file and return as a nested dict."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg, path):
    """Save config dict to YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
