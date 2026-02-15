"""Configuration management using OmegaConf."""

import os
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str, overrides: Optional[list] = None) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to configuration YAML file
        overrides: Optional list of override strings (e.g., ["training.batch_size=4"])

    Returns:
        DictConfig: Configuration object
    """
    # Load base config
    config = OmegaConf.load(config_path)

    # Apply overrides
    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_conf)

    # Resolve any interpolations
    OmegaConf.resolve(config)

    return config


def save_config(config: DictConfig, save_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configuration objects."""
    return OmegaConf.merge(*configs)


def config_to_dict(config: DictConfig) -> dict:
    """Convert DictConfig to regular dictionary."""
    return OmegaConf.to_container(config, resolve=True)
