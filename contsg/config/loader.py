"""
Configuration loader for ConTSG.

Handles loading and merging configurations from multiple sources:
- YAML files
- CLI arguments
- Environment variables
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from contsg.config.schema import (
    ExperimentConfig,
    TrainConfig,
    DataConfig,
    ModelConfig,
    EvalConfig,
    DATASET_PRESETS,
)


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Values from `override` take precedence over `base`.
    Nested dicts are merged recursively.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


class ConfigLoader:
    """
    Configuration loader that merges configs from multiple sources.

    Merge priority (lowest to highest):
    1. Built-in presets
    2. Base config file
    3. Dataset config file
    4. Model config file
    5. CLI arguments
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files. Defaults to ./configs
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")

    def from_yaml(self, path: str | Path) -> ExperimentConfig:
        """Load configuration from a single YAML file."""
        return ExperimentConfig.from_yaml(path)

    def _get_generator_config_path(self, dataset: str, model: str) -> Path:
        """Return the required generator config path for dataset+model mode."""
        return self.config_dir / "generators" / f"{model}_{dataset}.yaml"

    def from_args(
        self,
        dataset: str,
        model: str,
        data_folder: Optional[str | Path] = None,
        clip_folder: Optional[str | Path] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        seed: int = 42,
        device: str = "cuda:0",
        output_dir: str | Path = "experiments",
        **kwargs: Any,
    ) -> ExperimentConfig:
        """
        Build configuration from CLI arguments.

        Args:
            dataset: Dataset name
            model: Model name
            data_folder: Path to data folder (optional, uses preset if not provided)
            clip_folder: Path to CLIP model folder
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            seed: Random seed
            device: Device string
            output_dir: Output directory
            **kwargs: Additional model-specific parameters

        Returns:
            ExperimentConfig instance
        """
        generator_config_path = self._get_generator_config_path(dataset, model)
        if not generator_config_path.exists():
            raise FileNotFoundError(
                f"Required generator config not found: {generator_config_path}. "
                f"Use --config {generator_config_path} after creating it."
            )

        config_dict = self.from_yaml(generator_config_path).model_dump(mode="python")

        overrides: dict[str, Any] = {
            "seed": seed,
            "device": device,
            "output_dir": str(output_dir),
        }

        if data_folder:
            overrides.setdefault("data", {})["data_folder"] = str(data_folder)
        if clip_folder:
            overrides.setdefault("eval", {})["clip_model_path"] = str(clip_folder)
        if epochs is not None:
            overrides.setdefault("train", {})["epochs"] = epochs
        if batch_size is not None:
            overrides.setdefault("train", {})["batch_size"] = batch_size
        if lr is not None:
            overrides.setdefault("train", {})["lr"] = lr

        model_overrides = {
            key: value
            for key, value in kwargs.items()
            if value is not None and key not in ["dataset", "model", "data_folder", "clip_folder"]
        }
        if model_overrides:
            overrides.setdefault("model", {}).update(model_overrides)

        return ExperimentConfig(**deep_merge(config_dict, overrides))

    def load_with_overrides(
        self,
        base_config: str | Path,
        overrides: Optional[dict[str, Any]] = None,
    ) -> ExperimentConfig:
        """
        Load base config and apply overrides.

        Args:
            base_config: Path to base configuration file
            overrides: Dictionary of override values

        Returns:
            ExperimentConfig instance
        """
        base_config = Path(base_config)

        with open(base_config, "r") as f:
            config_dict = yaml.safe_load(f)

        if overrides:
            config_dict = deep_merge(config_dict, overrides)

        return ExperimentConfig(**config_dict)

    def merge_configs(
        self,
        *configs: str | Path | dict,
    ) -> ExperimentConfig:
        """
        Merge multiple configuration sources.

        Args:
            *configs: Paths to YAML files or config dictionaries.
                      Later configs override earlier ones.

        Returns:
            ExperimentConfig instance
        """
        merged: dict[str, Any] = {}

        for config in configs:
            if isinstance(config, (str, Path)):
                with open(config, "r") as f:
                    config_dict = yaml.safe_load(f) or {}
            else:
                config_dict = config

            merged = deep_merge(merged, config_dict)

        return ExperimentConfig(**merged)


# Singleton instance for convenience
_default_loader: Optional[ConfigLoader] = None


def get_loader() -> ConfigLoader:
    """Get the default config loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


def load_config(
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[str | Path] = None,
    **kwargs: Any,
) -> ExperimentConfig:
    """
    Convenience function to load configuration.

    Either provide (dataset, model) or config_path.

    Args:
        dataset: Dataset name
        model: Model name
        config_path: Path to YAML config file
        **kwargs: Additional arguments passed to ConfigLoader

    Returns:
        ExperimentConfig instance
    """
    loader = get_loader()

    if config_path:
        return loader.from_yaml(config_path)
    elif dataset and model:
        return loader.from_args(dataset=dataset, model=model, **kwargs)
    else:
        raise ValueError("Either (dataset, model) or config_path must be provided")
