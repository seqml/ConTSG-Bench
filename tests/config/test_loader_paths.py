from __future__ import annotations

import pytest

from contsg.config.loader import ConfigLoader


def test_loader_requires_model_dataset_generator_yaml(tmp_path):
    config_dir = tmp_path / "configs"
    generators_dir = config_dir / "generators"
    generators_dir.mkdir(parents=True)

    (generators_dir / "verbalts_synth-m.yaml").write_text(
        "seed: 42\n"
        "device: cuda:0\n"
        "train:\n"
        "  epochs: 321\n"
        "data:\n"
        "  name: synth-m\n"
        "  data_folder: ./datasets/synth-m\n"
        "  n_var: 2\n"
        "  seq_length: 128\n"
        "  normalize: false\n"
        "model:\n"
        "  name: verbalts\n"
        "  channels: 123\n",
        encoding="utf-8",
    )

    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.from_args(dataset="synth-m", model="verbalts")

    assert cfg.data.name == "synth-m"
    assert cfg.model.name == "verbalts"
    assert cfg.model.channels == 123
    assert cfg.train.epochs == 321
    assert cfg.data.normalize is False


def test_loader_errors_when_model_dataset_generator_yaml_missing(tmp_path):
    config_dir = tmp_path / "configs"
    (config_dir / "generators").mkdir(parents=True)

    loader = ConfigLoader(config_dir=config_dir)

    with pytest.raises(
        FileNotFoundError,
        match=r"configs/generators/my_model_debug\.yaml|generators/my_model_debug\.yaml",
    ):
        loader.from_args(dataset="debug", model="my_model")
