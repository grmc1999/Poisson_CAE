"""Unit tests for Utils.config (YAML load / overrides / serialization).

Pure-Python; runs without torch.
"""

import os
import tempfile

import pytest

from Utils.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    apply_overrides,
    load_config,
    to_dict,
)


def _write(tmp_path, content):
    p = os.path.join(tmp_path, "cfg.yaml")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(content)
    return p


def test_load_config_roundtrip(tmp_path):
    p = _write(
        tmp_path,
        """
name: banana
data:
  experiment: banana
  batch_size: 64
  seed: 7
model:
  hidden: 256
  z_dim: 16
train:
  lr: 1e-4
  steps: 100
""",
    )
    cfg = load_config(p)
    assert cfg.name == "banana"
    assert cfg.data.experiment == "banana"
    assert cfg.data.batch_size == 64
    assert cfg.data.seed == 7
    assert cfg.model.hidden == 256
    assert cfg.model.z_dim == 16
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.train.steps == 100

    d = to_dict(cfg)
    assert d["data"]["experiment"] == "banana"
    assert d["train"]["steps"] == 100


def test_default_name_from_experiment(tmp_path):
    p = _write(tmp_path, "data:\n  experiment: spirals\n")
    cfg = load_config(p)
    assert cfg.name == "spirals"
    assert cfg.data.batch_size == DataConfig().batch_size  # default preserved


def test_apply_overrides_typed_scalars():
    cfg = ExperimentConfig()
    cfg = apply_overrides(cfg, {"train.lam": "5e-3", "data.seed": "3", "train.steps": "1234", "train.viz_every": "0"})
    assert cfg.train.lam == pytest.approx(5e-3)
    assert cfg.data.seed == 3
    assert cfg.train.steps == 1234
    assert cfg.train.viz_every == 0
    assert isinstance(cfg.train.viz_every, int)


def test_apply_overrides_invalid_key_raises():
    cfg = ExperimentConfig()
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"bogus.x": "1"})
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"train.nope": "1"})


def test_apply_overrides_type_coercion_string():
    cfg = ExperimentConfig()
    cfg = apply_overrides(cfg, {"data.experiment": "rings"})
    assert cfg.data.experiment == "rings"
    assert isinstance(cfg.data.experiment, str)
