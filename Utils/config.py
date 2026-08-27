"""Configuration dataclasses and YAML loading for config-driven runs.

`main.py` exposes a flat argparse CLI. To make experiments reproducible and
sweepable on the cluster without touching `main.py`'s internals, we define an
explicit nested config here and a thin `run.py` wrapper that resolves it into
the same arguments `main.py` would receive (via its own kwargs).

A run is described by three nested blocks:

  data:     experiment name, seed, batching
  model:    architecture knobs (encoder/decoder widths, latent dim, encoder type)
  train:    optimizer / loss / corruption / Poisson-estimator / viz settings

Configs are expressed as YAML; any leaf key can be overridden on the command
line using dotted paths, e.g.  --train.lam 5e-3  --data.experiment banana.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


# -----------------------------
# Data block
# -----------------------------
@dataclass
class DataConfig:
    experiment: str = "banana"          # mog/spirals/banana/rings/breast_cancer/sinusoid_reg
    batch_size: int = 256
    seed: int = 0
    encoder_type: str = "mlp"           # mlp | gru (gru used for sinusoid_reg)


# -----------------------------
# Model block
# -----------------------------
@dataclass
class ModelConfig:
    hidden: int = 128                   # MLP hidden width (Encoder/Decoder)
    z_dim: int = 32                     # latent dimension


# -----------------------------
# Train block
# -----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    lam: float = 1e-2
    landmarks: int = 256
    steps: int = 5000
    viz_every: int = 500
    viz_dir: str = "outputs"
    corruption_mode: str = "gaussian"   # gaussian | ddpm | shift_scale | mixture
    corruption_T: int = 200
    corruption_beta_start: float = 1e-4
    corruption_beta_end: float = 2e-2
    corruption_sigma: float = 0.1
    poisson_eps: float = 1e-2


# -----------------------------
# Top-level config
# -----------------------------
@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = ""                      # optional experiment label; defaults to data.experiment


# -----------------------------
# YAML (de)serialisation
# -----------------------------
def _replace_dotted(obj: Any, dotted_key: str, value: Any) -> Any:
    """Return a copy of `obj` with the leaf at 'a.b.c' set to `value`."""
    keys = dotted_key.split(".")
    cur = obj
    for k in keys[:-1]:
        if not is_dataclass(cur):
            raise ValueError(f"Cannot descend into non-dataclass at '{'.'.join(keys[:keys.index(k)])}'")
        cur = getattr(cur, k)
    setattr(cur, keys[-1], value)
    return obj


def _dataclass_to_dict(dc: Any) -> dict:
    out = {}
    for f in fields(dc):
        v = getattr(dc, f.name)
        out[f.name] = asdict(v) if is_dataclass(v) else v
    return out


def _coerce(value: Any, typ: Any) -> Any:
    """Coerce a parsed value (possibly a numeric-looking string) to field type."""
    typ = _strip_typing(typ)
    if typ in (str,):
        return value if isinstance(value, str) else str(value)
    if typ is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("1", "true", "yes", "on")
    if typ in (int, float):
        if isinstance(value, typ) and not isinstance(value, bool):
            return value
        # numeric strings, scientific notation, bools -> parse
        try:
            f = float(value)
        except (TypeError, ValueError):
            return value  # leave as-is; will raise in the dataclass if wrong
        if typ is int:
            return int(f)
        return f
    return value


def _build_config(data: dict[str, Any]) -> ExperimentConfig:
    allowed = {f.name for f in fields(ExperimentConfig)}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    unknown = set(data) - allowed - {"data", "model", "train"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

    def _mk(block_name: str, block_cls: type):
        raw = data.get(block_name)
        if raw is None:
            return block_cls()
        if not isinstance(raw, dict):
            raise ValueError(f"'{block_name}' block must be a mapping")
        kwargs = {}
        for f in fields(block_cls):
            if f.name in raw:
                kwargs[f.name] = _coerce(raw[f.name], f.type)
        return block_cls(**kwargs)

    return ExperimentConfig(
        data=_mk("data", DataConfig),
        model=_mk("model", ModelConfig),
        train=_mk("train", TrainConfig),
        name=data.get("name", ""),
    )


def load_config(path: str | os.PathLike) -> ExperimentConfig:
    """Load a YAML config file into an ExperimentConfig."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build_config(raw)
    if not cfg.name:
        cfg.name = cfg.data.experiment
    return cfg


def apply_overrides(cfg: ExperimentConfig, overrides: dict[str, str]) -> ExperimentConfig:
    """Apply dotted-path overrides (e.g. {'train.lam': '5e-3'}) to a config.

    Parsed values are attempted as JSON/YAML scalars and coerced to the
    corresponding dataclass field type.
    """
    for key, raw in overrides.items():
        # normalize 'data.x' / 'model.x' / 'train.x' keys
        norm_key = key
        for block in ("data", "model", "train"):
            if norm_key.startswith(f"{block}."):
                break
        else:
            raise ValueError(
                f"Override {key!r} must start with 'data.', 'model.' or 'train.'"
            )
        container = {"data": cfg.data, "model": cfg.model, "train": cfg.train}[block]
        fname = norm_key.split(".", 1)[1]
        fobj = next((f for f in fields(container) if f.name == fname), None)
        if fobj is None:
            raise ValueError(f"Unknown override key: {key}")
        parsed = _coerce(raw, fobj.type)
        setattr(container, fname, parsed)
    return cfg


def _resolve_type(ann) -> Any:
    """Resolve a dataclass field annotation (type or string from __future__
    annotations) to a concrete built-in type."""
    if not isinstance(ann, str):
        return ann
    # strip typing wrappers written as strings
    s = ann.strip()
    base = s.split("[", 1)[0].strip()
    return {
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
    }.get(base, ann)


def _strip_typing(typ: Any) -> Any:
    typ = _resolve_type(typ)
    if typ is None:
        return None
    name = getattr(typ, "_name", None) or str(typ)
    # typing.Optional[int] / Union[..., None]
    if hasattr(typ, "__args__") and (
        name in ("Optional", "Union") or name.startswith("Optional")
    ):
        for a in typ.__args__:
            if a is not type(None):
                return _strip_typing(a)
    return typ


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "data": _dataclass_to_dict(cfg.data),
        "model": _dataclass_to_dict(cfg.model),
        "train": _dataclass_to_dict(cfg.train),
    }


def save_config(cfg: ExperimentConfig, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)


def make_run_dir(cfg: ExperimentConfig, root: str = "results") -> Path:
    """Create and return results/<experiment>/<timestamp>/ for this run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / (cfg.name or cfg.data.experiment) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def metrics_to_json(metrics: dict[str, Any], path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)


def parse_cli_overrides(argv: list[str]) -> dict[str, str]:
    """Parse '--train.lam 5e-3 --seed 0' style args into an overrides dict."""
    out: dict[str, str] = {}
    re_behind = re.compile(r"^--(.+)$")
    for a in argv:
        m = re_behind.match(a)
        if not m:
            continue
        key = m.group(1)
        out[key] = "True"  # placeholder; replaced if a value follows
    return out
