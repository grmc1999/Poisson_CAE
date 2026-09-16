"""Tests for reconstruction-sample generation."""
import json
import tempfile
from pathlib import Path

import torch

from Utils.config import load_config, to_dict
from Utils.pipeline import build_components
from Utils.sampling import generate_samples


def _make_dummy_run(tmp_path: Path) -> Path:
    """Create a minimal run_dir with config.yml + model_last.pt (mog, d=2)."""
    cfg = load_config("configs/mog.yaml")
    cfg.data.seed = 0
    cfg.train.steps = 3
    cfg.train.viz_every = 0

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Save config.yml
    from Utils.config import save_config
    save_config(cfg, run_dir / "config.yml")

    # Build model and save checkpoint
    model, _Pi, _est, _loader, _test, input_dim, task = build_components(cfg, "cpu")
    ckpt = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "cfg": to_dict(cfg), "task": task, "input_dim": input_dim}
    torch.save(ckpt, run_dir / "model_last.pt")
    return run_dir


def test_sampling_generates_files():
    with tempfile.TemporaryDirectory() as td:
        run_dir = _make_dummy_run(Path(td))
        results = generate_samples(run_dir, modes=["gaussian", "mask", "dropout"],
                                   device="cpu", n=4, seed=0)
        for mode in ["gaussian", "mask", "dropout"]:
            assert mode in results
            assert "recon_mse" in results[mode]
            assert "corrupt_mse" in results[mode]
            mode_dir = run_dir / f"samples_{mode}"
            assert (mode_dir / "recon_samples.png").exists(), f"PNG missing for {mode}"
            assert (mode_dir / "sample_metrics.json").exists(), f"JSON missing for {mode}"
            with open(mode_dir / "sample_metrics.json") as f:
                m = json.load(f)
            assert m["n"] == 4
            assert isinstance(m["recon_mse"], float)


def test_sampling_default_mode():
    with tempfile.TemporaryDirectory() as td:
        run_dir = _make_dummy_run(Path(td))
        results = generate_samples(run_dir, device="cpu", n=4)
        assert "gaussian" in results
        assert (run_dir / "samples_gaussian" / "recon_samples.png").exists()
