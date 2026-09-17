"""Regenerate the (labelled) field visualization of a finished run.

Field plots are produced during training at each ``viz_every`` step, before the
corruption-mode label existed. Only the final weights are stored
(``model_last.pt``), so this re-renders the *final-step* field plot
(``viz/fields_step_<steps>.png``) with ``corruption: <mode>`` in the suptitle,
using the same VizConfig as training.

Usage:
    python replot_fields.py --run_dir results/mnist_flat/v-d-..._4905
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Utils.config import load_config  # noqa: E402
from Utils.pipeline import build_components  # noqa: E402
from Utils.visualization import visualize_fields, VizConfig  # noqa: E402
from models import Poisson_reg  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None,
                    help="Step label for the plot (default: cfg.train.steps)")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args(argv)

    run_dir: Path = args.run_dir
    cfg = load_config(run_dir / "config.yml")
    mode = cfg.train.corruption_mode
    step = args.step if args.step is not None else cfg.train.steps

    model, Pi, estimator, loader, _test, _input_dim, _task = build_components(cfg, args.device)
    ckpt = torch.load(run_dir / "model_last.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()

    PR = Poisson_reg(estimator, model)

    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    x = x.to(args.device)

    out_dir = str(run_dir / "viz")
    out = visualize_fields(
        model=model,
        poisson_reg=PR,
        projector=Pi,
        x_batch=x,
        out_dir=out_dir,
        step=step,
        device=args.device,
        cfg=VizConfig(grid_n=160, padding=0.75, landmarks=cfg.train.landmarks, dpi=160,
                      corruption_mode=mode),
    )
    print(f"replotted fields for {run_dir} (corruption: {mode}) -> {out}")


if __name__ == "__main__":
    main()