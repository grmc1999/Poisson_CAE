"""Re-render the loss plot of a finished run with the corruption-mode label.

The pre-2.0 loss plots carried no perturbation label; this helper re-creates
``losses_step.png`` (with ``corruption: <mode>`` in the title) from the run's
saved ``config.yml`` + ``loss_history.json``, matching ``run.py`` rendering.

Usage:
    python replot_loss.py --run_dir results/mog/v-d-..._abc1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Utils.config import load_config  # noqa: E402
from run import _plot_loss_history  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    args = ap.parse_args(argv)

    run_dir: Path = args.run_dir
    cfg = load_config(run_dir / "config.yml")
    mode = cfg.train.corruption_mode

    with open(run_dir / "loss_history.json", encoding="utf-8") as fh:
        history = json.load(fh)

    # steps_per_epoch unknown post-hoc; use a stride so ~40 epoch lines overlay.
    steps_per_epoch = max(1, (len(history) + 39) // 40)
    out_png = str(run_dir / "losses_step.png")
    _plot_loss_history(history, steps_per_epoch, out_png, mode=mode)
    print(f"replotted {run_dir / 'losses_step.png'} (corruption: {mode}, {len(history)} points)")


if __name__ == "__main__":
    main()