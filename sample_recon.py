"""Generate reconstruction-sample grids from a completed run.

Usage
-----
    python sample_recon.py --run_dir results/v-d-i2-lam0.01-s0_20260915_123456_abc1 \
        --modes gaussian,mask,dropout \
        --n 8 --device cpu

Each corruption mode gets its own ``samples_<mode>/`` sub-directory under the
run directory, containing ``recon_samples.png`` and ``sample_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Utils.sampling import generate_samples


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True,
                    help="Path to the completed run directory (contains model_last.pt + config.yml)")
    ap.add_argument("--modes", type=str, default=None,
                    help="Comma-separated corruption modes to compare "
                         "(e.g. gaussian,mask,dropout,shift_scale,ddpm)")
    ap.add_argument("--n", type=int, default=8,
                    help="Number of sample images to display per mode (default 8)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="torch device to run the model on (default cpu)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for selecting the sample batch (default 42)")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else None
    results = generate_samples(
        run_dir=args.run_dir,
        modes=modes,
        device=args.device,
        n=args.n,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))
    return


if __name__ == "__main__":
    main()
