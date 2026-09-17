"""Batch reconstruction-sample generation across trained checkpoints.

For every run directory under results/<experiments>/ that contains
``model_last.pt``, produce a labelled reconstruction-sample grid for each
corruption mode and each draw seed:

    samples_<mode>_s<seed>/recon_samples.png + sample_metrics.json

and write a ``sample_summary.csv`` (run, train_corruption, sample_mode, seed,
recon_mse, corrupt_mse). Plots carry the perturbation name in the title.

Usage (on the cluster, inside the container):
    python generate_all_samples.py --results-dir results \
        --experiments mnist_flat,mog \
        --modes gaussian,mask,dropout,shift_scale,ddpm \
        --seeds 0,1,2 --n 8 --device cpu
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Utils.sampling import generate_samples  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--experiments", type=str, default="mnist_flat,mog",
                    help="Comma-separated subdir names under results-dir")
    ap.add_argument("--modes", type=str, default="gaussian,mask,dropout,shift_scale,ddpm",
                    help="Corruption modes to sample under (default all 5)")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="Comma-separated draw seeds (one grid each)")
    ap.add_argument("--n", type=int, default=8,
                    help="Number of samples per grid (default 8)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--summary-out", type=Path, default=Path("results/sample_summary.csv"))
    args = ap.parse_args(argv)

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_dirs: list[Path] = []
    for exp in experiments:
        root = args.results_dir / exp
        if not root.exists():
            print(f"[skip] no dir {root}")
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "model_last.pt").exists():
                run_dirs.append(d)

    if not run_dirs:
        print("no checkpoints found; nothing to do")
        return

    print(f"{len(run_dirs)} models × {len(modes)} modes × {len(seeds)} seeds = "
          f"{len(run_dirs) * len(modes) * len(seeds)} grids")

    rows: list[tuple] = []
    for run_dir in run_dirs:
        print(f"== {run_dir}", flush=True)
        for mode in modes:
            for seed in seeds:
                tag = f"s{seed}"
                out_dir = run_dir / f"samples_{mode}_{tag}"
                if (out_dir / "recon_samples.png").exists():
                    print(f"  [skip] samples_{mode}_{tag} exists", flush=True)
                    continue
                try:
                    res = generate_samples(
                        run_dir=run_dir, modes=[mode], device=args.device,
                        n=args.n, seed=seed, out_tag=tag,
                    )
                    rows.append((str(run_dir), mode, seed, res[mode]["recon_mse"],
                                 res[mode]["corrupt_mse"]))
                    print(f"  samples_{mode}_{tag} recon_mse={res[mode]['recon_mse']:.4g}", flush=True)
                except Exception as e:
                    print(f"  [fail] samples_{mode}_{tag}: {e!r}", flush=True)

    # Re-read metrics for the train corruption label on already-sampled dirs too
    if rows:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["run_dir", "sample_mode", "seed", "recon_mse", "corrupt_mse"])
            w.writerows(rows)
        print(f"wrote {args.summary_out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()