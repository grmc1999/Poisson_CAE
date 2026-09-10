"""Aggregate run metrics across results/ into a CSV table.

Each run produced by run.py writes:
    results/<name>/<timestamp>/config.yml
    results/<name>/<timestamp>/metrics.json

This walks results/ and flattens the metrics (plus a few top-level config keys)
into a single table for paper figures.

Usage
-----
    python scripts/collect_results.py results/ --out table.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Utils.config import to_dict  # noqa: E402


FLAT_KEYS = [
    "task", "input_dim", "accuracy", "mse", "correct", "total", "run_id",
]
CONFIG_KEYS = [
    "data.experiment", "data.seed", "data.batch_size",
    "model.hidden", "model.z_dim",
    "train.lam", "train.steps", "train.landmarks",
    "train.lr", "train.corruption_mode", "train.corruption_sigma",
    "estimator.scheme", "estimator.kernel_type", "estimator.t",
    "estimator.radius", "estimator.k", "estimator.max_neighbors",
    "estimator.mu", "estimator.inner_steps", "estimator.inner_lr",
]


def _config_key(cfg_dict: dict, key: str):
    cur = cfg_dict
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def collect(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        name = run_dir.parent.name if run_dir.parent else ""
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                metrics = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        cfg_path = run_dir / "config.yml"
        cfg_dict = {}
        if cfg_path.exists():
            try:
                cfg = load_config_file(cfg_path)
                cfg_dict = to_dict(cfg)
            except Exception:
                cfg_dict = {}
        row = {"results_name": name, "run_dir": str(run_dir.relative_to(root))}
        for k in FLAT_KEYS:
            row[k] = metrics.get(k, None)
        for k in CONFIG_KEYS:
            row[k] = _config_key(cfg_dict, k)
        rows.append(row)
    return rows


def load_config_file(path: Path):
    from Utils.config import load_config
    return load_config(str(path))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Collect run metrics into a CSV")
    ap.add_argument("root", type=Path, help="results/ directory")
    ap.add_argument("--out", type=Path, default=Path("table.csv"))
    args = ap.parse_args(argv)

    rows = collect(args.root)
    if not rows:
        print("No metrics.json found under", args.root)
        return 1
    headers = ["results_name", "run_dir"] + FLAT_KEYS + CONFIG_KEYS
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()