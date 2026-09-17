"""Evaluate a trained classifier under input perturbations.

Training-ablation protocol: load a run's ``model_last.pt`` + ``config.yml``,
evaluate clean test accuracy and accuracy under the run's training corruption
mode (or an explicit subset of modes). Writes ``perturbation_eval.json`` next to
the checkpoint.

Usage:
    python eval_perturbation.py --run_dir results/banana/v-d-...-gaussian_...
    python eval_perturbation.py --run_dir <dir> --modes gaussian,mask,dropout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Utils.pipeline import build_components, build_corruption_operator  # noqa: E402
from Utils.sampling import corruption_override, load_run_config  # noqa: E402


def evaluate_accuracy(model, loader, device, Pi=None) -> float:
    """Top-1 accuracy; if *Pi* is given, corrupt the inputs first."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            if Pi is not None:
                x = Pi(x)[0]
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / float(total) if total else float("nan")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--modes", type=str, default=None,
                    help="Comma-separated modes to evaluate (default: the run's training mode)")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args(argv)

    run_dir: Path = args.run_dir
    cfg = load_run_config(run_dir)
    train_mode = cfg.train.corruption_mode
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else [train_mode]

    model, _Pi, _est, _loader, test_loader, input_dim, task = build_components(cfg, args.device)
    if task != "classification":
        raise SystemExit(f"eval_perturbation expects a classification run, got task={task!r}")
    if test_loader is None:
        raise SystemExit("classification run has no test split")

    ckpt = torch.load(run_dir / "model_last.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()

    out = {
        "run_dir": str(run_dir),
        "task": task,
        "input_dim": input_dim,
        "seed": cfg.data.seed,
        "train_corruption": train_mode,
        "clean_accuracy": evaluate_accuracy(model, test_loader, args.device, None),
        "accuracy": {},
    }
    for m in modes:
        cfg_m = load_run_config(run_dir)
        corruption_override(cfg_m, {"mode": m})
        Pi_m = build_corruption_operator(cfg_m)
        out["accuracy"][m] = evaluate_accuracy(model, test_loader, args.device, Pi_m)

    with open(run_dir / "perturbation_eval.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()