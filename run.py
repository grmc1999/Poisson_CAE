"""Config-driven training wrapper over `main.py`.

Loads a YAML experiment config, builds the model / corruption operator /
Poisson estimator / dataloaders exactly as `main.py` does, runs training,
evaluates the held-out set when available, and writes the resolved config plus
final metrics to `results/<name>/<timestamp>/`.

Examples
--------
    python run.py --config configs/banana.yaml
    python run.py --config configs/banana.yaml --train.lam 5e-3 --train.steps 8000
    python run.py --config configs/banana.yaml --data.seed 1 --train.viz_every 0
"""

import argparse

import torch

from Utils.config import (
    ExperimentConfig,
    apply_overrides,
    load_config,
    metrics_to_json,
    make_run_dir,
    save_config,
)
from Utils.datasets import LoaderCfg, get_experiment_loaders
from Utils.estimator_factory import build_estimator
from Utils.projectors import CorruptionConfig, CorruptionOperator

from models import (
    AE_model,
    Classifier_model,
    Encoder,
    Decoder,
    GRUEncoder,
    Regressor_model,
)


def build_pipeline(cfg: ExperimentConfig, device: str, run_dir=None):
    """Build and run the full training pipeline for this config.

    Returns a metrics dict (task, input_dim, optional test accuracy/MSE).
    """
    torch.manual_seed(cfg.data.seed)

    # Data loaders
    if cfg.data.experiment == "mog":
        N = 5000
        centers = torch.tensor(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.25]], dtype=torch.float32
        )
        comp = torch.randint(0, centers.size(0), (N,))
        x = centers[comp] + 0.15 * torch.randn(N, 2)
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(
            TensorDataset(x), batch_size=cfg.data.batch_size, shuffle=True, drop_last=True
        )
        test_loader = None
        input_dim = 2
        task = "reconstruction"
    else:
        loader, test_loader, input_dim, task = get_experiment_loaders(
            cfg.data.experiment,
            LoaderCfg(
                batch_size=cfg.data.batch_size, shuffle=True, drop_last=True, num_workers=0
            ),
            seed=cfg.data.seed,
        )

    # Model selection by task (mirrors main.py)
    if task == "reconstruction":
        model = AE_model(
            Encoder(d=input_dim, h=cfg.model.hidden, z=cfg.model.z_dim),
            Decoder(z=cfg.model.z_dim, h=cfg.model.hidden, d=input_dim),
        )
    elif task == "classification":
        # all classification tasks here are binary (2 classes)
        model = Classifier_model(
            Encoder(d=input_dim, h=cfg.model.hidden, z=cfg.model.z_dim), n_classes=2
        )
    elif task == "regression":
        if cfg.data.encoder_type == "gru":
            enc = GRUEncoder(T=input_dim, din=1, hidden=cfg.model.hidden, z_dim=cfg.model.z_dim)
        else:
            enc = Encoder(d=input_dim, h=cfg.model.hidden, z=cfg.model.z_dim)
        model = Regressor_model(enc, out_dim=3)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Corruption operator Pi
    Pi = CorruptionOperator(
        CorruptionConfig(
            mode=cfg.train.corruption_mode,
            T=cfg.train.corruption_T,
            beta_start=cfg.train.corruption_beta_start,
            beta_end=cfg.train.corruption_beta_end,
            sigma=cfg.train.corruption_sigma,
        )
    )

    # Potential estimator (scheme / kernel_type / t from cfg.estimator)
    estimator = build_estimator(cfg.estimator, d=input_dim)

    # Reuse main.py's training loop
    from main import train

    viz_dir = str(run_dir / "viz") if run_dir is not None else cfg.train.viz_dir
    train_out = train(
        model=model,
        Pi=Pi,
        poisson_est=estimator,
        dataloader=loader,
        lr=cfg.train.lr,
        lam=cfg.train.lam,
        landmarks=cfg.train.landmarks,
        device=device,
        steps=cfg.train.steps,
        viz_every=cfg.train.viz_every,
        viz_dir=viz_dir,
    ) or {}

    metrics = {"task": task, "input_dim": input_dim}
    metrics.update(train_out.get("final") or {})
    if isinstance(estimator, torch.nn.Module):
        metrics["bailouts"] = int(getattr(estimator, "n_bailouts", 0))
    if run_dir is not None:
        history = train_out.get("history") or []
        steps_per_epoch = train_out.get("steps_per_epoch") or 1
        if history:
            import json

            from pathlib import Path

            with open(run_dir / "loss_history.json", "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)
            metrics["history_points"] = len(history)
            try:
                _plot_loss_history(history, steps_per_epoch, str(run_dir / "losses_step.png"))
                metrics["loss_plot"] = "losses_step.png"
            except Exception as e:  # plotting must never kill a completed run
                print(f"[plot] warning: loss plot failed: {e}")
    if test_loader is not None:
        metrics.update(evaluate(model, test_loader, task, device))
    if run_dir is not None and (run_dir / "viz").exists():
        metrics["viz_files"] = len(
            list((run_dir / "viz").glob("fields_step_*.png"))
        )

    return metrics


def evaluate(model, test_loader, task, device):
    model.to(device).eval()
    correct = 0
    total = 0
    mse = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), (batch[1] if len(batch) > 1 else None)
            if task == "reconstruction":
                # y is absent / irrelevant: compare reconstruction to clean input.
                mse += ((model(x) - x) ** 2).mean().item()
                total += 1
                continue
            y_pred = model(x)
            if task == "classification":
                if y is None:
                    raise ValueError("classification test loader must provide labels")
                pred = y_pred.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
            elif task == "regression":
                mse += ((y_pred - y) ** 2).mean().item()
                total += 1
    if task == "classification":
        metrics = {"accuracy": correct / float(total), "correct": correct, "total": total}
    elif task in ("regression", "reconstruction"):
        metrics = {"mse": mse / float(total)}
    else:
        metrics = {}
    return metrics


def _plot_loss_history(history: list[dict], steps_per_epoch: int, out_png: str) -> None:
    """Render per-step loss curves with epoch boundaries overlaid.

    4 stacked panels (recon / flux / bulk / total loss) share the training-step
    x-axis; dashed vertical lines mark each epoch transition and a secondary top
    axis labels the epoch index (step / steps_per_epoch). Headless-safe (Agg).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    total_steps = max(steps[-1], 1)
    n_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch

    stride = max(1, (n_epochs + 39) // 40)
    epoch_bounds = [e * steps_per_epoch for e in range(stride, n_epochs + 1, stride)]

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    panels = [
        ("recon", "reconstruction\n(logp)"),
        ("flux", "BC flux"),
        ("bulk", "bulk / D_loss"),
        ("loss", "total loss"),
    ]
    for ax, (key, label) in zip(axes, panels):
        ax.plot(steps, [h[key] for h in history], lw=0.8)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        for b in epoch_bounds:
            ax.axvline(b, color="gray", ls="--", lw=0.5, alpha=0.7)
        if history:
            v = history[-1][key]
            ax.annotate(
                f"{v:.3g}",
                xy=(steps[-1], v),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )

    axes[-1].set_xlabel("training step")
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    ax_top = axes[0].secondary_xaxis("top")
    ax_top.set_xlabel("epoch")
    top_epochs = sorted({0, n_epochs // 2, n_epochs})
    top_epochs = [e for e in top_epochs if e * steps_per_epoch <= total_steps]
    ax_top.set_xticks([e * steps_per_epoch for e in top_epochs])
    ax_top.set_xticklabels([str(e) for e in top_epochs])

    fig.suptitle("training loss per step", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Config-driven Poisson-CAE training")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--out", type=str, default=None, help="Override results dir")
    ap.add_argument("--device", type=str, default=None, help="cuda/cpu")
    # Everything after the known options are dotted-path overrides that argparse
    # would otherwise reject, e.g. --train.lam 5e-3 --data.seed=1.
    args, extras = ap.parse_known_args(argv)
    args.overrides = extras
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    overrides = {}
    tokens = list(args.overrides)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        body = tok[2:]
        if "=" in body:
            key, val = body.split("=", 1)
            overrides[key] = val
            i += 1
            continue
        key = body
        val = tokens[i + 1] if i + 1 < len(tokens) else "True"
        overrides[key] = val
        i += 2
    apply_overrides(cfg, overrides)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg, root=args.out or "results")
    run_dir = run_dir.resolve()
    save_config(cfg, run_dir / "config.yml")

    metrics = build_pipeline(cfg, device, run_dir=run_dir)
    metrics["run_id"] = run_dir.name
    metrics["method"] = {
        "name": cfg.name,
        "experiment": cfg.data.experiment,
        "input_dim": metrics.get("input_dim"),
        "seed": cfg.data.seed,
        "scheme": cfg.estimator.scheme,
        "kernel_type": cfg.estimator.kernel_type,
        "t": cfg.estimator.t,
        "radius": cfg.estimator.radius,
        "k": cfg.estimator.k,
        "max_neighbors": cfg.estimator.max_neighbors,
        "normalize": cfg.estimator.normalize,
        "variational": {
            "mu": cfg.estimator.mu,
            "inner_steps": cfg.estimator.inner_steps,
            "inner_lr": cfg.estimator.inner_lr,
            "inner_max_grad_norm": cfg.estimator.inner_max_grad_norm,
            "lam_d": cfg.estimator.lam_d,
            "bilevel": cfg.estimator.bilevel,
        }
        if cfg.estimator.scheme == "variational"
        else None,
        "lam": cfg.train.lam,
        "corruption": cfg.train.corruption_mode,
        "steps": cfg.train.steps,
    }
    metrics_to_json(metrics, run_dir / "metrics.json")
    return metrics


if __name__ == "__main__":
    main()
