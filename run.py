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


def build_pipeline(cfg: ExperimentConfig, device: str):
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

    train(
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
        viz_dir=cfg.train.viz_dir,
    )

    metrics = {"task": task, "input_dim": input_dim}
    if test_loader is not None:
        metrics.update(evaluate(model, test_loader, task, device))

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

    metrics = build_pipeline(cfg, device)
    metrics["run_id"] = run_dir.name
    metrics_to_json(metrics, run_dir / "metrics.json")
    return metrics


if __name__ == "__main__":
    main()
