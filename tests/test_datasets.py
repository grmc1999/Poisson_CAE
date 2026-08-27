"""Unit tests for dataset generators / loaders.

2D synthetic generators need only torch/numpy; sklearn-dependent datasets
(banana, rings, breast_cancer) are skipped if sklearn is unavailable.
"""

import pytest

torch = pytest.importorskip("torch")


def test_two_spirals_shapes_and_labels():
    from Utils.datasets import make_two_spirals

    x, y = make_two_spirals(n=1000, seed=0)
    assert x.shape == (1000, 2)
    assert x.dtype == torch.float32
    assert y.shape == (1000,)
    assert set(y.tolist()) <= {0, 1}
    # deterministic under the same seed
    x2, y2 = make_two_spirals(n=1000, seed=0)
    assert torch.equal(x, x2)


def test_get_experiment_loaders_spirals():
    from Utils.datasets import LoaderCfg, get_experiment_loaders

    loader, test_loader, input_dim, task = get_experiment_loaders(
        "spirals", LoaderCfg(batch_size=64, shuffle=True, drop_last=True), seed=0
    )
    assert input_dim == 2
    assert task == "classification"
    assert test_loader is None
    batch = next(iter(loader))
    assert len(batch) == 2  # (x, y)
    assert batch[0].shape[1] == 2


def test_loaders_reconstruction_mog(monkeypatch):
    # mog is handled in main.py directly; here just ensure the dataloader path
    # for a plain reconstruction-style dataset works via the menu.
    from Utils.datasets import LoaderCfg, get_experiment_loaders

    # breast_cancer requires sklearn
    sklearn = pytest.importorskip("sklearn.datasets")
    loader, test_loader, input_dim, task = get_experiment_loaders(
        "breast_cancer",
        LoaderCfg(batch_size=32, shuffle=True, drop_last=True),
        seed=0,
    )
    assert input_dim == 30
    assert task == "classification"
    assert test_loader is not None
    batch = next(iter(loader))
    assert batch[0].shape[1] == 30
