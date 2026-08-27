"""Smoke tests for model forward/loss shapes.

torch required; skip if unavailable.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from models import (
    AE_model,
    Classifier_model,
    Encoder,
    Decoder,
    GRUEncoder,
    Regressor_model,
    Poisson_reg,
)
from Utils.geometry_estimators import PoissonMCEstimator, PoissonMCConfig
from Utils.projectors import CorruptionOperator, CorruptionConfig


def test_ae_reconstruction_shapes():
    torch.manual_seed(0)
    d = 3
    enc = Encoder(d=d, h=16, z=8)
    dec = Decoder(z=8, h=16, d=d)
    model = AE_model(enc, dec)
    x = torch.randn(10, d)
    x_hat = model(x)
    assert x_hat.shape == x.shape
    loss = model.logp(x, x_hat)
    assert loss.dim() == 0 and float(loss.detach()) >= 0.0


def test_classifier_shapes():
    torch.manual_seed(1)
    d, C = 4, 2
    model = Classifier_model(Encoder(d=d, h=16, z=8), n_classes=C)
    x = torch.randn(12, d)
    logits = model(x)
    assert logits.shape == (12, C)
    y = torch.randint(0, C, (12,))
    loss = model.logp(y, logits)
    assert loss.dim() == 0


def test_regressor_shapes():
    torch.manual_seed(2)
    d, out = 5, 3
    model = Regressor_model(Encoder(d=d, h=16, z=8), out_dim=out)
    x = torch.randn(8, d)
    pred = model(x)
    assert pred.shape == (8, out)
    y = torch.randn(8, out)
    loss = model.logp(y, pred)
    assert loss.dim() == 0


def test_gru_encoder_shapes():
    torch.manual_seed(3)
    T = 10
    enc = GRUEncoder(T=T, din=1, hidden=8, z_dim=4)
    x = torch.randn(6, T)
    z = enc(x)
    assert z.shape == (6, 4)


def test_train_objective_terms_smoke():
    """Tiny end-to-end: Estimate_field_grads + BC/D losses run and return scalars."""
    torch.manual_seed(4)
    d = 2
    enc = Encoder(d=d, h=16, z=8)
    dec = Decoder(z=8, h=16, d=d)
    model = AE_model(enc, dec)

    pi = CorruptionOperator(CorruptionConfig(mode="gaussian", sigma=0.1))
    est = PoissonMCEstimator(PoissonMCConfig(eps=0.01, landmarks=16))
    pr = Poisson_reg(est, model)

    x = torch.randn(16, d, requires_grad=True)
    x_tilde, _ = pi(x)

    x_hat = model(x_tilde)
    logp = pr.ML_loss(x, x_hat)
    v, gradv = pr.Estimate_field_grads(x, x_tilde, landmarks=16)
    flux = pr.BC_loss(x, x_tilde, gradv)
    bulk = pr.D_loss(x, x, x_hat, gradv)

    for t in (logp, flux, bulk):
        assert t.dim() == 0
        assert torch.isfinite(t)
