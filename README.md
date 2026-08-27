# Poisson-Induced Potentials for Contractive Representations

Research code for a Poisson-based reformulation of **Contractive Auto-Encoder (CAE)**
regularization. Originated as a Tiny Paper at the GRaM workshop @ ICLR 2026; being
extended toward a main-track submission (see [`PLAN.md`](PLAN.md)).

## Theory recap

The classical CAE penalizes the Frobenius norm of the encoder Jacobian,

```
R_CAE = E_{x~p_D} || J_f(x) ||_F^2 ,   J_f(x) = ∇_x f(x)
```

Instead of optimizing this local penalty directly, we introduce an auxiliary potential
`v : Ω → R` as the solution of a **Poisson equation** whose source is contractivity,

```
Δ v(x) = s(x) := || J_f(x) ||_F^2   on Ω,   v|_{∂Ω} = 0
```

With `u = p_D` and `v` in Green's identity,

```
∫_Ω u Δv dx = ∫_{∂Ω} u ∂_n v dS  −  ∫_Ω ∇u·∇v dx
```

we obtain the decomposition `R_CAE = boundary term − interior score-coupling term`:

```
R_CAE = E_x [ s(x) ]
      = E_{x̃~Πψ(x)} [ ∇v(x̃)·n(x̃) ]   (boundary flux, via corruption operator Πψ)
      − E_x [ ∇log p_D(x)·∇v(x) ]         (interior coupling to the data score)
```

The **boundary term** is approximated by corrupting in-distribution inputs with a
corruption operator `Πψ` (Gaussian noise, DDPM, shift/scale, or a mixture), which
yields both the out-of-distribution evaluation points `x̃ = Πψ(x)` and a
normal-like direction `n(x) = (x̃ − x)/||x̃ − x||`. The **interior term** couples `v`
with the (unknown) score `∇log p_D` via the model's own score surrogate.

**Training objective**

```
L = L_downstream + λ ( flux − bulk )
```

where `flux` is the boundary-flux regularizer and `bulk` is the interior coupling term.

`v` and `∇v` are approximated by Monte-Carlo quadrature of the Green's function
representation `v(x) = ∫ G(x,y) s(y) dy` over landmark points, using the regularized
Laplacian Green's function `G_ε`.

## Repository layout

```
main.py                        Unified training entry point (reconstruction /
                               classification / regression), CLI-driven
models.py                      Poisson_reg (field gradients, flux/bulk losses),
                               AE/Classifier/Regressor heads, GRU encoder
run.py                         Config-driven wrapper over main (YAML + overrides)
Utils/
  grad_operations.py           Jacobian Frobenius norm, regularized Green's
                               function and its gradient
  projectors.py                Corruption operators Πψ (gaussian/ddpm/shift_scale/
                               mixture)
  geometry_estimators.py       PoissonMCEstimator (Monte-Carlo Green quadrature)
  datasets.py                  Toy + real dataset generators / loaders
  visualization.py             Grid-based (2D) and PCA-based (ND) field plots
  config.py                    Dataset/train/hyper-parameter config dataclasses
configs/                       Example YAML experiment configs
tests/                         CPU unit tests (pytest)
legacy/                        Archived stale entry points (see git history)
outputs_banana/, outputs_spiral/   Field visualizations
PLAN.md                        Research & execution plan for the main-track upgrade
```

## Supported experiments

| Experiment       | Task             | Input dim |
|------------------|------------------|-----------|
| `mog`            | reconstruction   | 2         |
| `spirals`        | classification   | 2         |
| `banana`         | classification   | 2         |
| `rings`          | classification   | 2         |
| `breast_cancer`  | classification   | 30        |
| `sinusoid_reg`   | regression       | 50        |

## Usage

### Direct CLI (original `main.py`)

```bash
python main.py --experiment banana --batch 256 --lr 1e-3 --lam 1e-2 \
               --landmarks 256 --steps 5000 --viz_every 500 --viz_dir outputs
```

### Config-driven (recommended)

```bash
python run.py --config configs/banana.yaml
# override on the command line:
python run.py --config configs/banana.yaml --train.lam 5e-3 --train.steps 8000
```

Each run writes its resolved config and metrics to
`results/<experiment>/<timestamp>/`.

### Tests

```bash
pytest tests/
```

The math-only tests (`green_reg`, `gradx_green_reg`) run without torch; tests that
require torch/scikit-learn will skip if those are not importable.

## Notes

- `legacy/` holds entry points that referenced APIs/classes deleted on the `images`
  branch (e.g. `ConvEncoderMNIST`, `Poisson_reg_latent`, `hutchinson_jacobian_fro_norm`).
  They are intentionally not runnable; recover the originals via git history if needed.
