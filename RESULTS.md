# Results — Poisson-CAE (running log)

Experiments live on the ICA cluster at
`/share_zeta/Proxy-Sim/guillermo.carrillo/Poisson_CAE/results/`. Every run is a
self-describing directory `results/<exp>/<method-tag>_<ts>_<rand>/` containing
`config.yml`, `metrics.json` (+ method block), `loss_history.json`,
`losses_step.png`, and `viz/` field plots. Method tag = scheme-kernel-(R|k)-t-\lambda-seed.

Plan / experiment tables: `PLAN.md`. Codebase: `run.py`, `main.py`,
`Utils/estimator_factory.py`, `Utils/variational_estimator.py`.

---

## Phase 1 — method comparison on banana (2D classification)

### Sep 11 finding — variational beats kernel, and is numerically stable

The `banana_variational` sweep ran scheme ∈ {compact, variational} × seeds {0,1}
(4 jobs, exit 0):

| scheme | seed | recon | flux | bulk | loss |
|---|---|---|---|---|---|
| compact (diffusion, R=2, t=0.25) | 0 | 0.897 | **−3206** | +8.9 | **−31.2** |
| compact (diffusion, R=2, t=0.25) | 1 | 2.486 | **−5364** | +315 | **−54.3** |
| variational (Ritz, i=5) | 0 | 0.072 | ~0 | −9e-5 | 0.072 |
| variational (Ritz, i=5) | 1 | 0.113 | ~0 | −3e-4 | 0.113 |

- The compact-diffusion **flux term explodes** (≈ −3e3…−5e3), producing
  pathological negative losses. This motivated dropping the wide kernel sweeps
  and making **variational the primary solver** (kernel = tuned ablation).
- The variational arm is stable and its loss tracks the reconstruction term.

Run ids: `results/banana/c-d-R2-t0.25-lam0.01-s{0,1}_20260911_143818_*`,
`results/banana/v-d-i5-t0.25-lam0.01-s{0,1}_20260911_143735_*`.

### Field-magnitude probe (`v_mag`, `gradv_mag`) — committed `6ed2bac`

Added per-step + final recording of `‖v‖` (mean |potential|) and
`‖∇v‖` (mean gradient norm) at queried points, in `loss_history.json` and
`metrics.json`. Settles the "is the field degenerate?" caveat raised in `PLAN.md`:

**The variational potential is meaningful, not ≈0** — `‖v‖ ≈ 6.9`,
`‖∇v‖ ≈ 1.3–1.8`. The apparent `flux ≈ 0` in earlier runs was a float-formatting
artifact (true flux ≈ 3e-2).

### Screening sweep — batch 1 (µ ∈ {0, 0.01}, seeds {0,1}; jobs 601532–601535, exit 0)

| µ | seed | recon | flux | bulk | loss | ‖v‖ | ‖∇v‖ |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0715 | +3.5e-2 | −9.3e-5 | 0.0719 | 6.94 | 1.76 |
| 0 | 1 | 0.1133 | −9.3e-3 | −3.4e-4 | 0.1132 | 6.91 | 1.34 |
| 0.01 | 0 | 0.0715 | +3.7e-2 | −9.5e-5 | 0.0719 | 6.91 | 1.75 |
| 0.01 | 1 | 0.1133 | −9.3e-3 | −3.4e-4 | 0.1132 | 6.87 | 1.34 |

Readings:
- **µ=0.01 is indistinguishable from µ=0** (loss identical to 4 decimals, field
  shifts ~0.5%). Screening only becomes visible at larger µ — batch 2
  (µ ∈ {0.1, 1.0}) is the discriminating test.
- **Contractive term is currently marginal**: `λ·(flux−bulk) ≈ 3.5e-4` vs recon
  0.0715 (~0.5%). Whether the potential improves robustness vs a plain AE will be
  answered by the `banana_var_lambda` sweep (λ=0 baseline), not yet run.

### Screening sweep — batch 2 (µ ∈ {0.1, 1.0}, seeds {0,1}; jobs 601572–601575, exit 0)

Full screening table (complete):

| µ | seed | recon | flux | bulk | loss | ‖v‖ | ‖∇v‖ |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0715 | +3.5e-2 | −9.3e-5 | 0.0719 | 6.94 | 1.76 |
| 0 | 1 | 0.1133 | −9.3e-3 | −3.4e-4 | 0.1132 | 6.91 | 1.34 |
| 0.01 | 0 | 0.0715 | +3.7e-2 | −9.5e-5 | 0.0719 | 6.91 | 1.75 |
| 0.01 | 1 | 0.1133 | −9.3e-3 | −3.4e-4 | 0.1132 | 6.87 | 1.34 |
| 0.1 | 0 | 0.0715 | +2.8e-2 | −9.6e-5 | 0.0718 | 6.61 | 1.72 |
| 0.1 | 1 | 0.1133 | −9.4e-3 | −3.3e-4 | 0.1132 | 6.58 | 1.31 |
| 1.0 | 0 | 0.0715 | +1.6e-2 | −7.6e-5 | 0.0717 | 4.62 | 1.44 |
| 1.0 | 1 | 0.1133 | −1.3e-2 | −2.8e-4 | 0.1132 | 4.61 | 1.10 |

Conclusions from the screening sweep:
- **µ compresses the field and quiets the flux**: `‖v‖` 6.94 → 4.62 and
  flux 3.5e-2 → 1.6e-2 as µ goes 0 → 1 — screening acts as designed (coercivity).
- **...but loss is flat across µ** (0.0719 → 0.0717; recon pinned at 0.0715 /
  0.1133). The contractive term is present yet does not move the training
  objective; the classification/recon term dominates. This makes the λ sweep the
  decisive test of whether the potential matters at all.

---

## Running log

| # | Sweep | Jobs | State |
|---|---|---|---|
| 1 | banana_variational (scheme × seeds) | 601516–601519 | ✔ completed |
| 2 | banana_var_screening (µ sweep, complete) | 601532–601535, 601572–601575 | ✔ completed |
| 3 | banana_var_inner (inner_steps × seeds) | pending | – |
| 4 | banana_var_lambda (λ × seeds) | pending | – |
| 5 | banana_var_bc (lam_d × seeds) | pending | – |

Sweeps generated under `cluster/jobs/banana_var_{screening,inner,lambda,bc}`.
Submit 4 at a time with `sbatch run_XXXX.sh` from the sweep dir.