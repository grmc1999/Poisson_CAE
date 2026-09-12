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
  objective; the classification/recon term dominates.

### Sep 11 root cause — the regularizer was inert *by construction* (stop-gradient)

**All of the above** (flat µ sweep, byte-identical λ sweep, λ(flux−bulk) ≈ 1e-6)
is explained by a single implementation fact: the inner field uses
`s = g_land.detach()` (stop-gradient per PLAN §A.3), so the **flux BC term has
zero gradient path to the encoder**, and the bulk term couples only through the
negligible `score_value` term (bulk ≈ −9e-5 — the score is ∥ ∇v on banana, so the
‖∇v‖² part vanishes). Sweep results therefore cannot discriminate anything about
the potential: changing λ left trained models byte-identical.

**Fix (implemented, validated locally):** true **bilevel** differentiation
through the inner solve — `EstimatorConfig.bilevel=True` (default). Implements a
BOP-style correction `gs = −∇ₛ⟨g,u⟩` with `H·u = b` (H = ∇²E(θ*), E inner energy;
conjugate-gradient solve), in `Utils/variational_estimator.py`
(`BOPPoissonSolve`). The old stop-gradient path is kept as `_forward_inert`
(`bilevel=False`) for the ablation. Two bugs found+fixed along the way: (1) inner
GD previously ran `energy.backward()` — now `torch.autograd.grad(energy, theta)`
so inner steps never backprop through `s` into the encoder; (2) the per-param
vjp merge zeroed the `∇v` vjp whenever the `v` vjp was None — exactly the
flux-only case — silently zeroing `gs`. PyTorch 2.13 gotcha: custom
`Function.forward` executes with grad disabled; everything is wrapped in
`torch.enable_grad()`.

**Validation:**

| check | result |
|---|---|
| FD test in `tests/test_localized.py` (all 50 tests pass) | bilevel gradient DFLUX→encoder matches finite-diff (≈4% at K=200, lr=0.1) with correct sign |
| IFT accuracy vs inner convergence | K=5 ratio −84 (meaningless) → K=200 ≈4% agreement — **need K ≥ ~50** |
| end-to-end smoke (banana, B=64, K=100, 10 steps, CPU) | clean run, 0.75 s/step |

**Repercussions for the sweep plan:**
- The µ screening and old λ sweeps are stale under the new mechanism (they tested
  an inert term) — µ sweep conclusions above stand only as a check of the field
  geometry, not of the regularizer's effect on training.
- The λ sweep must be **re-run with bilevel** and a convergence-sufficient K
  (K=100, lr=0.1 in the generated `banana_var_lambda_bl`; generated locally,
  regenerated on cluster for correct paths).
- The inner-GD sweep (now `banana_var_inner_bl`: K ∈ {10, 50, 200}) doubles as
  the BOP-fidelity curve and sets the default K for every later solver run,
  including the Phase 2 datasets.

**Sep 11 stability fix (before the re-sweep):** instrumented runs showed the inner
Ritz solve itself diverging to NaN (v/∇v non-finite in the *forward* at ~step
200–400, µ=0; µ=1e-2 alone insufficient), which then poisoned `loss = logp +
lam*(flux-bulk)` even at λ=0 (0·NaN=NaN) → whole model went NaN and all viz
panels were white. Fix land `EstimatorConfig.mu` default → 1e-2 (conditioning
aid), added `inner_max_grad_norm` (default 0.5) per-step gradient clip on the
inner GD, forward/backward finite-bailout guards (zero field + head reset,
log `n_bailouts` in metrics), and recorded μ/clip in method block. Validation:
50/50 tests pass; transition harness clean to 400 steps with **0 bailouts**
(default config and clip-only); full `run.py` banana CPU run to step 600 (the
old death window) finite loss, `bailouts: 0`, step-500 viz non-blank (std≈87,
was 0/blank at 92–94% white in dead runs).

---

## Running log

| # | Sweep | Jobs | State |
|---|---|---|---|
| 1 | banana_variational (scheme × seeds) | 601516–601519 | ✔ completed |
| 2 | banana_var_screening (µ sweep, complete) | 601532–601535, 601572–601575 | ✔ completed (inert era; geometry only) |
| 3 | banana_var_inner (bilevel, K × seeds) | `banana_var_inner_bl` | generated, pending |
| 4 | banana_var_lambda (bilevel, λ × seeds) | `banana_var_lambda_bl` | **running** — 601640–601643 (λ=0,1e-3 × s0,1), 601646–601649 (λ=1e-2,1e-1 × s0,1; queued) |
| 5 | banana_var_bc (lam_d × seeds) | pending | – |

Sep 11 note: earlier submits 601596–601599 (1 h wall) and 601606–601609 were
cancelled (stale divergence era). The re-submits above run the stability-fix code
(40817c8); step-500 vizes verified healthy (white%≈50, std≈87 vs 92–94/0 for dead
runs) on the first four jobs.

Sweeps generated via `sweep.py`; submit 4 at a time with `sbatch run_XXXX.sh`
from the sweep dir (regenerate the dirs *on the cluster* so `REPO` paths are
correct). Locally: 50/50 tests pass.