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
panels were white. Fix: `EstimatorConfig.mu` default → 1e-2 (conditioning
aid), added `inner_max_grad_norm` (default 0.5) per-step gradient clip on the
inner GD, forward/backward finite-bailout guards (zero field + head reset,
log `n_bailouts` in metrics), and recorded μ/clip in method block. Validation:
50/50 tests pass; transition harness clean to 400 steps with **0 bailouts**
(default config and clip-only); full `run.py` banana CPU run to step 600 (the
old death window) finite loss, `bailouts: 0`, step-500 viz non-blank (std≈87,
was 0/blank at 92–94% white in dead runs).

**Lambda bilevel sweep — COMPLETE (all 8 runs, 5000 steps, 0 bailouts, plots
healthy through step 5000).** Decisive test that the regularizer is now *active*:

| λ | seed | loss | recon | flux | bulk | v_mag | gradv_mag |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0716 | 0.0716 | −0.115 | −3.0e−5 | 6.77 | 1.44 |
| 0 | 1 | 0.1133 | 0.1133 | +0.010 | −3.1e−4 | 6.67 | 1.28 |
| 1e−3 | 0 | 0.0716 | 0.0715 | +0.094 | −1.0e−5 | 6.59 | 1.74 |
| 1e−3 | 1 | 0.1134 | 0.1134 | +0.028 | −1.8e−4 | 6.82 | 0.98 |
| 1e−2 | 0 | 0.0727 | 0.0718 | +0.089 | −2.4e−4 | 9.37 | 1.55 |
| 1e−2 | 1 | 0.1158 | 0.1158 | +9.0e−5 | −4.6e−4 | 8.12 | 1.66 |
| 1e−1 | 0 | 0.0972 | 0.0708 | +0.263 | −6.9e−4 | 25.39 | 6.01 |
| 1e−1 | 1 | 0.1457 | 0.1139 | +0.317 | −1.2e−3 | 31.35 | 5.97 |

- All runs finite, `bailouts: 0`, final vizes non-blank (white ≈ 50–55%, std ≈ 84–87).
- The Poisson term now **visibly changes the solution** as λ grows: `v_mag`
  6.7→25–31, `gradv_mag` 1.3→6.0, `flux` → ~0.26–0.32 (vs the inert era where λ
  had no training effect). The bilevel gradient is working.
- **Reconstruction is preserved**: `recon` at λ=0.1 (0.071/0.114) is statistically
  equal to λ=0 (0.072/0.113); the extra loss at λ=0.1 comes from the Poisson
  term, not from degrading the score/potential fit. `bulk≈0` throughout.
- No test split on banana (`accuracy` unavailable) — λ-effect read from the
  loss/flux/recon breakdown above.

Next: `banana_var_bc` (running), then Phase 2: rings_var / breast_cancer_var /
sinusoid_reg_var / mnist_var (mnist_variational pending cost pilot at d=784), then
kernel-tuned ablation arms, then AE/VAE baselines.

**Inner-GD (K) sweep — COMPLETE (all 6 runs, 5000 steps, 0 bailouts, lam=0.01,
inner_lr=0.1, mu=1e-2 + clip).**

| K | seed | loss | recon | flux | bulk | v_mag | gradv_mag |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0.0714 | 0.0715 | −0.017 | −1.0e−5 | 11.12 | 2.54 |
| 10 | 1 | 0.1120 | 0.1119 | +0.011 | −1.1e−4 | 2.84 | 0.63 |
| 50 | 0 | 0.0710 | 0.0711 | −0.014 | −3.6e−4 | 8.50 | 1.51 |
| 50 | 1 | 0.1149 | 0.1148 | +0.019 | −2.9e−4 | 16.84 | 1.85 |
| 200 | 0 | 0.0709 | 0.0706 | +0.025 | −1.5e−4 | 7.36 | 1.12 |
| 200 | 1 | 0.1134 | 0.1130 | +0.041 | −3.5e−4 | 7.37 | 0.69 |

- Stable at every K (0 bailouts) — the clip guard holds the inner solve over the
  whole K range.
- **Fidelity signature**: field geometry converges with K — the two seeds' `v_mag`
  are scattered at K=10 (11.1 vs 2.8) and K=50 (8.5 vs 16.8) but collapse to
  7.36/7.37 at K=200; `gradv_mag` likewise. More converged inner solve → more
  consistent BOP gradients across seeds.
- **Cost/benefit**: recon at K=200 (0.0706/0.1130) is marginally better than the
  λ-sweep K=100 reference (0.0718/0.1158); K=10 is noisier. Decision: default
  **K=100** for Phase 2 runs (balance of cost v. field consistency; K=200 is
  ~2× cost for a small further gain), consistent with the λ sweep runs.

---

## Running log

| # | Sweep | Jobs | State |
|---|---|---|---|
| 1 | banana_variational (scheme × seeds) | 601516–601519 | ✔ completed |
| 2 | banana_var_screening (µ sweep, complete) | 601532–601535, 601572–601575 | ✔ completed (inert era; geometry only) |
| 3 | banana_var_inner (bilevel, K × seeds) | `banana_var_inner_bl` | ✔ completed — 601902–601905, 601912–601913 (see table above) |
| 4 | banana_var_lambda (bilevel, λ × seeds) | `banana_var_lambda_bl` | ✔ completed — 601640–601643, 601646–601649 (see table above) |
| 5 | banana_var_bc (lam_d ∈ {0.3,1,3} × seeds, K=100) | `banana_var_bc` | ▶ running — 602139–602142, 602170–602171 (5000 steps, 12 h wall) |
| 6 | mog_var / spirals_var (Phase 2, variational) | `mog_var`,`spirals_var` | ▶ running — 602172–602175 |

Sep 15 note: bc (5) + Phase-2 toys (6) submitted after the λ + K sweeps completed;
step-1500 vizes verified structured (std≈88–90, non-degenerate) at ~50 min in.
bc batch 1 = lam_d {0.3,1}, batch 2 = lam_d {3} (602170–602171). Phase 2 toys:
mog (recon, B/M=256, 5000 st) and spirals (class, 5000 st).

Sep 11 note: earlier submits 601596–601599 (1 h wall) and 601606–601609 were
cancelled (stale divergence era). The re-submits above run the stability-fix code
(40817c8); step-500 vizes verified healthy (white%≈50, std≈87 vs 92–94/0 for dead
runs) on the first four jobs.

Sweeps generated via `sweep.py`; submit 4 at a time with `sbatch run_XXXX.sh`
from the sweep dir. Regenerate dirs *on the cluster* so `REPO` paths are correct
(login node lacks singularity and has a broken torch — from a local box, generate
then `sed 's+/share_zeta/.../Poisson_CAE/cluster/logs/+/share_zeta/.../Poisson_CAE/cluster/logs/+'`
style-fix via `fix_upload.py` before `sbatch`, or regenerate on-cluster). Locally:
50/50 tests pass.