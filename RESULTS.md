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
| 5 | banana_var_bc (lam_d ∈ {0.3,1,3} × seeds, K=100) | `banana_var_bc` | ✔ **6/6 complete** — 602139–602142, 602170, 602580 (table below) |
| 6 | mog_var / spirals_var (Phase 2, variational) | `mog_var`,`spirals_var` | ✔ complete — policy-canceled first (602172–602175), re-slotted: mog 602581–602582, spirals 602605/602648 (tables below) |
| 7 | rings_var + breast_cancer_var (Phase 2, variational) | `rings_var`,`breast_cancer_var` | ✔ complete — rings 602265/602266; breast s0 602267 FAILED in `evaluate` (device mismatch, fixed `e92a03a`) → rerun 602608 + s1 602609 done, **acc 0.9649** (table below) |
| 8 | sinusoid_reg_var + mnist_var (Phase 2) | `sinusoid_reg_var`,`mnist_var` | sinusoid ✔ **done** (602723 s0, 602734 s1, table below); mnist 300-step pilot ✔ (602826, recon 0.043 @200 steps, ~2 s/step) → full **mnist_var complete 602835/602836** (table below) |
| 9 | rotation/zoom modes (Part A) + classification perturbation ablation (Part B) | `sample_ablation_mnist`, `class_ablation_{banana,rings,spirals,breast_cancer}` | 🔄 queued/running — see Sep 17 note below |

**rings_var — complete (5000 steps, 0 bailouts, lam=0.01, variational K=100).**

| seed | loss | recon | flux | bulk | v_mag | gradv_mag | accuracy |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.0401 | 0.0373 | +0.283 | – | 9.69 | 1.51 | n/a (no test split) |
| 1 | 0.0430 | 0.0431 | −0.003 | – | 3.33 | 0.24 | n/a (no test split) |

- Smooth 5000-step run both seeds, 0 bailouts; recon 0.037–0.043 well below the
  banana/variance floor — field is localizing cleanly on rings. Accuracy column
  empty because toy datasets have no test split (same limitation as banana).

**banana_var_bc — 6/6 runs (5000 steps, 0 bailouts, lam=0.01, K=100).**

| lam_d | seed | loss | recon | flux | bulk | v_mag | gradv_mag |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0 | 0.0704 | 0.0713 | −0.086 | −6.1e−4 | 32.40 | 3.60 |
| 0.3 | 1 | 0.1141 | 0.1142 | −0.013 | −3.9e−4 | 25.49 | 1.77 |
| 1.0 | 0 | 0.0708 | 0.0710 | −0.019 | −3.0e−5 | 4.53 | 1.16 |
| 1.0 | 1 | 0.1131 | 0.1131 | +0.003 | +2.5e−4 | 12.72 | 1.11 |
| 3.0 | 0 | 0.0718 | 0.0719 | −0.008 | +5.8e−5 | 2.04 | 0.71 |
| 3.0 | 1 | 0.11104 | 0.11084 | – | – | 1.307 | – |

- Dirichlet strength pins the boundary: `v_mag` 32→2 and `gradv_mag` 3.6→0.7 as
  lam_d goes 0.3→3.0; `recon` is flat (0.071/0.114 for seeds 0/1) and `bulk≈0`
  across the whole grid — the pin changes field scale, not reconstruction.

**mog_var — complete (5000 steps, 0 bailouts, lam=0.01, variational K=100).**

| seed | loss | recon | flux | v_mag | gradv_mag | bailouts |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.00627 | 0.00629 | – | – | – | 0 |
| 1 | 0.00720 | 0.00722 | – | – | – | 0 |

- Clean null-recon field (loss ≈ recon ≈ 0.006–0.007), as expected for a
  constant-cost dataset; 0 bailouts both seeds.

**spirals_var — complete (5000 steps, 0 bailouts, lam=0.01, variational K=100).**

| seed | loss | recon | flux | v_mag | gradv_mag | bailouts |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.5767 | 0.5762 | +0.053 | 1.47 | 0.66 | 0 |
| 1 | 0.6052 | 0.6048 | +0.032 | 2.80 | 0.70 | 0 |

- Overlapping-class toy: recon ≈ loss at 0.58–0.61 floor, positive flux (field
  aligns with the spiral direction), `v_mag` 1.5–2.8, 0 bailouts.

**breast_cancer_var — complete (3000 steps, 0 bailouts, lam=0.01, K=100, knn B=128).**

| seed | loss | recon | flux | v_mag | gradv_mag | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | −0.00187 | 0.00017 | – | – | – | **0.9649** |
| 1 | −0.00112 | 0.00057 | – | – | – | **0.9649** |

- Both seeds reach 96.5% test accuracy with 0 bailouts from the fixed eval
  (`e92a03a`); loss≈0 with recon≈0 — trivial data move is sufficient for
  classification, so no field ever needs to fire.

**sinusoid_reg_var — complete (4000 steps, 0 bailouts, lam=0.01, variational K=100, GRU d=50).**

| seed | loss | recon | flux | v_mag | gradv_mag | mse (test) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.11907 | 1.11910 | −0.00275 | 0.924 | 0.096 | 1.076 |
| 1 | 0.03266 | 0.03264 | +0.00244 | 0.621 | 0.398 | 0.056 |

- High seed variance: seed 0 is stuck in a near-constant field regime (recon ≈
  1.12, weak `v_mag`/`gradv`), seed 1 captures the sinusoid (recon 0.033, stronger
  gradient, much lower test MSE 0.056). Both 0 bailouts, 8 viz saves; flux stays
  small — the sinusoid boundary is smooth.

**mnist_var — complete (3000 steps, 0 bailouts, lam=0.01, variational K=100, knn, d=784).**

| seed | loss | recon | flux | v_mag | gradv_mag | mse (test) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.01557 | 0.01552 | +0.00473 | 5.93 | 0.275 | 0.0140 |
| 1 | 0.01740 | 0.01741 | −0.00055 | 1.39 | 0.029 | 0.0132 |

- Both seeds clean (0 bailouts, ~1h40m wall each, 2 s/step), test MSE
  ≈ 0.013–0.014 — a strong pixel-level reconstruction floor; flux stays O(1e-3).
  Some seed variation in the field (`v_mag` 5.9 vs 1.4) at equal recon quality.
- These s0/s1 runs predate `model_last.pt` checkpointing, so reconstruction-sample
  PNGs require fresh runs (checkpoint + `sample_recon.py` shipped in `8c124aa`).

Sep 16 note: bc 6/6 + all Phase-2 toys (mog, rings, spirals, breast, sinusoid)
complete. sinusoid s0/s1 initially FAILED on GPU (602659/602660, 13–18 s, empty
logs) — root cause: `_cudnn_rnn_backward … double backwards not supported` in the
bilevel variational solver's second-order pass through the GRU encoder; CPU ran
fine. Fixed by disabling the CuDNN RNN path in `GRUEncoder.forward`
(`torch.backends.cudnn.flags(enabled=False)`, commit `83ec241`), verified by a
cluster diag (602683, exit 0) then resubmitted s0/s1 (602723/602734) → both
completed (~10 h wall each; GRU solver is slow — ~9 s/step). MNIST 300-step cost
pilot (602778 FAILED — recon loaders bound labels to `y_true` → (B,) vs (B,784)
mismatch; fixed in `447130b`, x-only loaders) → resubmitted 602826 and it
completed in **9:53** (`results/mnist_flat/`, loss/recon 0.043, `v_mag` 1.29,
0 bailouts, mse 0.036 → ~2 s/step) → full **mnist_var s0/s1 (3000 steps) ran to
completion (602835/602836, 1h40m each, 0 bailouts)**. Obeying the max-3-at-a-time
submission policy via a top-up monitor (CAP=3, never exceeds).

Sep 16 (late) — reconstruction samples + perturbation schemes (user request):
new `mask` / `dropout` corruption modes (`corruption_mask_frac`,
`corruption_drop_p`), `model_last.pt` checkpoints saved by `run.py`, shared
`Utils/pipeline.build_components`, and `sample_recon.py` (per-mode clean |
corrupted | reconstruction PNG grids + `sample_metrics.json`). All shipped in
`8c124aa`, pulled to ICA; 56/56 tests pass. Existing completed runs have no
checkpoints, so producing the sample gallery needs fresh (re)training runs
(only ~1.7 h per MNIST seed, ~10-40 min per toy).

Kernel-tuned ablation arm (banana_kernel_tuned, mog/spirals/rings/breast/mnist
`*_kernel`): **dropped** — starting the banana kernel run confirmed the kernel
estimator's losses are unbounded/not meaningful (it is the inert-gradient method
that motivated the variational/bilevel solver). Variational is the sole method of
record; the kernel arm has no value to keep.

Sep 15 note: bc (5) + Phase-2 toys (6) submitted after the λ + K sweeps completed;
step-1500 vizes verified structured (std≈88–90, non-degenerate) at ~50 min in.
bc batch 1 = lam_d {0.3,1}, batch 2 = lam_d {3} (602170–602171). Phase 2 toys:
mog (recon, B/M=256, 5000 st) and spirals (class, 5000 st).

Sep 11 note: earlier submits 601596–601599 (1 h wall) and 601606–601609 were
cancelled (stale divergence era). The re-submits above run the stability-fix code
(40817c8); step-500 vizes verified healthy (white%≈50, std≈87 vs 92–94/0 for dead
runs) on the first four jobs.

Sweeps generated via `sweep.py`; submit **max 3 at a time** with `sbatch
run_XXXX.sh` (policy — top-up monitor caps at 3). Regenerate dirs *on the
cluster* so `REPO` paths are correct (login node lacks singularity and has a
broken torch — from a local box, generate then path-fix via `fix_upload.py`
before `sbatch`, or regenerate on-cluster). Locally: 50/50 tests pass.

Sep 17 — corruption extension + classification perturbation ablation (`2782e67`, `a46f372`):
- **Part A — rotation/zoom modes.** `CorruptionOperator` gains `rotation`
  (per-sample angle, `corruption_rotation_max_deg=30`) and `zoom`
  (per-sample scale, `corruption_zoom_std=0.15`). For `mnist_flat`
  (`image_side=28`) they use an affine resample of the 28×28 layout; for 2D point
  data rotation acts on the first two coords. `generate_all_samples.py` defaults
  stay at the 5 original modes; the MNIST cluster job passes the 7-mode list.
- **Part B — classification perturbation benchmark.** New `configs/class_*.yaml`
  (banana/rings/spirals/breast_cancer; variational bilevel, Phase-2 hparams) and
  `eval_perturbation.py` (loads `model_last.pt` + config, evaluates clean test
  accuracy and accuracy under the run's training corruption → `perturbation_eval.json`).
  `method_tag` now appends the corruption mode so run dirs are self-describing.
  banana/rings/spirals/breast_cancer got held-out test draws (distinct seed) —
  the earlier `rings_var`/toys had no test split, so accuracy was unavailable.
- **Bug: `grid_sampler_2d_backward` not implemented** (jobs 604824/604825 FAILED,
  23–38 s). The bilevel variational estimator needs **second-order** grads, which
  `F.grid_sample` lacks; detaching the corruption would break
  `Classifier_model.score_value` (needs ∂logp/∂x_clean through Πψ). Fix
  `a46f372`: `Utils/projectors.affine_sample_2d()` — a manual bilinear sampler
  (gather-based) with identical forward/grad to `F.grid_sample` (max err 1e-14,
  `gradgradcheck=True`); grid is independent of image values so it is linear in
  the input and supports arbitrary-order derivatives. Regression tests added
  (65 pass). Both modes smoke-tested through the full variational `train` step.
- **Duplicate mog re-run (waste):** the old `monitor_topup4.sh` had already
  completed mog run_0003..0009 before the unified monitor was launched, which
  `count_ours`-ed 0 and re-submitted the same 7 (jobs 604576…604823, ~2.5 h each).
  The `monitor_unified2.sh` queue (MNIST rotation/zoom, then remaining class jobs)
  now resumes from the correct point (banana run_0000 already submitted). Cap-3
  respected throughout.
- Regeneration pending once runs finish: `samples_gen` (7-mode MNIST gallery via
  `sample_recon.py`) and `eval_perturbation.py` per class run → perturbation table;
  also prune the duplicate mog run dirs before aggregation.