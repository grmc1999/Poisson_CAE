# Poisson-CAE → ICLR 2027 Main Track: Research & Execution Plan

**Goal:** Extend "Poisson-Induced Potentials for Contractive Representations" (GRaM workshop @ ICLR 2026,
Tiny Paper) into a main-track submission.

**Deadlines:** Abstract **Sep 18, 2026** · Full paper **Sep 25, 2026** (AoE)
**Compute:** Cluster only (SDumont/ICA via SLURM). Local machine = code edits, CPU smoke tests, plotting.
**Method decisions:** Diffusion (heat) kernel replaces the high-$d$ Poisson kernel
(localized solvers L1 + L2 prune the diffusion quadrature) · Potential stays in input
space · Variational (Ritz) solver as an ablation alongside the diffusion kernel.

---

## Phase 0 — Cleanup & infrastructure (Aug 24–30)

- [x] Move broken/stale entry points to `legacy/`: `main_ori.py`, `main_MNIST.py`,
      `main_flat_mnist.py`, `main_conv_mnist.py` (reference deleted classes/APIs;
      recoverable from git history on `images` branch)
- [x] Add `README.md`: theory recap (Poisson reformulation of CAE penalty, Green's identity
      decomposition), repo layout, usage
- [x] Add `requirements.txt` (pinned torch/numpy/scikit-learn/matplotlib) + Python version
- [x] Config-driven runs: YAML per experiment, fixed seeds; every run writes config + metrics
      to `results/<exp>/<timestamp>/` (JSON/CSV)
- [x] CPU unit tests (`pytest`):
  - `green_reg` / `gradx_green_reg` vs finite differences
  - MC Green estimator vs analytic Poisson solution (e.g., Gaussian blob source)
  - Jacobian Frobenius norm correctness
- [x] Merge validated `image_branch` → `main`

## Phase 1 — Diffusion-kernel potential estimators (Aug 31–Sep 6)

**Decision (Sep 2):** the high-d Laplacian fundamental solution
$G(r)\propto r^{2-d}$ is numerically dead for large $d$: at $d=784$ the sphere
constant $\omega_d$ overflows float64 and the gradient $r^{-d}$ under/overflows to
zero/nan. **Replace the Poisson kernel with the diffusion (heat) kernel**
$G_t(x,y)=(4\pi t)^{-d/2}\,e^{-\lVert x-y\rVert^2/4t}$, whose gradients carry no
$r^{2-d}$ pole and remain finite at any $d$.

New module `Utils/localized_estimators.py`, drop-in compatible with `PoissonMCEstimator`
interface (`forward(x_query, x_land, g_land) -> (v_hat, gradv_hat)`, differentiable).
L1/L2 locality remains the way to *prune* quadrature; the diffusion kernel is what
*makes gradients finite in high-d*.

- [x] **Same L1 / L2 quadrature machinery** (compact-support gather, kNN gather)
      applied to the *diffusion* kernel instead of `r^{2-d}`
- [x] Exploit locality *for free*: $G_t$ decays as $e^{-\lVert x-y\rVert^2/4t}$, so
      nearest landmarks dominate; ball/kNN truncation is statistically justified
- [x] Verify `BC_loss` / `D_loss` work unchanged with both estimators
- [x] Numerics of the prefactor $(4\pi t)^{-d/2}$: store/compute in log space and
      fold the global scale into `λ` (it scales all queries equally)
- [ ] Toy validation: field error maps (localized vs dense diffusion);
      wall-clock/memory scaling vs d ∈ {2…784}, M, B, t
- [x] **Ablation — variational (Ritz) Poisson solver**: neural-field $v_\theta$
      with Dirichlet BC, online inner GD, compared against diffusion-kernel fields
      (§ below)

### Phase 1 — complete mathematical definitions (L1 + L2, diffusion kernel)

**Setup.** Let $f : \mathbb{R}^d \to \mathbb{R}^m$ be the encoder. The source (data
term) is the squared encoder Jacobian norm, and the contractive-penalty potential
$v$ is a smoothed (diffused) version of this source on $\Omega \subseteq \mathbb{R}^d$:

$$
s(y) = \lVert J_f(y)\rVert_F^2 = \sum_{k=1}^{m}\lVert \nabla_y f_k(y)\rVert_2^2
\qquad \text{(source / bulk score)}
$$

---

**Diffusion kernel $G_t$ (the chosen potential kernel).** Let
$t > 0$ be a diffusion scale. The heat/diffusion Green's function on $\mathbb{R}^d$ is

$$
G_t(x,y) \;=\; \frac{1}{(4\pi t)^{d/2}}\ \exp\!\Bigl(-\frac{\lVert x-y\rVert^2}{4t}\Bigr)
\qquad\text{(density of }\mathcal{N}(x,\,2t\,I)\text{)}
$$

and its input-space gradient is

$$
\nabla_x G_t(x,y) \;=\; -\frac{x-y}{2t}\ G_t(x,y)
\qquad\Longrightarrow\qquad
\bigl\lVert \nabla_x G_t(x,y)\bigr\rVert_2 = \frac{\lVert x-y\rVert}{2t}\, G_t(x,y).
$$

The potential and its gradient are

$$
v(x) = \int_\Omega G_t(x,y)\, s(y)\, dy, \qquad
\nabla v(x) = \int_\Omega \nabla_x G_t(x,y)\, s(y)\, dy
= -\frac{1}{2t}\int_\Omega (x-y)\, G_t(x,y)\, s(y)\, dy.
$$

**Why this fixes the high-integer-exponent problem.** The Poisson fundamental
solution $G(r)\propto r^{2-d}$ has exponent $2-d$: at $d=784$ it is
$r^{-782}$, which (i) under/overflows float64 for $r\ne 1$, and (ii) requires the
unit-sphere area $\omega_d = 2\pi^{d/2}/\Gamma(d/2)$, which itself overflows to
$\infty$ for $d \gtrsim 300$. Neither issue affects $G_t$: its only $d$-dependence
is the *scalar* prefactor $(4\pi t)^{-d/2}$, and its gradient is just
$-\frac{x-y}{2t}G_t$ with **no** radius-to-a-power term. The Gaussian factor
$e^{-\lVert x-y\rVert^2/4t}$ is always $\in [0,1]$ and finite in any dimension.

**Numerics of the prefactor (important).** For large $d$, $(4\pi t)^{-d/2}$ is
tiny for $t\gtrsim 1$ (or huge for tiny $t$). Because it is a *global scalar*, it
applies identically to every query and only scales the field; we therefore:

- evaluate everything in **log space** and subtract the max, i.e. compute
  $\tilde G_t = \exp\bigl(-\tfrac{d}{2}\log(4\pi t)-\tfrac{\lVert x-y\rVert^2}{4t} - c\bigr)$
  for a shift $c$, so no under/overflow occurs; and
- fold the global scale factor into the regularizer weight $\lambda$ (it does not
  change gradients' *directions*, only the common magnitude).

In practice the *relative* weighting between landmarks,
$e^{-\lVert x-y\rVert^2/4t}$, is what controls the field shape, and it is tame.

**Locality / why L1–L2 are now natural and cheap.** Because $G_t$ decays like a
Gaussian in $\lVert x-y\rVert$, the nearest landmarks dominate the integral; the
L1 (Wendland/compact-support gather) and L2 (kNN gather) estimators from before
remain unchanged in structure — they now *prune* the diffusion quadrature, which
is statistically justified by the kernel's own decay.

---

**Rejected baseline — the high-d Poisson kernel (kept for motivation in §4).**
The original formulation used the Laplacian (Poisson) fundamental solution $G_\varepsilon$,
which for $d \ge 3$ is $G\propto r^{2-d}$ with the unit-sphere constant $\omega_d$:

$$
\begin{aligned}
d = 2:\quad
& G_\varepsilon(x,y) = -\tfrac{1}{2\pi}\log r,
\; & \nabla_x G_\varepsilon(x,y) &= -\tfrac{1}{2\pi}\frac{x-y}{\lVert x-y\rVert^2 + \varepsilon^2} \\[2mm]
d \ge 3:\quad
& G_\varepsilon(x,y) = c_d\, r^{\,2-d},
\; & \nabla_x G_\varepsilon(x,y) &= c_d\,(2-d)\,(x-y)\, r^{-d}
\end{aligned}
\qquad
r = \sqrt{\lVert x-y\rVert^2 + \varepsilon^2}
$$
$$
c_d = \frac{1}{(d-2)\,\omega_d}, \qquad
\omega_d = \frac{2\,\pi^{d/2}}{\Gamma(d/2)}.
$$

**Why this is rejected for $d\gg 3$ (two independent failures):**
1. **$\omega_d$ overflows.** $\omega_d = 2\pi^{d/2}/\Gamma(d/2)$ grows super-exponentially;
   $\Gamma(392)$ exceeds float64 max, so $\omega_d \to \infty$ and hence $c_d \to 0$
   for $d \gtrsim 300$. The constant itself is not representable.
2. **$r^{-d}$ under/overflows.** The gradient $\propto r^{-d} = r^{-784}$ at $d=784$
   is $10^{-236}$ for a single unit gap and diverges as $r\to 0$; the field is
   numerically empty or divergent for any feasible radius.

The diffusion kernel $G_t$ (above) avoids both because its $d$-dependence is only a
scalar prefactor and its gradient carries no radius-to-a-power.

**Monte-Carlo quadrature (dense).** Given a batch of $B$ queries and $M$ landmark
samples $\{(y_j, s(y_j))\}_{j=1}^{M}$ (`x_land`, `g_land`), the dense diffusion estimate is

$$
\hat{v}(x_i) = \frac{1}{M}\sum_{j=1}^{M} G_t(x_i, y_j)\, s(y_j),\qquad
\nabla\hat{v}(x_i) = \frac{1}{M}\sum_{j=1}^{M} \nabla_x G_t(x_i, y_j)\, s(y_j).
$$

The gradient w.r.t the *corrupted* query points `x_tilde` (not the landmarks) is what
`D_loss`/`BC_loss` consume. This is the dense, global estimator — O(B·M·d) memory.

---

**L1 — Compact-support (ball-truncated) diffusion quadrature.**

A Wendland C² window $w$ (compactly supported on the ball of radius $R$):

$$
w(r) = \bigl(1 - r/R\bigr)_{+}^{4}\,\bigl(1 + 4\,r/R\bigr), \quad r \le R; \qquad w(r) = 0, \quad r > R
$$
$$
w(0) = 1, \quad w(R) = 0, \quad w'(R) = w''(R) = 0 \qquad \text{(C² continuous everywhere)}
$$

Because $G_t$ itself decays like a Gaussian, the window is a *soft outer truncation*:
it drops landmarks beyond a ball, and inside the ball the natural
$e^{-\lVert x-y\rVert^2/4t}$ Gaussian decay already dominates the integral, so
truncation adds negligible bias at the tail. The window keeps the interaction region
strictly bounded and differentiable through $r=R$.

Quadrature restricts to the local ball $N_R(x_i) = \{\, y : \lVert x_i-y\rVert \le R \,\}$:

$$
\hat{v}(x_i) = \frac{1}{M_i} \sum_{j\,:\; \lVert x_i-y_j\rVert \le R}
w\!\bigl(\lVert x_i-y_j\rVert/R\bigr)\, G_t(x_i, y_j)\, s(y_j)
$$
$$
\nabla\hat{v}(x_i) = \frac{1}{M_i} \sum_{j\,:\; \lVert x_i-y_j\rVert \le R}
w\!\bigl(\lVert x_i-y_j\rVert/R\bigr)\, \nabla_x G_t(x_i, y_j)\, s(y_j)
$$
$$
M_i = \#\{\, j : \lVert x_i-y_j\rVert \le R \,\} \qquad \text{(active neighbours of } x_i; \ge 1 \text{ required)}
$$

Implementation (`CompactSupportEstimator`): per query, gather the `n` nearest
landmarks (`n = max_neighbors`, a memory knob), then multiply by the window `w`,
which is exactly 0 beyond R. So the *gathered* block is bounded (O(B·n·d) memory)
yet the *active* ball is still exactly enforced by `w`. The source values `s(y_j)`
are gathered with the same indices so they follow their landmarks.

- `normalize_by_cardinality=True`: divide by $M_i$ (local mean; the estimator is a
  local average of $G_t\cdot$source). Default.
- `normalize_by_cardinality=False`: divide by the total $M$ (global-style quadrature
  weights; matches the base estimator in the $R \to \infty$ limit).

Gradient path: $\nabla\hat{v}$ w.r.t $x_i$ flows through both $\nabla_x G_t$
(kernel) and $w$ (window on $\lVert x_i-y_j\rVert$); w.r.t $y_j$ / $s(y_j)$ through the
gathered source values.

---

**L2 — k-nearest-neighbour quadrature.**

Keep only the $k$ nearest landmarks per query, no radial window:

$$
\hat{v}(x_i) = \frac{1}{N_i} \sum_{j \in N_k(x_i)} G_t(x_i, y_j)\, s(y_j)
$$
$$
\nabla\hat{v}(x_i) = \frac{1}{N_i} \sum_{j \in N_k(x_i)} \nabla_x G_t(x_i, y_j)\, s(y_j)
$$
$$
N_k(x_i) = \text{the } k \text{ landmarks minimizing } \lVert x_i-y_j\rVert
\qquad (N_i = k, \text{ or } M \text{ if } M < k)
$$

Implementation (`KNNEstimator`): exact `torch.topk` on the `(B, M)` distance matrix
(`largest=False`), then gather landmark coords and their source values by the same
indices → memory O(B·k·d).

- `normalize_by_k=True`: divide by $k$ (local mean). Default.
- `normalize_by_k=False`: divide by total $M$; then recovering the global estimator
  requires $k \to M$.

**Differentiability / autodiff contract.** Both estimators return tensors that keep
the graph to `x_query` and `g_land` (through `torch.gather` and the kernel eval), so
`gradv_hat` can backprop into `x_tilde` exactly as the dense estimator does — this is
what keeps `Poisson_reg.D_loss` ($\nabla\!\cdot\!(\text{score}\cdot\nabla\hat v)$) and
`BC_loss` (normal flux on the corruption manifold) unchanged.

---

**Error decomposition (paper §4).** For a query $x$ (dropping the index), the potential
is $v(x) = \int_\Omega G_t(x,y)\, s(y)\, dy$, i.e. the $t$-smoothed source. The estimator
$\hat{v}$ over the active set $N(x) \subseteq \{1,\dots,M\}$ with weights $w_j$ differs
from this target through three sources:

$$

\begin{aligned}
\mathbb{E}[\,\hat{v}\,] - v
&= \mathbb{E}\!\Bigl[\,\tfrac{1}{\sum_j w_j}\sum_{j \in N(x)} w_j\, G_t(\cdot,y_j)\,s(y_j)\Bigr]
- \int_\Omega G_t(\cdot,y)\, s(y)\, dy \\
&= \underbrace{\mathrm{quadrature\_error}(M, N(x))}_{\text{finite landmark sampling}}
+ \underbrace{\mathrm{truncation\_bias}(x,\, R \text{ or } k)}_{\text{Gaussian tail left out of } N(x)}
+ \underbrace{\mathrm{smoothing\_bias}(t)}_{v_t \text{ is } t\text{-smoothed } s}
\end{aligned}
$$

- **Quadrature error**: variance/LLN over the $M$ MC landmarks; shrinks as $M$ grows.
- **Truncation bias**: controlled by the *Gaussian tail*,
  $\mathrm{bias}_{R}(x) = \int_{\Omega \setminus N_R(x)} G_t(x,y)\, s(y)\, dy$, with the bound
  $\lvert \mathrm{bias}_{R}\rvert \le \lVert s\rVert_\infty \int_{r > R} \frac{e^{-r^2/4t}}{(4\pi t)^{d/2}}\, dr$,
  which decays **exponentially fast** in $R^2/(4t)$ in any dimension (no $r^{2-d}$ tail);
  L2 replaces $N_R$ by $N_k(x)$ (the $k$ nearest) with the same exponential-in-distance
  decay. This is the conceptual payoff of the diffusion choice: truncation is cheap and
  provably mild in high $d$.
- **Smoothing (modeling) bias**: $v = G_t \ast s$ is the heat-smoothed source; it removes
  high-frequency content of $s$ at scale $\sqrt{t}$. This is intrinsic to the model (the
  potential is smooth) and is *not* reduced by increasing $M$; it is the analogue of the
  removed $\varepsilon$-singularity in the Poisson case, but with a well-defined, tunable
  scale $t$ instead of an ad-hoc clip.

**Consistency in the *quadrature* sense.** As $R \to \infty$ (L1: $w \to 1$, active set
$\to$ all $M$) or $k \to M$ (L2: active set $\to$ all $M$),

$$
\mathbb{E}[\,\hat{v}\,] \;\to\; \frac{1}{M}\sum_{j=1}^{M} G_t(x_i, y_j)\, s(y_j),
$$
the dense diffusion-MC estimator. Verified by `tests/test_localized.py`.

**Consistency in the *model* sense.** As $t \to 0$, $G_t \to \delta$, so
$v_t(x) \to s(x)$: the smoothed potential recovers the unsmoothed source, exposing the
full $\lVert J_f\rVert_F^2$ field. This gives a principled "tuning knob" connecting the
manifestly-contractive (small $t$) and smooth-surrogate (large $t$) regimes.

**Cost–locality trade-off (what the scaling experiment measures).**

$$
\begin{array}{l|ccc}
\mathbf{Base/dense} & \mathrm{memory}\ O(B\,M\,d) & \mathrm{flops}\ O(B\,M\,d) & \mathrm{kernel\ evals}\ 2BM \\[1mm]
\mathbf{L1} & O(B\,n\,d) & - & \mathrm{evals}\ 2Bn,\ \ \mathrm{active}\ \sim B\rho_R \qquad (n \le M) \\[1mm]
\mathbf{L2} & O(B\,k\,d) & - & \mathrm{evals}\ 2Bk,\ \ \mathrm{active}\ = Bk \qquad (k \le M)
\end{array}
$$
where $\rho_R$ is the average ball occupancy at radius $R$. The remaining dense cost is
the $(B, M)$ distance matrix for the top-k search itself — $O(B\,M)$ in $B\cdot M$,
independent of $d$ — which we keep for exactness and profile for a later sparse / kd-tree
upgrade if needed. Field error maps compare $\hat{v}_{\mathrm{loc}}$ vs
$\hat{v}_{\mathrm{global}}$; scaling compares wall-clock & peak memory vs
$(d, M, B, R, k, t)$.

**Theory deliverables (paper §4):**
- Error decomposition: quadrature error + Gaussian truncation bias + smoothing bias $t$
- Numerical form of $G_t$: log-space evaluation, folding the global scale into $\lambda$
- Consistency: $R \to \infty$ / $k \to M$ recover the dense estimator; $t \to 0$ recovers the raw $\lVert J_f\rVert_F^2$ source
- Cost–locality trade-off analysis; role of $t$ as the natural interaction scale

---

## Phase 1 Ablation — Variational (Ritz) Poisson solver

**Decision (Sep 2):** add the variational/energy minimization approach as an
*ablation alongside* the diffusion-kernel estimator (not a replacement). The
variational solver produces $v$ and $\nabla v$ as a *learned neural field* rather
than a kernel sum, which has the advantage of no kernel prefactor / no radius-to-a-
power at all (neither $r^{2-d}$ nor $(4\pi t)^{-d/2}$); this lets us empirically
compare the two families in controlled settings and study how the potential's shape
depends on the solver.

### A.1 The Ritz (energy-minimization) formulation

We want a scalar field $v : \Omega \to \mathbb{R}$ satisfying (in the weak sense) the
screened Poisson BVP

$$
-\Delta v + \mu v = s \quad \text{on } \Omega, \qquad v|_{\partial\Omega} = 0
\qquad (\text{Dirichlet BC}).
$$

where $\mu \ge 0$ is a screening parameter (for pure Poisson set $\mu = 0$; the
screened form $\mu > 0$ makes the problem coercive even without a geometric boundary).

The corresponding Ritz energy functional is

$$
J(v) = \underbrace{\tfrac{1}{2}\int_\Omega \lVert\nabla v\rVert^2}_{\text{Dirichlet / elastic energy}}
+ \underbrace{\tfrac{\mu}{2}\int_\Omega v^2}_{\text{screening / mass penalty}}
- \underbrace{\int_\Omega s(x)\, v(x)\, dx}_{\text{source coupling}}.
$$

Its minimizer (with $v = 0$ on $\partial\Omega$) is the unique weak solution of the
BVP. This avoids the kernel entirely: $v$ is found directly.

### A.2 Discretized energy (MC over collocation points)

Given the landmark/collocation set $\{x_j\}_{j=1}^{M} \subseteq \Omega$ with
sources $s_j = \lVert J_f(x_j)\rVert_F^2$, and a candidate field $v_\theta$
parameterized by a small neural network $\theta$, the MC-estimated energy is

$$
\mathcal{L}_V(\theta) =
\frac{1}{2M}\sum_{j=1}^{M}\bigl\lVert\nabla_x v_\theta(x_j)\bigr\rVert^2
+ \frac{\mu}{2M}\sum_{j=1}^{M} v_\theta(x_j)^2
- \frac{1}{M}\sum_{j=1}^{M} s_j\; v_\theta(x_j)
$$

with a soft Dirichlet penalty on boundary collocation points (see §A.4).

The energy involves only **first derivatives** $\nabla_x v_\theta$; no second
derivatives or Hessians are needed. This is the key computational advantage over
the residual-minimization ($L_2(-\Delta v + \mu v - s)^2$) alternative, which
requires Hessians of the neural field.

### A.3 Online inner solve (coupled to the encoder training)

At each outer training step, the source $s$ depends on the encoder $f$ being
jointly trained. The inner solve proceeds as follows:

1. **Freeze** $f$ (and hence all $s_j$) and draw the landmark/source batch.
2. Take $K$ gradient-descent steps on $\theta$ to approximately minimize
   $\mathcal{L}_V$ (the energy is convex *in $v$* for linear $\theta$-spaces; for
   nonlinear nets, $K$ small steps suffice).
3. **Evaluate** the approximately-optimal $v_\theta^*$ at the query points
   $\tilde x_i$ to obtain $v_\theta^*(\tilde x_i)$ and
   $\nabla_x v_\theta^*(\tilde x_i)$.
4. These outputs are passed unchanged to $\texttt{D\_loss}$ and $\texttt{BC\_loss}$.

**Stop-gradient convention.** The encoder's own gradient (w.r.t. the outer loss)
is **not** allowed to flow through the inner $\theta$ parameters — $\theta$ is
treated as an inner variable. Only the dependence of $v_\theta^*(x)$ on the *query
point* $x = \tilde x_i$ carries the outer gradient. Formally, the outer loss is
differentiated w.r.t. $\tilde x$ while $\theta$ is held fixed at the result of the
inner solve. This yields a clean bilevel structure and avoids complicated
second-order gradient-through-optimization machinery.

### A.4 Enforcing Dirichlet: boundary collocation on OOD points

In high dimension the geometric boundary $\partial\Omega$ is hard to sample. We
therefore use the **corrupted (OOD) points** $\tilde x = \Pi_\psi(x)$ as boundary
anchors: these are by construction *off-manifold*, lying where the clean data are
not, and therefore where the potential should be small (or zero) by the Dirichlet
condition. The soft penalty is

$$
\mathcal{L}_{\mathrm{BC}} = \frac{\lambda_\partial}{|\tilde{X}|}\sum_{i} v_\theta(\tilde x_i)^2,
\qquad \tilde x_i \in \tilde{X}
$$

and the total inner energy is $\mathcal{L}_V + \mathcal{L}_{\mathrm{BC}}$. The
hyperparameter $\lambda_\partial > 0$ controls how strongly the Dirichlet condition
is enforced at the OOD anchors.

### A.5 Interface contract (same as kernel estimators)

The `forward(x_query, x_land, g_land) -> (v, gradv)` interface is identical:

- **Input**: `x_query` $= \tilde x$ (corrupted batch), `x_land` $= \{x_j\}$
  (corrupted landmarks, reused as collocation), `g_land` $= \{s_j\}$.
- **Output**: `v` $= v_\theta^*(\tilde x) \in \mathbb{R}^{B}$,
  `gradv` $= \nabla_x v_\theta^*(\tilde x) \in \mathbb{R}^{B \times d}$.
- The `gradv` tensor is connected by autodiff to $\tilde x$ so that
  $\texttt{D\_loss}$ and $\texttt{BC\_loss}$ can backpropagate through it into the
  encoder exactly as before.

### A.6 Why this complements the diffusion kernel (positioning)

| Aspect | Diffusion kernel ($G_t$) | Variational ($v_\theta$) |
|---|---|---|
| **Kernel / prefactor** | $(4\pi t)^{-d/2}$ prefactor; needs log-space + $\lambda$-absorb | No kernel or prefactor at all |
| **Gradient cost** | Quadrature sum over $M$ gathered terms | One autodiff pass on $v_\theta$ |
| **Smoothness** | Controlled by $t$ (fixed) | Controlled by $v_\theta$ architecture + BC |
| **Boundary** | Implicit (free-space kernel) | Explicit Dirichlet via OOD collocation |
| **Inner solve** | None (closed-form kernel) | $K$ gradient-descent steps per outer step |
| **Edge** | Fast, analytical, no trainables for $v$ | No prefactor, no $d$-dependent scaling |

The two are complementary. The variational solver is the right *ablation* because
it reveals what the kernel's Gaussian smoothing actually does to the field shape:
any discrepancy between the two at small $t$ is attributable to the kernel's
fixed smoothing scale, while the variational field can in principle adapt freely.
A quantitative comparison (field error, wall-clock, gradient quality) will be
reported in §5 of the paper.

### A.7 Validation plan (for Phase 2 experiments)

- **Sanity check (2D):** with $K = 0$ inner steps and a linear $v_\theta$, the
  gradient of $\mathcal{L}_V$ w.r.t. $\theta$ is exactly the Galerkin residual of
  the screened Poisson BVP — verify it converges to the analytic solution on a
  small test problem.
- **Consistency against diffusion:** compare the variational $v$ field (large $K$,
  sufficient convergence) against the diffusion-kernel $v_t$ at matched $\mu$ and
  $t$ — quantify the residual.
- **Dimension scaling:** vary $d \in \{2, 10, 30, 100, 784\}$; report $v$-field
  quality, wall-clock, and memory as a function of $d$, $K$, $M$.
- **Hyperparameter sensitivity:** sweep $\mu$, $\lambda_\partial$, $K$, inner
  learning rate; plot convergence of $\mathcal{L}_V$ vs $K$.

---

## Phase 2 — Experiments on cluster (Sep 7–17)

- [ ] SLURM job templates + sweep runner committed to repo; results synced back via git/tarball
- [ ] Experiment ladder:
  - Toys: mog / spirals / banana / rings (2D)
  - Tabular: breast cancer (30D classification)
  - Time series: sinusoid regression (50D)
  - Images: MNIST flat + conv (input-space potential)
- [ ] Baselines: AE, CAE, DAE, VAE
- [ ] Ablations: λ, locality radius R, neighbors k, global-vs-localized estimator,
      corruption mode (gaussian / ddpm / shift_scale),
      diffusion-kernel vs variational (Ritz) potential solver
- [ ] Metrics: task accuracy/MSE under corruption, linear-probe representation quality,
      ‖J_f‖ control, compute scaling curves

## Phase 3 — Paper (Sep 10–25, overlapping)

- [ ] ICLR 2027 LaTeX skeleton in `paper/` from day one; figures auto-regenerated by script
- [ ] Sep 18: genuine abstract submitted on OpenReview
- [ ] Sep 24–25: internal review pass → full paper submission

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Single author, 4 weeks | Scope frozen at MNIST; CIFAR-10 only if ahead of schedule |
| Cluster queue delays | Submit sweeps early; jobs ≤ a few hours; prioritize paper-critical ablations |
| Localization underdelivers empirically | Fallback venue ICML 2027 (Jan 2027 deadline), expanded theory |

## Context snapshot (as of Aug 24, 2026)

- Branch `image_branch` (clean, synced): current working code — `main.py` unified entry
  (recon/classification/regression tasks), `models.py` (`Poisson_reg`, AE/Classifier/
  Regressor/GRU heads), `Utils/` (Green-MC estimator, corruption operators, datasets, viz)
- Workshop paper: qualitative only (field visualizations on toys + image data)
- Outputs: `outputs_banana/`, `outputs_spiral/` field plots (steps 500–8000, Jan 29 2026)
