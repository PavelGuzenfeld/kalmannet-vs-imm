Benchmark for a gimbal/camera tracker on Jetson Orin (JetPack 6.2): does a learned Kalman filter
(KalmanNet family) beat a well-tuned classical IMM for maneuvering-target tracking, judged on
accuracy, uncertainty calibration, and edge-deployment cost? See [#14](source-notes.md) for the literature
extraction this implementation is grounded in.

## 1. Methodology

### 1.1 State, motion models, measurement model

- **State** (shared by every filter, `filters/state.py`): `x = [px, py, pz, vx, vy, vz, ax, ay, az]`,
  9-dim, sensor at the origin.
- **Motion models** (`filters/motion_models.py`): CV, CT, and CA/Singer are all instances of one
  continuous-time model discretized via **Van Loan's method** (`filters/discretize.py`), verified
  against the textbook closed-form discrete white-noise-acceleration result
  (`tests/test_discretize.py`) rather than hand-typed per-model formulas.
  - CV: Singer axis with a fast-decaying (near-zero) acceleration state.
  - CA: Singer axis with a multi-second correlation time and larger driving noise.
  - CT: exact closed-form coordinated-turn transition (Li & Jilkov survey, Part I) on the
    horizontal `(px,vx,py,vy)` block for a *known* turn rate, Singer axis on `z`. The ground-truth
    generator randomizes the turn rate (and sign) per segment within a configured range; the
    IMM's CT modes assume a fixed nominal `|omega|` (tuned on validation, see below). This
    model/reality mismatch is deliberate; the robustness sweep probes it.
- **Measurement model** (`filters/measurement_model.py`): nonlinear az/el/range,
  `h: Cartesian -> spherical`, sensor at the origin. Every filter shares this same `h()` and its
  Jacobian/sigma-point handling. A scenario may place the fixed sensor elsewhere
  (`"sensor": {"position": [e, n, u]}`); the flight, its measurements and every filter then run in
  the sensor-centred frame, and only the displayed positions are shifted back.
- **Angle wrapping**: `wrap_angle` (`filters/measurement_model.py`) wraps every angular residual to
  `[-pi, pi)` before use, in three independent code paths (numpy filters, UKF's circular sigma-point
  mean, and the torch path used by KalmanNet's BPTT). Each path has its own wrap-boundary
  unit test (`tests/test_measurement_model.py`, `tests/test_ekf_ukf.py`,
  `tests/test_torch_ops.py`, `tests/test_knet_filter.py`). This is deliberate: the KalmanNet
  reference implementation this benchmark ports from was built for the Lorenz-attractor/linear
  cases and never wraps angles, so this is the one place a regression could silently
  reappear.

### 1.2 Data (`data/`)

- `data/config.py` defines the nominal regime (50 Hz, 15 s trajectories, piecewise CV/CT/CA
  segments with randomized durations/turn-rates/accelerations) and four shifted regimes for the
  robustness sweep: `high_maneuver`, `high_noise`, `low_noise`, `heavy_tailed` (Student-t
  measurement noise, evaluated against the nominal Gaussian `R` every filter is told to
  assume; the filters are not informed of this shift).
- `data/trajectory_gen.py` generates ground truth and noisy measurements; `data/make_dataset.py`
  produces train/val/test splits plus a high-MC subset (`test_highmc.npz`: few
  trajectories, 100 noise realizations each) for the ANEES/NIS chi-square bounds,
  which need many Monte Carlo realizations per time step to be discriminating. The main
  200-trajectory/5-MC test set is too thin for that statistic on its own; it is
  still used for the fraction-inside-bounds check across many (trajectory, step) pairs.
- Every generation parameter (seeds, regime configs, split sizes, git commit) is recorded in
  `datasets/manifest.json`. Train/val/test use independent `numpy.random.SeedSequence`-spawned
  seeds.

### 1.3 Fairness / information asymmetries (stated explicitly, per the brief)

- **EKF, UKF**: single motion model (CV only), the sanity floor.
- **IMM**: 4 modes — CV, CT+, CT- (symmetric nominal turn-rate hypotheses), CA/Singer — tuned on
  the validation set only (see below).
- **Vanilla KalmanNet**: given the *same single CV `f()`* as the EKF/UKF sanity floor, not
  IMM's multi-mode mixture. This is deliberate: KalmanNet's architecture assumes
  one fixed (possibly mismatched) system model and is supposed to compensate for unmodeled
  dynamics via the learned gain alone. That makes it the direct test of whether a learned gain with a
  mismatched `f()` can do what IMM does with explicit model-switching. It shares the same `h()`.
- **Vanilla KalmanNet has no usable covariance.** With state dim `m=9 > n=3` measurement dim, the
  "KG-only, recover Sigma post-hoc" pseudo-inverse trick (needs `H` full column rank) does not
  apply. This is a property of the tracking problem (many more state dims than measurement dims),
  not an artifact of our 9-dim state choice. Vanilla KalmanNet is therefore evaluated on RMSE only;
  NEES/NIS/coverage are reported as N/A. That is itself a finding (see [#14](source-notes.md)
  design-implications section): a filter with no claimed uncertainty cannot be graded on
  calibration, and cannot feed a track-management gate that needs one.
- **Recursive KalmanNet (RKN)** gets the same single CV `f()`/`h()` as vanilla KalmanNet (same
  fairness rationale), but its architecture produces an actual state covariance every step (sec.
  1.6). Unlike vanilla KalmanNet, RKN can be judged on NEES/NIS/coverage on the same footing
  as EKF/UKF/IMM. That comparison is the point of building it.
- **Measurement noise on the sweep**: EKF/UKF/IMM are built with each regime's own `R`
  (`scripts/eval.py`), so on `high_noise` and `low_noise` they are told the shifted noise level.
  Both learned filters take no `R` and see only what they learned from nominal-noise training
  data. On `heavy_tailed` every filter assumes the nominal Gaussian `R`. This asymmetry matters
  for the RKN calibration result in sec. 2.3 and the verdict in sec. 4.

### 1.4 IMM tuning (`scripts/tune_imm.py`)

Coordinate search (`filters/imm_tuning.py`) from `p_stay=0.99, omega_CT=25 deg/s,
sigma_a^CA=2 m/s^2, tau^CA=2 s, sigma_a^CT=0.3 m/s^2` over `p_stay in {0.95, 0.97, 0.99, 0.995, 0.998}`,
nominal `|omega_CT| in {15,20,25,30,40} deg/s`, `sigma_a^CA in {1,2,4} m/s^2`, `tau^CA in {1,2,4} s`,
`sigma_a^CT in {0.1,0.3,1.0} m/s^2`. An axis whose best value sits on a grid edge is extended
geometrically (capped at 3 extensions) so the selection is never an edge artifact; `sigma_a^CT` was
extended once, to 3.33. CV `sigma_a` is pinned at the truth generator's 0.05 (validation RMSE is flat
in it). 28 candidates, each scored by mean position RMSE on a 30-trajectory validation subset; the
winner re-verified on the full 200-trajectory validation set. Full search log:
`results/imm_tuning.json`.

**Chosen**: `p_stay=0.99`, `omega_CT=25 deg/s`, `sigma_a^CA=2 m/s^2`, `tau^CA=2 s`,
`sigma_a^CT=1.0 m/s^2` — position RMSE 1.11 m on the 30-trajectory search subset
(`best.val_pos_rmse`), **1.19 m** on the full 200-trajectory validation set
(`best_full_val_pos_rmse`). The same values drive the page (`web/imm_tuning.json`).

### 1.5 KalmanNet architecture and training (`filters/knet/`, `scripts/train_knet.py`)

Architecture #2 from the reference implementation (`KNet/KalmanNet_nn.py`, three GRUs — Q/Sigma/S
— plus FC pre/post layers and the backward-flow Sigma-hidden-state overwrite), ported with two
changes documented in [#14](source-notes.md) and in code comments:

1. **Angle-wrap fix** (`filters/torch_ops.py`, `filters/knet/knet_filter.py`): the reference never
   wraps angular residuals; ours does, in every feature (`obs_diff`, `obs_innov_diff`) and in the
   update innovation itself.
2. **Near-zero gain-output initialization** (`filters/knet/gain_network.py`): a freshly initialized
   gain matrix has no restoring-force guarantee the way an analytic Kalman gain does. Closed-loop
   `(I-KH)F` can have spectral radius > 1, so an untrained network diverges from step
   one. The final `FC2` layer is initialized near-zero so training starts from "trust the CV
   prediction" (itself stable) rather than an arbitrary, possibly-unstable correction.

**Training-stability findings:** even with the near-zero
init, gradient-norm clipping, and a finite-loss/finite-gradient skip-guard (never a ground-truth
state injection; that would grade KalmanNet on an easier protocol than the classical filters,
which never diverge and never get one), a single-trajectory overfit test showed the
closed-loop dynamics can still drift into instability over many gradient updates, occasionally to
the point of permanent NaN weights without the skip-guard. This mirrors the original paper's
acknowledgment that different BPTT strategies (full vs. truncated-then-full vs. truncated-only)
were needed for training stability on challenging trajectories ([#14](source-notes.md) sec. 1b). It is a
property of this architecture applied to our az/el/range/maneuvering setup, not an implementation
shortcut. Final training run: `results/knet_vanilla/training_log.json` (protocol: truncated BPTT,
truncation length 20 steps, Adam at lr 3e-4, per-step squared-error loss time-averaged per
trajectory, L2 weight decay 1e-4, gradient-norm clip 1.0, validation-based early stopping; batch
size 16, up to 100 epochs, patience 10, seed 0).

### 1.6 Recursive KalmanNet architecture and training (`filters/knet/rkn_*.py`, `scripts/train_rkn.py`)

Adapted from the official reference (`github.com/ixblue/RecursiveKalmanNet`,
`Algo/RecursiveKalmanNet.py` + `Algo/KalmanFilter.py`), per [#14](source-notes.md) sec. 2 and the "from-repo
integration, not a from-paper reimplementation" design implication. As with vanilla KalmanNet, the
reference source was read (not just the paper) before porting.

- **Two independent GRU networks**, each FC-pre -> GRU -> FC-post -> linear, matching the
  reference's `GRUNetwork`: `rnn_K` reads `H` and the state correction rotated into the predicted
  line of sight's (azimuth, elevation, range) frame and outputs a gain `N` there, mapped back to
  the Cartesian `K in R^{9x3}` as `K = blockdiag(R^T) N diag(rho, r, 1)` with `R` the row-normalised
  position block of `H`, so the gain is mirror-equivariant by construction (#94); `rnn_cov`
  outputs the `m(m+1)/2 = 45` free lower-triangular entries of a Cholesky factor `L`, from which
  `B_t = L_t L_t^T` (PSD for any real `L`, so no positivity constraint is needed on the
  diagonal the way Cholesky-KalmanNet's PDEL requires; confirmed from the reference
  code, not assumed).
- **Same 4 features as vanilla KalmanNet, squared elementwise and, for `rnn_cov`, L2-normalised
  per group so a runaway state cannot saturate the GRU (#86)**: innovation, observation
  difference, the measurement Jacobian `H_t` (flattened), and the state correction from the
  previous step. The reference's own features assume a linear `H` multiplying `x`
  directly; adapting to our nonlinear az/el/range `h()` required computing `H_t` as an actual
  Jacobian (`filters/torch_ops.py::H_jacobian_torch`, verified against the numpy `H_jacobian` used
  by EKF/UKF/IMM) rather than reading off a constant matrix.
- **The consistency mechanism**: `P_t = A_t + B_t`, where `B_t` is the learned term above and
  `A_t = (I-K_tH_t) F_t P_{t-1} F_t^T (I-K_tH_t)^T` is computed in closed form from the
  just-estimated gain and the *previous* covariance. This recursive dependence on the network's
  own prior output, and the explicit use of `F_t`, is what the paper calls out as the improvement
  over Split-KalmanNet and Cholesky-KalmanNet ([#14](source-notes.md) sec. 2).
- **Loss**: the reference's tuning-free Gaussian NLL, `L_t = e_t^T P_t^{-1} e_t + logdet(P_t)`,
  batch+time-averaged (`train_rkn.py::_gaussian_nll`). Ported using `torch.linalg.slogdet`
  instead of the reference's raw `log(det(...))`: same value, numerically safer at the
  edges.
- **Same angle-wrap fix as vanilla KalmanNet, in one additional place**: RKN's raw
  observation-difference feature (`z_t - z_previous`) also contains azimuth and needs wrapping,
  not just the innovation. Caught and tested (`tests/test_rkn.py`) before any training.
- **Same near-zero output-layer initialization** applied to *both* `rnn_K` and `rnn_cov`. An
  untrained gain has no stability guarantee (as established for vanilla
  KalmanNet), and an untrained covariance network has an analogous risk: `P_t` recursively
  depends on its own past output through `A_t`, so starting `B_t~0` means `P_t` starts as just the
  closed-form `A_t` term (itself PSD by construction), a classical-filter-like baseline
  to learn deviations from. Applying this up front, rather than rediscovering the
  instability RKN's paper-documented BPTT sensitivity would predict, meant every RKN unit test
  passed on the first run.
- **Network size: the reference's default multipliers do not fit our latency budget.** The
  reference's default `GRUNetwork` config (`fc1_mult=10, hidden_mult=10, fc2_mult=20`) is sized for
  the paper's own (smaller) example systems; at our `m=9`, it produces a 7.85M-parameter
  network whose forward pass alone measures 2.76 ms/step single-threaded, already over the
  2 ms target before any accelerator is involved (sec. 3.1). We report that reference-default
  number as a deployment-audit data point and also train a
  scaled-down variant (`fc1_mult=3, hidden_mult=3, fc2_mult=4`, 301,266 parameters, matching
  vanilla KalmanNet's rough size) for a latency-competitive, directly comparable result. Training
  protocol (`scripts/train_rkn.py` defaults): truncated BPTT (25-step chunks), Adam at lr 3e-4,
  weight decay 1e-4, finite-loss/finite-gradient skip-guard, no ground-truth state injection,
  validation-based early stopping; batch size 32, up to 30 epochs, patience 6, seed 0. Best epoch 9,
  validation position RMSE 1.56 m. Training log: `results/rkn/training_log.json`.

### 1.7 Bayesian KalmanNet (`filters/knet/numpy_forward.py`, `scripts/tune_bkn.py`, #135)

Bayesian KalmanNet (BKN) per [#14](source-notes.md) sec. 4 and the paper's Algorithm 1 (Dahan et al.,
arXiv:2309.03058, p. 6), built on the vanilla KalmanNet of sec. 1.5. f, h and the three GRUs are
unchanged; only this differs:

- **MC dropout** with Bernoulli `p` on every ReLU output of the gain network's FC layers (FC5, FC6,
  FC1, FC7, FC2's hidden layer, FC3, FC4), never on the gain output itself. It stays on at
  inference. The state_dict is the vanilla one, so the page loads BKN weights with the same
  numpy forward pass.
- **J realisations per step.** Every realisation predicts from the previous ensemble mean (Algorithm
  1 line 6), so all of them share the prior and the innovation; each keeps its own GRU hidden state
  and gets fresh masks every step (line 4). The estimate is the sample mean (eq 22a); `P` is the
  sample covariance around it, normalised by `J` (eq 22b). At `p = 0` every realisation is vanilla
  KalmanNet, so BKN reproduces it bit for bit with `P = 0` (`tests/test_bayesian_kalmannet.py`).
- **That `P` is an ensemble spread, not a propagated covariance** (MS-39). It is the one-step spread
  of the gain times the innovation and carries no prior uncertainty and no `Q`. Its rank is at most
  `J - 1`, so `J >= 10` is needed to whiten a 9-state error at all.
- **NIS** needs an innovation covariance the paper never defines. This benchmark uses
  `S_t = H_t F P_{t-1} F^T H_t^T + R` with nominal `R` and `P_0 = 0` (MS-40), so BKN's NIS judges
  that construction, not the paper's method.
- **Training** is `scripts/train_knet.py --dropout-p p`: the sec. 1.5 protocol unchanged (plain
  state MSE, seed 0), with dropout on during training and validation. **Deviation from the paper,
  flagged rather than resolved:** the paper trains with its eq 23 loss (empirical second-moment
  term plus a KL term on a learned `p`, initialised uniformly in [0.5, 0.8]) on its Architecture
  #1, with `J = 20`. #135 asks for plain MSE on our Architecture #2 with a fixed `p`. The
  calibration below is therefore that of MSE-trained MC dropout, not of the paper's full method.
  The eq 23 training is built and measured separately below (#149); plain MSE stays the default.
- **Dropout seed: per trajectory.** One numpy Generator per run, spawned from the run's seed
  (`dropout_rng`; its own stream, never the measurement-noise stream drawn from the same seed),
  draws every step's masks in realisation order. eval.py seeds runs with a counter per regime; the
  nominal one carries on through `test.npz` (seeds 0-999), `test_highmc.npz` (1000-1499) and the two
  scenarios (1500-1501), so its scenarios row is not the run the page draws for those flights. The
  page and `replay.py` seed it with the scenario seed, so a replay of an exported scenario redraws the
  page's masks (`tests/test_live_flight.py`). Reseeding per step from one seed would repeat the same
  masks at every step; deriving a per-step seed buys nothing a single Generator does not already
  give.

**Choosing p and J on validation only** (`results/bkn_tuning.json`, `val.npz`, 200 trajectories, 1
noise realisation each, through the page's numpy forward pass). One checkpoint per `p` in
{0.05, 0.1, 0.2} (best epochs 15, 10 and 3; single-realisation validation RMSE 725.5, 694.7 and
746.0 m, no burn-in or divergence cut), each run at `J` in {10, 20, 40}. The rule, fixed before the
runs: keep the runs within 5 points of the fewest diverged and within 10% of the best position RMSE,
then take the ANEES closest to 9 in log ratio, the smaller `J` on a tie.

| p | J | Pos RMSE (m) | Diverged | ANEES | ANIS | Coverage@95 |
|---|---|---|---|---|---|---|
| 0.05 | 10 | 74.63 | 90.5% | 2.6e22 | 7,699 | 0 |
| 0.05 | 20 | 74.67 | 90.5% | 1.3e14 | 7,961 | 0 |
| 0.05 | 40 | 74.70 | 90.5% | 1.5e13 | 7,922 | 0 |
| 0.1 | 10 | 73.18 | 88.0% | 1.4e21 | 5,947 | 0 |
| 0.1 | 20 | 73.08 | 88.0% | 5.4e13 | 5,882 | 0 |
| 0.1 | 40 | 73.05 | 88.0% | 1.2e13 | 5,836 | 0 |
| 0.2 | 10 | 74.03 | 89.5% | 2.5e14 | 5,962 | 0 |
| **0.2** | **20** | 74.06 | 89.5% | **5.9e11** | 5,942 | 0 |
| 0.2 | 40 | 74.07 | 89.5% | 9.9e11 | 5,974 | 0 |

Every run passes both accuracy filters; **`p = 0.2`, `J = 20`** wins on ANEES. The choice is between
equally failed calibrations: no run puts a single counted step inside its 95% ellipsoid, and the
ANEES mean swings by orders of magnitude with `J` because a few near-singular steps dominate it.
Accuracy is flat in `J` (under 0.2 m) and within 1.7 m across `p`.

**Eq 23 training with a learned p (#149).** `scripts/train_knet.py --learned-p` trains the same
network the way the paper's sec. IV-B says (source-notes sec. 4), on our Architecture #2:

- **Loss.** Per step, eq 20: `(1 - β) ℓ2 + β Σ_j |e_j² - Σ̂_jj|` (eq 19), on the mean and the
  per-state spread of a `J = 10` training ensemble. The ensemble runs Algorithm 1 as the evaluator
  does: every member predicts from the previous ensemble mean (`BayesianKalmanNetFilter` in
  `filters/knet/knet_filter.py`). Both terms carry the sec. 1.5 per-state weights, so position does
  not swamp the rest. The paper's expectation over dropout draws is one draw of `J` masks per step.
- **Learned p.** One `p` per dropout layer (the seven above), as concrete dropout (Gal, Hron, Kendall
  2017, the paper's [47]): a relaxed Bernoulli mask, temperature 0.1, eps 1e-7, from their code. Each
  `p` starts uniform in [0.5, 0.8] (the paper's sec. V-A) and has no weight decay.
- **KL term** (eq 24), added once per update: `c1 ‖θ_λ‖² / (1 - p_λ) - c2 H(p_λ) n(λ)`, with `c1 =
  l²/N`, `c2 = 2/N`, `l = 1e-4` and `N = 1000` training trajectories, [47]'s scaling. Our dropout sits
  on FC outputs, so `θ_λ` is the weight columns the dropped units feed and `n(λ)` their count: FC5 ->
  GRU_Q input (45), FC6 -> GRU_Sigma input (45), FC1 and FC7 -> GRU_S input (9, 30), FC2's hidden
  layer -> its output layer (3,600), FC3 -> FC4 (81), FC4 -> GRU_Sigma's hidden-to-hidden weights
  (81), since FC4's output becomes that GRU's state. Adam's 1e-4 weight decay still applies to every
  weight; `c1` is 1e-11, so the weight term is negligible next to it.
- **Unspecified in the paper, chosen here:** `J_train = 10` (the fewest that give a 9-state `P` full
  rank at inference; the 1/J normalisation biases the training spread to 0.9σ²); β ramps linearly to
  its final value over 2 epochs and no checkpoint is kept before it gets there (the paper only says
  "small, then increased"); early stopping on validation eq 20 at the final β, patience 4, at most 12
  epochs. Patience and the cap are shorter than sec. 1.5's 10 and 100: an epoch took 20-45 min on the
  shared CPU. Final β in {0.5, 0.9}; a first round that also tried 0.1 was stopped because every
  run's best epoch fell inside the ramp, where the loss was at least 80% plain MSE.
- **Inference** is unchanged: hard Bernoulli masks from the numpy forward pass, each layer at its
  learned `p` (`learned_dropout_p`), not the relaxed masks training used.

Both checkpoints went through the tuning rule above (`results/bkn_eq23_tuning.json`, `val.npz`):

| β | best epoch | learned p (FC5, FC6, FC1, FC7, FC2, FC3, FC4) | J | Pos RMSE (m) | Diverged | ANEES | ANIS | Coverage@95 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 5 | 0.64, 0.55, 0.58, 0.63, 0.48, 0.51, 0.50 | 10 | 73.04 | 88.0% | 2.7e23 | 6,402 | 0 |
| 0.5 | 5 | | 20 | 72.75 | 88.0% | 4.2e12 | 5,983 | 0 |
| 0.5 | 5 | | 40 | 72.76 | 88.0% | 3.4e11 | 5,844 | 0 |
| 0.9 | 3 | 0.70, 0.59, 0.60, 0.68, 0.50, 0.60, 0.52 | 10 | 73.69 | 91.5% | 9.2e16 | 6,372 | 0 |
| 0.9 | 3 | | 20 | 73.62 | 91.5% | 1.3e11 | 6,235 | 0 |
| **0.9** | **3** | | **40** | 73.87 | 90.5% | **5.6e10** | 6,188 | 0 |

Every run passes both accuracy filters; **β = 0.9, J = 40** wins on ANEES. The learned `p` drifted
down from their draws (0.55-0.79) and were still drifting when early stopping fired; FC2's hidden
layer, where the entropy term weighs 3,600 units, settled at 0.48-0.49, next to the prior's 0.5. On
validation the eq 23 runs sit in the plain-MSE band: at `J >= 20` ANEES 5.6e10 to 4.2e12 against
plain MSE's 5.9e11 to 1.3e14, no counted step inside the 95% ellipsoid, the same accuracy and
divergence.

### 1.8 Unscented KalmanNet (`filters/knet/ukn_*.py`, `filters/knet/numpy_forward.py`, `scripts/train_ukn.py`, `scripts/tune_ukn.py`, #129)

Unscented KalmanNet (UKN; Ko & Shafieezadeh, arXiv:2608.04201v2) keeps the UKF of sec. 1.1 as its
backbone and learns three things on top: the process noise, the measurement noise and a correction
to the gain. No reference code exists, so this is built from the paper. Spec lines MS-43 to MS-48 in #1.

- **Backbone (MS-43).** f, h, the sigma points, the MS-33 jitter, the circular azimuth mean and the
  Joseph update are `filters/ukf.py`'s own; UKN calls them. The weights stay at α = 1, β = 2, κ = 0.
  The paper uses β = 0, so this is a **deviation**, by the author's ruling on #129: untrained UKN is
  our UKF, not the paper's. Both output layers start at zero, so the untrained filter is the UKF bit
  for bit on estimate, `P`, `y` and `S` (`tests/test_unscented_kalmannet.py`, the slice test).
- **NoiseNet (MS-44)** scales the CV `Q` and the nominal `R` through lower-triangular Cholesky
  multipliers `A` (eq 9, 12, 13). Each diagonal entry is `sqrt(clip(exp(4 tanh q), 0.1, 50))`, each
  strictly lower entry `tanh(a)`. The product is computed as `base + L(A Aᵀ − I)Lᵀ`, equal to eq 9 in
  exact arithmetic and the baseline bit for bit at `A = I`. **No jitter floor was needed**: the CV
  `Q` has eigenvalues from 4.4e-13 to 1.4e-3 (cond 3.1e9) and its Cholesky succeeds in float64 in
  numpy and torch. The classical UKF is therefore unchanged, and its rows in sec. 2.5 reproduce the
  sec. 2.1 and 2.4 numbers exactly. Ceiling: 50× the variance of σ_a = 0.05 m/s² is about
  0.35 m/s², far below the manoeuvres, so NoiseNet alone cannot absorb a turn.
- **Features (MS-45).** GainNet reads eq 19 (whitened innovation, NIS, log det S, predicted
  increment, log diag P, `K_UKF y`), 32 inputs. NoiseNet reads eq 10 and 11 with the measurement as
  `[sin az, cos az, el, range]` and no control term, so 25 inputs, one more than #129 counted. Every
  innovation-derived input uses the wrapped residual. Sizes as the paper's manoeuvring-target
  example: d_e 32, H_G 128, H_N 64; 130,420 parameters.
- **GainNet (MS-46), deviation.** Column `j` of `ΔK` is bounded by `c_K ‖K_UKF[:, j]‖`, not by eq 20's
  single `c_K ‖K_UKF‖_F`. K's angle columns are in m/rad (about the range, up to km) and its range
  column in m/m, so eq 20 lets the range column move by thousands of times its own entries. With eq
  20 as written at `c_K = 1`, one Adam step made every later training batch lose positive
  definiteness; per column, none did.
- **Posterior (MS-47).** `x = x⁻ + (K_UKF + ΔK) y`, `P = P_UKF + ΔK S ΔKᵀ` (eq 17), the paper's
  plug-in form: it treats `ΔK` as fixed although it depends on the innovation.
- **Training (MS-48)**, `scripts/train_ukn.py`, nominal training set only. Loss eq 28: KalmanNet's
  weighted state MSE + `r w_cal L_cal` (eq 23) + `r w_meas L_meas` (eq 24) + `w_ΔK ‖ΔK‖²/‖K_UKF‖²`.
  **Deviations:** the ΔK term is relative to `‖K_UKF‖²` (the units of MS-46), and the MSE carries
  KalmanNet's per-group weights. Constants fixed before any run: soft clamp `c = 9`, Student-t
  `ν_df = 4`, EMA `β = 0.7`, `η = 0.5`, `τ = 0.1`, `w` in [0.1, 10], `w_ΔK = 0.01`, 2 MSE-only
  epochs then a 3-epoch ramp, Adam lr 1e-3 without weight decay, batch 32, truncated BPTT 25 steps,
  gradient clip 1, at most 30 epochs with patience 6 on validation position RMSE (no burn-in, no
  divergence cut) once the ramp is complete, seed 0, start `x0 ~ N(truth, P0)` with eval.py's `P0`.
  The adaptive weights went to their cap of 10 in the first epoch and stayed there: the smoothed
  discrepancies never fell below τ (at the chosen run, `g_cal` 1.2 and `g_meas` 0.55 at the end),
  so after warmup eq 27 acted as fixed weights of 10.

**Choosing `c_K` on validation only** (`results/ukn_tuning.json`, `val.npz`, 200 trajectories, 1
noise realisation each, through the page's numpy forward pass). One training run per `c_K` in
{0.1, 0.3, 1.0}; best epochs 11, 14 and 8 of 18, 21 and 15 run (about 2.4-3 h each on 3 threads).
The rule is BKN's (sec. 1.7), fixed before the runs: within 5 points of the fewest diverged and 10%
of the best position RMSE, the ANEES closest to 9 in log ratio, the smaller `c_K` on a tie.

| c_K | Pos RMSE (m) | Vel RMSE (m/s) | Diverged | ANEES | ANIS | Coverage@68/95 |
|---|---|---|---|---|---|---|
| 0.1 | 3.50 | 19.85 | 0% | 4,902 | 20.4 | 0.095 / 0.176 |
| **0.3** | **1.71** | **8.78** | 0% | **101** | **5.55** | **0.362 / 0.491** |
| 1.0 | 11.74 | 19.54 | 0% | 11,063 | 187.5 | 0.153 / 0.223 |

**`c_K = 0.3`** is the only run inside the accuracy band, and it is also the best on every other
column; it sits inside the grid. The checkpoints were trained from the tree of this PR's first
commit before its last bitwise-equivalent rewrites (artifacts stamped `staged-129`).

### 1.9 Cholesky-KalmanNet (`filters/knet/ckn_*.py`, `filters/knet/numpy_forward.py`, `scripts/train_ckn.py`, `scripts/tune_ckn.py`, #136)

Cholesky-KalmanNet (CKN; Ko & Shafieezadeh, IEEE SPL 32 (2025) 326–330) is ported from its
released code, `github.com/RAMSIS-Lab/ckn-spl-public`, onto our CV f and az/el/range h. Spec lines
are MS-51 to MS-53 in #1. Source-notes sec. 3 lists what the released code does that the paper
does not say.

- **Structure (MS-51).** CKN is Split-KalmanNet with a PDEL on both outputs. Branch 1 learns the
  prior covariance G¹ and branch 2 the inverse innovation covariance G². The gain is
  `K = G¹ Hᵀ G²`. Each branch is Linear(→960)–ReLU, a 2-layer GRU of 180, then
  Linear(→108)–ReLU–Linear, sized by the reference's dnn.py formulas: 1,751,535 parameters. The PDEL
  fills a lower-triangular factor row by row and sets its diagonal to `|a| + 1e-8` (eq 4).
- **Units, deviation.** The reference trains on unit-scale toys. Here `S⁻¹` reaches about 1e6 rad⁻²
  on the angles, so the factors are emitted in scaled units: `G¹ = D_x L₁L₁ᵀ D_x` and
  `G² = D_z⁻¹ L₂L₂ᵀ D_z⁻¹`. `D_x` is KalmanNet's σ_init (30 m, 3 m/s, 0.5 m/s²) and `D_z` is the
  nominal measurement σ. The features use the same units, including the Jacobian as
  `D_z⁻¹ H D_x`.
- **Not ported from the reference.**
  - Its posterior line is a Hadamard product by operator precedence. Ours is the Joseph form at
    the learned prior, carried as a QR factor (MS-52).
  - It inverts branch 2's output twice, adding a 1e-8 jitter in between. We use `K = G¹ Hᵀ G²`
    directly.
  - Its GRU start states are random and are not saved with the weights. Ours start at zero.
  - It never wraps angles. Ours wraps `y` and `z_t − z_{t−1}`. The linearisation error
    `h(x̂) − H x̂` is exactly `[az, el, 0]` for our h, so it enters as `[sin az, cos az, el]`.
  - Its output layer uses PyTorch's default init. Ours starts with diagonal bias 2⁻⁶ and 1, because
    a zero factor gets no gradient through `L Lᵀ`.
- **NIS** is `yᵀ G² y`: S = (G²)⁻¹ is CKN's own innovation covariance, not `H G¹ Hᵀ + R`.
- **Training (MS-53).**
  - Loss eq 5–7 with the reference's reductions: `(1 − β)` times the MSE averaged over the 9 states,
    plus `β` times the L1 summed over all 81 entries of `e eᵀ − P`. Entries are weighted by KalmanNet's
    1/σ_init², so both terms are unitless.
  - As in the reference, Adam runs at 5e-4 on one branch at a time, alternating every step, with
    the gradient clipped to 1. We clip the active branch only.
  - As in our other learned filters: truncated BPTT of 25 steps, batch 32, at most 30 epochs with
    patience 6 on validation position RMSE, and seed 0.
  - **The released β = 0.05 does not mean what it means in the reference.** Summed over 81 entries,
    the L1 term is 16× the MSE term untrained and 24× after a smoke run. On the reference's 2-state
    toy it is at most about 3.3×. So β = 0.05 weights the two terms about 1:1 here, where the
    reference weights MSE about 6:1, and β ≈ 0.01 reproduces the reference's balance.

**Choosing β on validation only** (`results/ckn_tuning.json`, `val.npz`, 200 trajectories, 1 noise
realisation each, through the page's numpy forward pass). One run per β in {0.01, 0.05, 0.25}, a grid
fixed before training. The best epochs were 2, 5 and 5 of 9, 12 and 12 run, about 1.5–1.8 h each
with the three running in parallel on 6 threads each. The selection rule is BKN's (sec. 1.7).

| β | Pos RMSE (m) | Vel RMSE (m/s) | Diverged | ANEES | ANIS | Coverage@68/95 |
|---|---|---|---|---|---|---|
| 0.01 | 9.00 | 12.47 | 1.5% | 1.4e11 | 8.59 | 0.001 / 0.002 |
| 0.05 (released) | 7.16 | 12.51 | 1.5% | 2.7e12 | 12.6 | 0.000 / 0.000 |
| **0.25** | **5.19** | 12.92 | 2.5% | **31,325** | 36.5 | **0.002 / 0.006** |

**β = 0.25** is the tuned β. It is the only run inside the 10% RMSE band, and it also has the ANEES
closest to 9. It sits at the top edge of the grid, so a larger β was not tried. Every run's validation RMSE swung by 10× or more between epochs
(β = 0.05: 563, 154, 3268, 30, 410, 10.6 m over epochs 0–5), so the selected checkpoints are single
lucky epochs, not converged runs.

## 2. Results

All numbers below: full `datasets/test.npz` (200 trajectories x 5 Monte Carlo noise
realizations each) for RMSE and the fraction-inside-95%-bounds check; `test_highmc.npz` (5
trajectories x 100 realizations) for the headline ANEES/ANIS/coverage numbers, which need many
realizations per time step to be discriminating. First 20 steps (0.4 s) excluded from every
statistic to let the initial-condition perturbation settle.

**Divergence rule ([#111](https://github.com/PavelGuzenfeld/kalmannet-vs-imm/issues/111)).** A run
is diverged from the first step its position error exceeds 200 m or goes non-finite, and the
evaluator stops stepping it there. Every statistic counts only the steps before that step, and
every RMSE row also carries `diverged_fraction` (share of runs that diverged) and
`median_divergence_step` (of 750 steps at 50 Hz). This changes what a pos RMSE means for any
filter that diverges: it is the error *while still tracking*, capped near 200 m by construction,
not the whole-run error. The earlier revision of this report counted every step, so its numbers
for EKF/UKF and vanilla KalmanNet (112.59 m and 576.75 m nominal) are not comparable with the ones
below; for those filters, read the diverged fraction first. IMM and RKN never diverge, so their
numbers mean the same thing as before.

`scripts/eval_table.py` copies the IMM, KalmanNet, RKN and BKN rows of those files into one
`results/eval_table.csv` (RKN's ANEES, coverage and n_mc from `results/rkn_eval.json`; empty cells
where an evaluator reports no such column).

Full CSV: `results/results_table.csv`
(classical filters and Bayesian KalmanNet) + `results/knet_test_rmse.csv` / `results/rkn_test_rmse.csv`
(learned filters); full high-MC JSON: `results/anees_highmc.json` (classical and BKN) +
`results/rkn_eval.json` (RKN); plots
(`scripts/make_plots.py`, which stops each run at its divergence step like the evaluators) in
`results/plots/`: `trajectory_overlay.png` (a maneuvering test trajectory — IMM tracks the turn the
CV-only floor cuts the corner on), `rmse_vs_time.png` (EKF/UKF grow from ~1 m to ~85 m by 7.5 s as
maneuvers accumulate, then saw-tooth between ~60 m and ~85 m as runs that pass 200 m drop out of
the average; IMM stays flat near 1 m throughout), `nees_vs_time.png` (EKF's ANEES peaks at
~4x10^6 near 8 s against an expected value of 9, stepping down as diverged runs drop out; IMM's
oscillates around the expected value near the chi-square band), `coverage.png` (EKF/UKF near zero, IMM near nominal, RKN in between),
`accuracy_vs_intensity.png` (RKN within ~1.3x of IMM everywhere and below it on `high_maneuver`;
vanilla KalmanNet and EKF/UKF two orders of magnitude above both), `latency_vs_parameters.png` (the
reference-default RKN sizing sits far above the 2 ms line; every trained/deployed variant sits
under it).

### 2.1 Accuracy and calibration, nominal regime

| Filter | Pos RMSE (m) | Vel RMSE (m/s) | Diverged (median step) | ANEES (main, N=5) | bounds (95%) | frac in 95% | ANEES (high-MC, N=100) | bounds (95%) | Coverage@68/95 |
|---|---|---|---|---|---|---|---|---|---|
| EKF (CV-only) | 55.89 | 42.56 | 29.5% (536) | 1,186,926 | 5.67-13.08 | 0.123 | 906,770 | 8.19-9.85 | 0.036 / 0.047 |
| UKF (CV-only) | 55.89 | 42.56 | 29.5% (536) | 1,186,897 | 5.67-13.08 | 0.123 | 906,769 | 8.19-9.85 | 0.036 / 0.047 |
| IMM (CV/CT+/CT-/CA, tuned) | **1.17** | **4.48** | 0% | 8.98 | 5.67-13.08 | 0.337 | **9.53** | 8.19-9.85 | 0.702 / 0.862 |
| KalmanNet (vanilla, arch #2) | 74.57 | 37.12 | 86.3% (408) | N/A | N/A | N/A | N/A | N/A | N/A |
| Recursive KalmanNet (RKN, 301k params) | 1.49 | 5.99 | 0% | 10.27 | 5.67-13.08 | 0.362 | 15.02 | 8.19-9.85 | 0.553 / 0.732 |

*Accuracy*: IMM beats the single-model CV floor by ~48x on position RMSE (1.17 m vs. 55.89 m) even
with the floor's diverged tails cut off, and 29.5% of the floor's runs pass 200 m before 15 s.
That is expected: the test trajectories spend most of their time in CT/CA maneuver segments a
CV-only filter cannot track. RKN is the first learned filter here in IMM's range: 1.49 m, 1.27x
IMM, no diverged runs. *Calibration*: EKF/UKF's ANEES is ~90,000x the upper bound of its own
claimed 95% interval over the steps before divergence, with only 3.6-4.7% empirical coverage where
a well-calibrated filter gives ~68%/~95%. That is an overconfident filter, the failure mode the
brief calls out as unsafe for track-management gating. IMM now lands inside its own high-MC ANEES
bound (9.53 vs. 8.19-9.85), and its coverage (70%/86% vs. nominal 68%/95%) is slightly short at
95% but in the right regime. RKN is overconfident by ~1.5x on the high-MC ANEES (15.02) with
coverage 55%/73%, closer to IMM than to the floor.

### 2.2 Vanilla KalmanNet: accuracy, and why there is no calibration number at all

Best checkpoint (epoch 11, early-stopped at epoch 21): validation position RMSE 718.42 m (no
burn-in exclusion, no divergence cut; `results/knet_vanilla/training_log.json`). From epoch 1 on,
validation RMSE swung between 718 m and 4,493 m from epoch to epoch (epoch 0: 5,718 m). On the test set
(`results/knet_test_rmse.csv`), **86.3% of nominal runs diverge**, at a median step of 408 (8.2 s
into a 15 s trajectory); the position RMSE over the steps before divergence is 74.57 m. Across the
sweep regimes 74-94% of runs diverge (high_maneuver worst: 94%, median step 249), with
pre-divergence RMSE 70-79 m.

This is worse than the naive CV-only EKF/UKF floor (14-50% of runs diverged, 51-71 m before
divergence), let alone IMM (never diverges; 1.17 m nominal, 0.44-3.88 m across the sweep). Per
sec. 1.5, this is a timeboxed finding, not an under-tuned strawman:

- The architecture-#2 port was cross-checked line-by-line against the reference
  `KNet/KalmanNet_nn.py` and unit-tested (`tests/test_knet_filter.py`, `tests/test_torch_ops.py`)
  including a dedicated angle-wrap-boundary test mirroring the classical filters'.
  `tests/test_discretize.py` and `tests/test_motion_models.py` independently verify the shared
  `f()` both KalmanNet and the EKF/UKF floor use is correct.
  - A single-trajectory overfit diagnostic (timeboxed, not left in the repo; its
    conclusion is captured here) showed the architecture and gradient path are sound: the
    network can reduce its own training loss and does affect tracking behavior. It also
    showed that the closed-loop recursion is only conditionally stable. An updated gain
    matrix can push `(I-K_tH_t)F` outside a stable regime, and once it does, the position error
    compounds for the rest of that trajectory. Neither a much lower learning rate nor a tighter
    gradient-clip norm eliminated this. It is a property of training a from-scratch recurrent gain
    on this system, not a hyperparameter oversight, and it echoes the original paper's
    documented need for multiple BPTT strategies (full / truncated-then-full / truncated-only) to
    get stable training on harder trajectories ([#14](source-notes.md) sec. 1).
  - No ground-truth state is ever injected to mask this (the training loop's only safety net
    is skipping a non-finite gradient update, never resetting the recurrent state to truth). The
    86.3% divergence rate is the model's own, unassisted performance.
- **No usable covariance, structurally, not by omission.** Vanilla KalmanNet ("KG-only" in Bayesian
  KalmanNet's own taxonomy, [#14](source-notes.md) sec. 4) has no covariance output at all; the one documented
  post-hoc recovery trick needs `H` full column rank, i.e. state dim `<=` measurement dim. Our
  problem has `m=9 > n=3`, a property of tracking a 9-dim kinematic state from a 3-dim
  az/el/range sensor, not an artifact of choices made for this benchmark. A vanilla KalmanNet
  deployed for real track management would have no calibrated uncertainty to gate on. That is
  arguably a more clear-cut negative finding than "overconfident but present," and it is why
  the brief's required reading list prioritizes Recursive KalmanNet as the variant that fixes this
  gap ([#14](source-notes.md) sec. 2). Built and evaluated next (sec. 2.3).

**Bayesian KalmanNet gives vanilla KalmanNet a calibration number, and it is a failing one.** MC
dropout on the same gain network (sec. 1.7, `p = 0.2`, `J = 20`) produces a covariance with no
full-column-rank requirement, judged on the same metric code as every other filter
(`results/results_table.csv`, `results/anees_highmc.json`):

| Filter | Pos RMSE (m) | Diverged (median step) | ANEES (main, N=5) | ANEES (high-MC, N=100; bounds 8.19-9.85) | ANIS (high-MC; bounds 2.54-3.50) | Coverage@68/95 |
|---|---|---|---|---|---|---|
| KalmanNet (vanilla) | 74.57 | 86.3% (408) | N/A | N/A | N/A | N/A |
| Bayesian KalmanNet (p 0.2, J 20) | 72.48 | 87.4% (395) | 1.2e12 | 2.6e12 | 577 | 0.000 / 0.000 |
| Bayesian KalmanNet, eq 23 (learned p, β 0.9, J 40) | 74.39 | 87.8% (399) | 4.6e10 | 3.5e11 | 440 | 0.000 / 0.000 |
| IMM | 1.17 | 0% | 8.98 | 9.53 | 2.95 | 0.702 / 0.862 |

The ensemble mean tracks like vanilla KalmanNet (72.48 m vs. 74.57 m before divergence, 87.4% vs.
86.3% of runs diverged), as expected from a network trained the same way. Its covariance is
overconfident by about eleven orders of magnitude in ANEES, and no counted step of any high-MC run
falls inside even the 68% ellipsoid. On three validation trajectories the ensemble's position
standard deviation sat between 0.1 mm and 48 cm while the error was metres; the median ratio of position error
to that standard deviation was 1,400-3,600. That is MS-39's caveat in numbers: the spread of one
step's gain times a milliradian-scale innovation is not the error the recursion has accumulated.
This is MC dropout trained on plain MSE (the #135 scope), not the paper's eq 23 training, whose
second-moment term exists to fix exactly this (sec. 1.7).

**The paper's eq 23 training does not fix it (#149).** Trained with the second-moment and KL terms
and a learned `p` per layer (sec. 1.7, `results/bkn_eq23_b0.9/`), BKN's covariance moves in the right
direction and stays unusable (`results/eval_bkn_eq23/`, stamped `33c301d+149wip`, the code of
`2a71d30`; same seeds as the plain row, and its EKF/UKF/IMM rows reproduce sec. 2.1 exactly).
High-MC ANEES drops 7.4x, from 2.6e12 to 3.5e11, against bounds 8.19-9.85; the main-set ANEES drops
26x, ANIS from 577 to 440 against 2.54-3.50, and coverage stays 0 at both 68% and 95%. Part of that
drop is `J`: the eq 23 row runs at the `J = 40` validation picked, the plain row at 20, and ANEES moves
by orders of magnitude with `J` (sec. 1.7). At the same `J = 20` the gain is smaller: 4.6x on
validation ANEES (1.3e11 against 5.9e11) and 2.7x on the median high-MC NEES (8.8e7 against 2.4e8). Over every
counted high-MC step (about 153,000 per filter, divergence and burn-in cut as in the evaluator) the
median position spread, `sqrt(tr P_pos)`, grows from 1.4 cm to 3.4 cm at the same `J = 20` (3.6 cm
at `J = 40`), and the median ratio of position error to it falls from 2,900 to 1,060 (990 at `J =
40`), while the median error drops from 48 m to 41 m. The spread is 2.5x wider and still three orders
of magnitude short (scratch diagnostic over `bkn_steps`, not committed; `results/spread_bkn_*.json`). Accuracy and divergence do not change: 74.39 m before divergence, 87.8% of
runs diverged at a median step of 399, against 72.48 m, 87.4% and 395. On the four sweep regimes and the
scenarios BKN's main ANEES falls 9x-25x (still 2.5e9 to 1.3e11) and coverage stays 0 everywhere. The loss asks the spread of one step's gain
times the innovation to match the error the recursion has accumulated; with dropout at `p` 0.5-0.7
the network still cannot spread that wide, and the tracking failure it shares with vanilla
KalmanNet is untouched. The ensemble spread is not a usable covariance under either loss.

### 2.3 Recursive KalmanNet: close to IMM, ahead of it under hard maneuvers

RKN (301k-parameter trained variant, sec. 1.6) is the one result in this report that breaks a
clean "learned filters lose" story:

| Metric | EKF/UKF | IMM | KalmanNet (vanilla) | RKN |
|---|---|---|---|---|
| Diverged runs, nominal | 29.5% | 0% | 86.3% | 0% |
| Position RMSE, nominal (m) | 55.89 (pre-divergence) | **1.17** | 74.57 (pre-divergence) | 1.49 |
| Position RMSE, high_maneuver (m) | 71.12 (pre-divergence) | 2.53 | 79.41 (pre-divergence) | **1.89** |
| ANEES, high-MC (bounds 8.19-9.85) | 906,770 | **9.53** | N/A | 15.02 |
| Coverage @68/95, high-MC | 0.036 / 0.047 | 0.702 / 0.862 | N/A | 0.553 / 0.732 |

- **Accuracy is close to IMM, and ahead of it on the hardest maneuvers.** 1.49 m vs. 1.17 m
  nominal, never diverging, within 1.6x of IMM in every regime (`accuracy_vs_intensity.png`;
  `results/rkn_test_rmse.csv`). On `high_maneuver` (turn rates 20-60 deg/s against IMM's fixed
  25 deg/s CT modes) RKN is 25% better than IMM, 1.89 m vs. 2.53 m. That is the case the brief
  framed as KalmanNet's reason to exist: a learned gain absorbing dynamics outside a fixed model
  bank. On the two recorded flights both sit at 1.49 m.
- **Calibration is far better than EKF/UKF and than vanilla KalmanNet's N/A, but short of IMM
  under nominal conditions.** RKN's high-MC ANEES (15.02) is ~1.5x its own upper bound, where IMM's
  (9.53) is inside it and EKF/UKF's is ~90,000x over. Coverage 0.55/0.73 vs. IMM's 0.70/0.86. The
  closed-form `A_t` + learned-Cholesky `B_t` construction produces a usable covariance on the same
  metric code as every classical filter.
- **Calibration does not survive a noise shift.** RKN was trained only on the nominal measurement
  noise and learned its covariance from it, where IMM is told the true `R`. On `high_noise`
  (4x sigma) RKN's ANEES is 86.25 with coverage 0.014/0.046, i.e. it reports a covariance far
  tighter than its real error; IMM there is 8.79 with 0.72/0.87. On `heavy_tailed` RKN is 20.62
  (0.35/0.58) vs. IMM's 16.56 (0.42/0.67). On `high_maneuver` the order flips: RKN 45.63
  (0.47/0.56), IMM 91.22 (0.34/0.38), both overconfident. `results/rkn_eval.json`,
  `results/results_table.csv`.
- **Interpretation for the deployment question**: RKN as trained here is competitive with IMM on
  accuracy and wins where IMM's model bank is mismatched, but it is overconfident under
  nominal conditions and badly so under a measurement-noise shift it was not trained on. A track
  gate fed RKN's covariance in a noisier-than-trained environment would reject good
  measurements. Sec. 4 weighs this.
- **Reference-default sizing (7.85M params, untrained) latency-audited but not trained**: forward
  pass alone measures 2.76 ms/step, over budget before any weights are learned (sec. 1.6, sec. 3.1).
  Training it would need either a smaller architecture (as done here) or accepting it cannot run
  in this deployment's 20 ms cycle regardless of accuracy. That is why we did not spend the
  (substantially larger) compute budget training the reference-default size.

### 2.4 Robustness sweep (test-only shifted regimes; filters never trained/tuned on these)

Pos RMSE in metres; a percentage is the share of runs diverged under the 200 m rule, whose RMSE
counts only the steps before divergence (sec. 2). IMM and RKN diverge in no run of any regime.

| Regime | EKF/UKF | IMM | KalmanNet | RKN | IMM frac-in-95% | Note |
|---|---|---|---|---|---|---|
| high_maneuver (turn rate 20-60 deg/s, higher Q) | 71.12 (50%) | 2.53 | 79.41 (94%) | **1.89** | 0.081 | IMM's CT modes assume a fixed nominal 25 deg/s turn rate; degrades gracefully under a 2-3x wider true-rate range, but RKN beats it here |
| high_noise (4x az/el, 4x range sigma) | 65.44 (48%) | **3.88** | 76.60 (88%) | 5.00 | 0.341 | RKN overconfident, sec. 2.3 |
| low_noise (0.3x az/el, 0.25x range sigma) | 51.37 (14%) | **0.44** | 70.26 (74%) | 0.68 | 0.334 | |
| heavy_tailed (Student-t, dof=3, same nominal R assumed) | 60.62 (44%) | **2.09** | 77.96 (90%) | 2.46 | 0.434 | see fix note below |
| scenarios (2 recorded flights, nominal R) | 71.99 / 71.94 (0%) | 1.49 | 50.50 (50%) | 1.49 | 0.303 | one run per flight |

Bayesian KalmanNet (`p = 0.2`, `J = 20`) on the same regimes, pre-divergence RMSE (share diverged):
high_maneuver 79.30 m (97.6%), high_noise 80.31 m (91.2%), low_noise 67.43 m (78.0%),
heavy_tailed 76.85 m (92.0%), scenarios 52.82 m (50%). Coverage@95 is 0.000 in every regime.

Vanilla KalmanNet diverges in most runs of every regime; `high_maneuver` brings its divergence
earliest (median step 249 vs. 408 nominal). RKN stays within 1.6x of IMM throughout and beats it
on `high_maneuver`. Its calibration is the regime-sensitive part (sec. 2.3): fine at `low_noise`
(ANEES 4.30, under its bound; coverage 0.88/0.94), badly overconfident at `high_noise`.

**A bug caught by this sweep**: the first evaluation run produced `NaN` for IMM on the heavy-tailed
regime. Root cause: the mode-probability update computed each mode's *raw* Gaussian likelihood via
`exp(log_likelihood)`; an occasional heavy-tailed outlier innovation drives every mode's raw
likelihood to exactly `0.0` in floating point (exponent underflow), so `sum(likelihoods)==0` and
the mode-probability normalization divided by zero, producing `NaN` that then poisoned every
subsequent step for that trajectory. Fixed in `filters/imm.py` by moving the mode-probability
update into log-space (subtracting the max log-weight before exponentiating, a standard
log-sum-exp/softmax stabilization), so it stays numerically stable however extreme a single
innovation is. Full test suite green after the fix; the entire evaluation
was then re-run from scratch on the corrected code. In this revision every evaluation and latency
number comes from one commit (`f74fc30`), from the IMM tuning and RKN training run at `1478abc`
and vanilla KalmanNet trained at `4c68643`; the tuner, both trainers, the filters and the data
generator are unchanged between those commits. Each artifact records its commit. The Bayesian
KalmanNet rows and its validation table were run later from `36aec07` plus the #135 changes (artifacts
stamped `36aec07+bkn-wip`; eval.py and `numpy_forward.py` as committed in the #135 branch); the
EKF/UKF/IMM rows of that same run reproduce the `f74fc30` numbers above exactly.

IMM's accuracy holds across every shifted regime (sub-4 m position RMSE throughout, never
diverging, vs. 14-50% diverged runs for the CV-only floor). The tuning in sec. 1.4 generalizes past
the validation conditions it was chosen on.

### 2.5 Unscented KalmanNet: no diverged runs, IMM-range accuracy, overconfident

`c_K = 0.3` (sec. 1.8), same test sets and metric code as every other filter
(`results/results_table.csv`, `results/anees_highmc.json`; run `results/eval_ukn/` at `ba851f9`,
whose EKF/UKF/IMM rows reproduce the `f74fc30` numbers exactly). Pos RMSE in metres; ANEES is the
full 9-dim one; coverage is joint 9-dim.

| Regime | UKF | IMM | RKN | BKN | UKN | UKN ANEES (main) | UKN ANIS | UKN coverage@68/95 |
|---|---|---|---|---|---|---|---|---|
| nominal | 55.89 (29.5%) | **1.17** | 1.49 | 72.48 (87.4%) | 1.66 | 70.8 | 5.48 | 0.351 / 0.480 |
| high_maneuver | 71.11 (50%) | 2.53 | **1.89** | 79.30 (97.6%) | 2.35 | 362.0 | 6.28 | 0.247 / 0.330 |
| high_noise | 65.44 (48%) | **3.88** | 5.00 | 80.31 (91.2%) | 5.39 | 49.7 | 5.48 | 0.316 / 0.445 |
| low_noise | 51.37 (14%) | **0.44** | 0.68 | 67.43 (78.0%) | 0.73 | 422.4 | 6.14 | 0.343 / 0.451 |
| heavy_tailed | 60.62 (44%) | **2.09** | 2.46 | 76.85 (92.0%) | 2.78 | 75.0 | 14.98 | 0.130 / 0.271 |
| scenarios (2 flights) | 71.94 (0%) | **1.49** | 1.49 | 52.82 (50%) | 1.58 | 14,484 | 5.64 | 0.375 / 0.487 |

High-MC calibration (5 trajectories x 100 realisations, bounds 8.19-9.85): UKN ANEES **50.8**,
ANIS 5.61 (bounds 2.54-3.50), coverage 0.256 / 0.350. Against UKF 906,769 (0.036 / 0.047), IMM
9.53 (0.702 / 0.862), RKN 15.02 (0.553 / 0.732) and BKN 2.6e12 (0 / 0).

- **Accuracy.** UKN never diverges in any regime, where its own backbone diverges in 14-50% of runs.
  Its error sits in IMM's range: 1.41x IMM nominal, 1.11x RKN, and lower than IMM on
  `high_maneuver` (2.35 m vs 2.53 m; no significance test was run, so this is not a "beats" claim
  under MS-26). The gain residual does the work NoiseNet's 50x ceiling cannot (MS-44).
- **Clean vs heavy-tailed.** UKN does not reject Student-t outliers here. Its error grows from 1.66 m
  (nominal) to 2.78 m (`heavy_tailed`), 1.33x IMM's 2.09 m, and its coverage collapses to
  0.13 / 0.27 with ANIS 15.0. The paper's gain over UKF is mostly heavy-tail rejection, but it
  trains on data with 10% heavy-tailed returns; ours trains on Gaussian nominal data only, and the
  shifted regimes stay test-only (sec. 2.4). The eq 24 Student-t term alone did not teach it.
- **Calibration.** Overconfident everywhere: the high-MC ANEES of 50.8 is 5.2x its upper bound,
  between RKN's 15.0 and the UKF's 9e5, with joint coverage 0.26 / 0.35 against nominal 0.68 / 0.95.
  ANIS sits near 5.5 against an expected 3 in the Gaussian regimes, so `S` is also too small. As
  the paper admits, L_cal sees only diag(P) (MS-48); this run does not separate the diagonal from
  the joint shortfall, because per-axis coverage is not computed (follow-up).

### 2.6 Cholesky-KalmanNet: positive definite, not calibrated, about 6x IMM's nominal error

Both β values of sec. 1.9 were run on the same test sets with the same metric code as every other
filter. The released β = 0.05 is in `results/eval_ckn/` and the tuned β = 0.25 in
`results/eval_ckn_tuned/`, each its own `scripts/eval.py` run. Their EKF, UKF and IMM rows
reproduce sec. 2.5's. Pos RMSE is in metres, with the diverged fraction in brackets where it is
non-zero. ANEES is the full 9-dim one.

| Regime | IMM | RKN | UKN | CKN β=0.05 | CKN β=0.25 | ANEES 0.05 / 0.25 | ANIS 0.05 / 0.25 | Coverage@95 0.05 / 0.25 |
|---|---|---|---|---|---|---|---|---|
| nominal | **1.17** | 1.49 | 1.66 | 7.75 (2.0%) | 6.75 (2.2%) | 1.1e11 / 93,357 | 18.9 / 75.5 | 0.001 / 0.006 |
| high_maneuver | 2.53 | **1.89** | 2.35 | 17.14 | 5.49 (2.0%) | 5.6e10 / 63,550 | 28.2 / 63.1 | 0.001 / 0.005 |
| high_noise | **3.88** | 5.00 | 5.39 | 9.84 (4.0%) | 7.55 (5.6%) | 2.9e13 / 48,737 | 55.4 / 185.7 | 0.000 / 0.000 |
| low_noise | **0.44** | 0.68 | 0.73 | 7.24 | 3.61 | 2.1e11 / 23,005 | 11.2 / 20.5 | 0.004 / 0.090 |
| heavy_tailed | **2.09** | 2.46 | 2.78 | 7.89 (2.4%) | 2.75 (2.8%) | 6.3e10 / 10,582 | 29.1 / 25.6 | 0.000 / 0.001 |
| scenarios (2 flights) | **1.49** | 1.49 | 1.58 | 22.65 (50%) | 33.79 (50%) | 1.4e10 / 1.0e6 | 648 / 1,731 | 0.026 / 0.006 |

High-MC calibration (5 trajectories x 100 realisations, bounds 8.19–9.85): CKN ANEES **4.3e11** at
β = 0.05 and **14,339** at β = 0.25. ANIS is 9.7 and 11.1 (bounds 2.54–3.50), and coverage is
0.001 / 0.001 and 0.003 / 0.007. For comparison, IMM is at 9.53 (0.702 / 0.862).

- **Positive definite, as built.** Every emitted P is symmetric and passes Cholesky at every step
  of both committed scenarios, for all three trained checkpoints. The largest cond(P) is 2.2e10
  (β = 0.05, break turn). This is the slice test's property checked on trained weights
  (`tests/test_cholesky_kalmannet.py` checks it on a stand-in).
- **Accuracy.** CKN does not reach the other learned filters. Nominal error is 6.6x IMM's at
  β = 0.05 and 5.8x at β = 0.25, and 2–6% of runs diverge in most regimes. On the overhead pass
  its median position error was 76–412 m across the three checkpoints, the large-H̃ region the
  MS-51 open item names. Training never settled. Validation RMSE swung by 10x or more between
  epochs at every β, so these checkpoints are the best single epochs, not converged runs.
  Lowering Adam's 5e-4 is the first thing to try. It is the reference's value, kept on purpose.
- **Calibration.** The PDEL makes P positive definite, not right-sized. P is far too small at both
  β: coverage is under 1% where 68% and 95% are expected. Raising β from 0.05 to 0.25 cuts the
  high-MC ANEES by 7 orders of magnitude, to 1,500x its upper bound, and also lowers the error
  in all five test regimes (not on the two scenarios). The released β is the worse of the two here
  (sec. 1.9 explains why 0.05 does not mean the same thing at 9 states). The pattern matches the
  RKN paper's criticism of CKN's calibration (source-notes sec. 2), but in this benchmark the
  accuracy gap is the larger problem.

## 3. Deployment analysis (Jetson Orin, JetPack 6.2; GPU reserved for vision, CPU/DLA/PVA only)

Component versions confirmed from NVIDIA's JetPack 6.2 archived release notes:
L4T 36.4.3, CUDA 12.6, TensorRT 10.3, cuDNN 9.3, VPI 3.2, DLA 3.1. These match the brief's stated
target.

### 3.1 Measured CPU latency (batch=1, single-threaded, 200-step warmup, 10,000 measured steps,
median/p99 — methodology per the brief; `results/latency.json`)

| Filter | Median (ms) | p99 (ms) | vs. 2 ms target |
|---|---|---|---|
| EKF (CV-only) | 0.025 | 0.032 | 80x headroom |
| UKF (CV-only, 19 sigma points) | 0.115 | 0.140 | 17x headroom |
| IMM (4 modes, mix+update+combine) | 0.239 | 0.292 | 8x headroom |
| KalmanNet (vanilla, 3 GRUs, 526,116 params) | 0.333 | 0.669 | 6x headroom |
| Recursive KalmanNet (trained, 2 GRUs, 301,266 params) | 0.559 | 0.772 | 3.6x headroom |
| Bayesian KalmanNet (numpy, J = 20 passes of the 526,116-param gain network, plus NEES/NIS) | **4.183** | **4.548** | **over budget, 2.1x at the median** |
| Unscented KalmanNet (numpy, UKF backbone + 3 GRU calls, 130,420 params, plus NEES/NIS; separate loaded run, see below) | 0.644 | 0.762 | 3.1x headroom |
| Cholesky-KalmanNet (numpy, two branches of Linear + 2-layer GRU, 1,751,535 params, plus NEES/NIS; separate loaded run, see below) | 1.146 | 1.380 | 1.7x headroom |
| Recursive KalmanNet (**reference-default sizing**, 7,847,112 params, untrained) | **2.759** | **3.831** | **over budget at median, not just p99** |

Measured on an x86 dev host (not the Orin), one container pinned to a single idle core
(`--cpuset-cpus`, load average ~2.6 on 20 cores), all six filters in one run. RKN's median is up
from the earlier revision's 0.391 ms; its gain has since moved into the line-of-sight frame (#94),
and this run does not isolate how much of the change that accounts for.

The UKN row is the page's numpy forward pass (`scripts/bench_latency.py --ukn-weights`), measured in
a separate run pinned to one idle core of the same host while other jobs loaded it (load average
~9 on 20 cores; `results/eval_ukn/latency.json`). That run put the UKF at 0.343 ms and EKF at
0.059 ms, 3.0x and 2.3x their rows above, so UKN is about 1.9x the UKF in the same run, and its
unloaded figure is likely lower than 0.644 ms. A step is 19 sigma points twice, three GRU calls
(NoiseNet's Q and R stages, GainNet) and three Cholesky factorisations (two of P, one of S).

The CKN row is the page's numpy forward pass (`scripts/bench_latency.py --ckn-weights`), measured in
a separate run on one pinned core of the same host while other jobs loaded it (load average about 10
on 20 cores; `results/ckn_latency.json`). In that run the UKF took 0.417 ms and the IMM 0.764 ms,
3.6x and 3.2x their rows above. CKN was 2.7x the UKF in the same run, so its unloaded figure is
probably lower than 1.146 ms. CKN keeps the reference's sizes, and a step runs two 2-layer GRUs whose
input layer is 960 wide.

The BKN row is the page's numpy forward pass (`scripts/bench_latency.py --bkn-weights`), measured in
a separate run on one pinned core of the same host (`results/eval_bkn/latency.json`; the other rows
stay those of `results/latency.json`): a single numpy KalmanNet pass
there takes 0.175 ms, and BKN runs `J = 20` of them one after another, plus the NEES/NIS scoring the
page does every step. The realisations are independent, so one batched pass over all 20 is the
obvious cut. Since #148 the 20 realisations run as one stacked pass, bit for bit the serial estimates;
the row above is the serial figure and has not been re-measured on the quiet host.

Every filter we trained and deployed, except Bayesian KalmanNet at J = 20, fits the <2 ms per-step target on CPU
alone, single-threaded, with no DLA or PVA offload, including RKN once scaled to a
practical size. The one exception: RKN built at the
reference paper's own default network-size multipliers misses
the latency budget by ~1.4x at the median, before any accelerator or quantization step.
This shows why "port the architecture faithfully" and "meets an
edge latency budget" are separate questions that both need checking. For every filter
sized to fit, the CPU is not the bottleneck, so the DLA/PVA offload question is about
headroom-for-a-future-larger-network, not a correctness-or-budget requirement today.

### 3.2 DLA compatibility audit of the gain network

Researched against the TensorRT 10.3 archived documentation. A version-pinned snapshot
could not be located live on NVIDIA's site: the vendor's `archives/` paths for TensorRT 9.x/10.x
all redirect to `/latest/`. Verified via a Wayback Machine capture whose embedded revision history
confirms it describes TensorRT 10.3.0's chapter structure, cross-checked against NVIDIA's current
docs, which carry a banner stating that their retained DLA chapter "describes DLA
behavior for earlier ... releases" because DLA support was pulled from the 11.x line entirely.

- **DLA has no recurrent-layer support in any TensorRT version, 10.3 included. This is
  architectural, not a version gap.** DLA is a fixed-function CNN accelerator with no gating/recurrence unit in
  silicon. NVIDIA's own `Deep-Learning-Accelerator-SW` repo lists GRU/LSTM/RNN as
  "Reconstruction", not "Supported": the reference reconstruction decomposes a GRU cell into
  Convolution (as the matmul), elementwise Add/Mul, Sigmoid, and Tanh, each of which is
  DLA-supported (with wide margins for our sizes: kernel/channel/batch limits all sit
  orders of magnitude above what a ~9-30-dim network needs).
- **This benchmark did not build the DLA-reconstructed feedforward variant** (priority-order item
  5 in the brief, below the classical-filter + vanilla-KalmanNet baseline this report built and
  evaluated first). The audit does establish that it is architecturally
  buildable: our 20 ms cycle only needs one new measurement processed per invocation, so unlike a
  general RNN deployment we would never need to unroll a sequence; the reconstructed cell is a
  single-timestep feedforward graph with the hidden state carried as an app-managed buffer between
  DLA invocations, matching NVIDIA's own reconstruction pattern.
- **Precision**: DLA 3.1 supports FP16 and INT8 only (no FP32). INT8 requires calibration (either
  TensorRT's built-in PTQ calibrator, or a pre-quantized ONNX graph with baked-in Q/DQ scales; the
  two paths are mutually exclusive). RKN (sec. 2.3) is the covariance-producing variant that
  would make the brief's quantization-vs-consistency measurement (FP16/INT8-simulated calibration
  drift) meaningful; vanilla KalmanNet has no covariance output to degrade. We did not
  perform that measurement: it presupposes a DLA-deployable network, and
  RKN's trained (301k-param) size was chosen for CPU-latency competitiveness, not for a completed
  DLA port (sec. 1.6, sec. 3.4). Doing both in one benchmark pass was out of scope. It is
  the most concrete next step for extending this work: quantize the trained RKN checkpoint
  (PyTorch dynamic/static quantization is enough to *measure* the effect; an actual DLA build is a
  separate step) and re-run the existing `filters/metrics.py` ANEES/coverage code on the quantized
  outputs. The harness exists; only the quantization step is missing.
- **The undocumented, and likely decisive, risk: per-invocation CPU<->DLA dispatch latency.**
  NVIDIA publishes no official per-submission overhead figure for DLA. The only evidence found is
  informal developer-forum reports, ranging from "DLA increases throughput, not latency" and a
  4.8 ms->23.9 ms GPU-to-DLA regression on a real CNN, to one report of `cudaEventSynchronize()`
  taking >100 ms against ~0.5 ms of actual DLA compute (attributed by NVIDIA staff to the
  synchronization mechanism, not the workload). For a network this small (sub-millisecond compute
  even on CPU), a synchronous per-cycle CPU->DLA round trip at 50 Hz risks costing more
  in submission/sync latency than the network could save. This is the concern the
  brief flags. Only measurement on real Orin hardware via the cuDLA
  standalone/hybrid-mode path can resolve it; it cannot be projected from documentation alone, so
  no number is given here.
- **DLA Standalone/cuDLA is the right integration path if pursued**, not the TensorRT-engine
  runtime: `EngineCapability::kDLA_STANDALONE` emits a raw DLA loadable consumed directly via the
  cuDLA API (`cudlaCreateDevice` -> `cudlaModuleLoadFromMemory` -> `cudlaSubmitTask`), bypassing the
  TensorRT runtime and its GPU-fallback machinery, which is irrelevant overhead here since
  the GPU is unavailable by design.
- **Hardware caveat**: DLA is present as 2x NVDLA v2 cores on Jetson AGX Orin, 1x on Orin NX, and
  not present on Orin Nano. Confirm which Orin SKU is the actual
  target before investing engineering effort in this path.

### 3.3 PVA (VPI 3.2) assessment: not worth it for this workload

Verified against a version-pinned VPI 3.2 documentation snapshot (unlike TensorRT, VPI's docs
preserve one).

- VPI 3.2's PVA backend exposes a closed catalog of fixed-function image-processing algorithms
  (box/Gaussian filters, pyramid generation, KLT tracking, Harris corners, pyramidal LK optical
  flow, DCF tracking, AprilTag detection, etc.). No general dense linear algebra (matrix-vector
  products, small matrix inversion) is exposed through VPI's public API on any backend, PVA
  included.
- Reaching PVA for Kalman-filter-style dense linear algebra would require the separate, low-level
  **cuPVA SDK**: hand-authoring kernels for a 7-way VLIW SIMD DSP via the Synopsys ASIP Designer
  toolchain. That is a specialized embedded-DSP programming model, different from CUDA or
  even DLA/cuDLA, with no public precedent found for Kalman-filter-style workloads.
- This repo already has an in-production precedent for what PVA is good for here:
  `gst/nvmmsamurai/gmc_vpi_pva.hpp` is a VPI/PVA-backed global-motion-compensation backend,
  the kind of fixed-function vision workload PVA is designed for. A Kalman filter's per-step
  dense linear algebra (9x9 and smaller matrix operations, all under 0.5 ms even on CPU per sec.
  3.1) is a much worse fit.
- **Verdict: PVA is not worth pursuing for any component of this filter.** The classical filter
  algebra is already trivial for the CPU at these latencies (sec. 3.1). PVA offload would free
  marginal CPU headroom, not add speed, and the cuPVA engineering cost to get there is disproportionate
  to that payoff.

### 3.4 Deployment mapping table (required regardless of what was built; measured vs. projected
stated explicitly per row)

| Component | Candidate | Engine | Precision | Cross-engine transfer | Per-step latency |
|---|---|---|---|---|---|
| Predict + update, single model | EKF | CPU | FP64 (FP32 in production) | none | **0.025 ms measured** |
| Predict + update, sigma points | UKF | CPU | FP64 | none | **0.115 ms measured** |
| Mixing + 4x predict/update + combination | IMM | CPU | FP64 | none | **0.239 ms measured** |
| Gain network (3 GRUs, 526k params) | KalmanNet vanilla | CPU | FP64 (FP32 production) | none | **0.333 ms measured** (p99 0.669 ms) |
| Gain RNN + Cholesky-factor covariance RNN (2 GRUs, 301k params, scaled down for latency) | Recursive KalmanNet, trained | CPU | FP64 (FP32 production) | none | **0.559 ms measured** (p99 0.772 ms) |
| Same architecture, reference paper's default sizing (7.85M params, untrained) | Recursive KalmanNet, reference-default | CPU | FP64 | none | **2.759 ms measured — over the 2 ms budget at the median**, illustrating that faithful-to-paper sizing and edge-latency fitness are separate questions |
| Gain network, reconstructed feedforward (Conv/FC/Sigmoid/Tanh) | KalmanNet DLA-variant | DLA | FP16 | CPU->DLA round trip **every 20 ms cycle** | **not built; not projectable** — compute itself plausibly sub-0.1 ms at our trained sizes, but no documented per-submission dispatch/sync overhead exists to bound the total; treat as an open hardware-measurement question, not a number to plan around |
| Any component | -- | PVA | -- | -- | **not attempted, and assessed not worth attempting** (sec. 3.3) |

### 3.5 Latency budget check

Recommended configuration for this problem size, given every measured CPU number for a
practically-sized network sits well under budget: **IMM entirely on CPU, single-threaded, no
accelerator offload.** 0.239 ms median (0.292 ms p99) against the 2 ms filter-level target
and the 20 ms full-cycle budget leaves 8x and ~65x headroom respectively, before accounting for
the filter being only one consumer of that 20 ms cycle. DLA is not needed at any of the
network sizes we deployed and carries a currently unquantifiable dispatch-latency risk that
could make a synchronous per-cycle round trip *slower* than doing everything on CPU. Reserve it
for a meaningfully larger future gain network. Per sec. 3.1/3.4, "meaningfully
larger" is not hypothetical: the reference paper's own default RKN sizing already
crosses into DLA territory on latency grounds alone.

## 4. Verdict

For this Jetson-Orin gimbal tracker (CPU/DLA/PVA only, GPU unavailable, 20 ms cycle), the
classical, well-tuned IMM is still the choice, but no longer on every axis. RKN is now close on
accuracy and ahead of IMM under hard maneuvers; vanilla KalmanNet is not in contention.

| | IMM | KalmanNet (vanilla) | RKN |
|---|---|---|---|
| Diverged runs, nominal | 0% | 86.3% (median step 408) | 0% |
| Position RMSE, nominal (m) | **1.17** | 74.57 (pre-divergence) | 1.49 |
| Position RMSE, high_maneuver (m) | 2.53 | 79.41 (pre-divergence, 94% diverged) | **1.89** |
| Calibration (high-MC ANEES vs. bounds 8.19-9.85) | **9.53 (inside)** | N/A (no covariance) | 15.02 (~1.5x over) |
| Coverage @68/95 | 0.702 / 0.862 | N/A | 0.553 / 0.732 |
| ANEES, high_noise (bounds 5.67-13.08) | 8.79 | N/A | 86.25 |
| Measured CPU latency (median) | 0.239 ms | 0.333 ms | 0.559 ms (301k params) / 2.759 ms (reference-default 7.85M params) |
| Params | 0 | 526,116 | 301,266 |

- **Accuracy**: IMM beats RKN by 1.27x nominal and by 1.2-1.6x on the noise regimes, and they tie
  on the recorded flights; RKN beats IMM by 25% on `high_maneuver`, where true turn rates run 20-60 deg/s past IMM's fixed
  25 deg/s CT modes. Vanilla KalmanNet diverges in 86% of nominal runs. IMM never diverges; the
  CV-only floor does in 29.5% of nominal runs.
- **Calibration**: IMM lands inside its own high-MC 95% ANEES interval, a usable, gateable
  uncertainty estimate, and stays consistent when the measurement noise is 4x nominal because it
  is told `R`. Vanilla KalmanNet produces no uncertainty estimate at all (structural, `m=9 > n=3`).
  RKN produces a real covariance, ~1.5x overconfident nominally, but it learned that covariance
  from nominal-noise training data and does not adapt: at `high_noise` its ANEES is 86 and its
  95% coverage 5%. For a track gate that is the disqualifying failure: a noisier-than-trained
  sensor would make RKN reject good measurements. It is better calibrated than IMM on
  `high_maneuver` (45.6 vs. 91.2), where both are overconfident.
- **Deployment cost**: IMM runs in 0.239 ms median on a single CPU core, no accelerator required,
  8x under the 2 ms target. KalmanNet and RKN, at the sizes we trained, also fit
  (0.333 ms, 0.559 ms), and so do UKN and CKN (0.644 ms, 1.146 ms), so deployment latency is not
  what rules them out. BKN at J = 20 is the one over budget (4.183 ms serial). RKN's reference-default
  paper sizing also shows that a faithfully-large version of this architecture
  family can blow the latency budget on CPU alone, before DLA/PVA are considered.
- **Later variants (sec. 2.5, 2.6, 1.7)**: none changes the verdict. UKN never diverges and tracks
  at 1.66 m nominal, but its high-MC ANEES is 50.8. CKN's covariance is positive definite and far
  too small (ANEES 14,339 at β = 0.25), at 6.75 m nominal. BKN inherits KalmanNet's divergence
  (87.4% nominal) and its coverage is 0.
- **Engineering cost**: the classical filters (EKF/UKF/IMM) total well under 500 lines and were
  validated (unit tests including five independent angle-wrap-boundary
  tests across every filter's code path and a from-first-principles verification of the Van Loan
  discretization against a textbook closed form) in a fraction of the effort spent getting either
  learned filter to train at all.

The earlier revision of this report called this a clean negative result for the learned filters.
Since then: the IMM tuner extends edge-hit axes and searches `sigma_a^CT` (#52); RKN's covariance
features are normalised per group (#86) and its gain lives in the line-of-sight frame (#94);
RKN's NEES is whitened through `S` like the classical filters' (#98); vanilla KalmanNet was
retrained on `train_knet.py`'s current defaults; and the divergence rule (#111) changed what a
diverging filter's RMSE counts. This rerun does not isolate which change moved RKN from 798 m to
1.49 m. The result is now a split one. Vanilla KalmanNet's from-scratch training still does not
produce a stable filter, which echoes the original paper's own documented BPTT-stability struggles
([#14](source-notes.md) sec. 1b). RKN trains to a stable filter within 1.6x of IMM everywhere and
better than it on the regime IMM's model bank was not built for; its weakness is a covariance that
does not follow a change in sensor noise.

**Conditions under which the verdict could flip, and what would need to change:**

1. **RKN calibration that survives a noise shift.** Accuracy is no longer the obstacle. Training
   across a range of measurement-noise levels, or feeding the known `R` into the covariance
   network, would need to bring RKN's `high_noise` ANEES inside its bounds without giving back the
   `high_maneuver` accuracy. If it did, RKN would trade ~0.3 m of nominal accuracy and ~0.3 ms
   of CPU for better behavior under unmodeled maneuvers.
2. **A motion-model family IMM cannot easily be extended to** (e.g. unknown, highly
   nonlinear maneuver classes with no small library of candidate models). IMM's mode bank is a
   limitation when a hand-designed model library doesn't cover the target's true behavior.
   The `high_maneuver` sweep (turn rates up to 2.4x the CT modes' nominal) is already enough for
   RKN to pass IMM on accuracy; a target whose real maneuver envelope looks like that regime
   rather than the nominal one is where this condition starts to bind.
3. **A much larger gain network** where DLA's per-invocation dispatch overhead becomes negligible
   relative to compute. Not the case at the sizes (301k-526k params) that fit our CPU budget
   today, but the reference paper's own *default* RKN size (7.85M params) is already in the
   range where this tradeoff starts to matter.

Condition 2 holds in part, on one regime, and condition 1 does not hold yet.
**Recommendation: ship the tuned IMM;** RKN is the candidate worth the next round of work.

## 5. What was not completed, and why (priority-order accounting)

Per the brief's stated priority order, items 1-4 (data generator + EKF/UKF + metrics with the
angle-wrap test; tuned IMM; vanilla KalmanNet; a consistency-corrected variant) plus evaluation are
the minimum acceptable deliverable, with the deployment mapping table required regardless.

- **Completed and evaluated (all four priority-order items 1-4, plus evaluation)**: data generator,
  EKF, UKF, IMM (tuned), vanilla KalmanNet (trained, evaluated, and reported as
  underperforming), **Recursive KalmanNet** (adapted from the official
  `github.com/ixblue/RecursiveKalmanNet` reference per [#14](source-notes.md) sec. 2, trained, and evaluated with
  the full metric suite it, unlike vanilla KalmanNet, can support), full metric suite
  (RMSE, ANEES with brief-specified N-scaled chi-square bounds, NIS, coverage) applied identically
  across EKF/UKF/IMM/RKN, robustness sweep across 4 shifted regimes for every filter including both
  learned ones, deployment mapping table with measured CPU latency (both learned filters, at two
  RKN sizes) and a documented DLA/PVA audit. A bug was caught and fixed along the way (IMM's
  heavy-tailed-regime NaN, sec. 2.4) and the whole evaluation re-run afterward for consistency.
- **Built after that, same metric suite**: Bayesian KalmanNet (sec. 1.7), Unscented KalmanNet
  (sec. 1.8, 2.5, #129) and Cholesky-KalmanNet (sec. 1.9, 2.6, #136).
- **Not completed**: the ensemble-of-KalmanNets stretch goal.
- **Also not completed**: the DLA-reconstructed feedforward gain-network variant and the
  quantization-vs-consistency (FP16/INT8) calibration-drift measurement (sec. 3.2). The latter is
  now meaningful (RKN produces a covariance to degrade) but presupposes a DLA-sized network, which
  RKN's trained variant here was not built to be (it was sized for CPU-latency competitiveness
  instead, sec. 1.6).

