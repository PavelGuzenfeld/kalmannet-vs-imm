# Model spec

Layer 0 of `model-vv.md`. Skeleton drafted from the code on 2026-09-13, not from intent.
Untagged lines were reconstructed from the code and confirmed on 2026-09-13. `(needs intent)` marks a question
only the author can answer. Tests cite a line as `# MS-n`; the mutation gate refuses a
citation to a tagged line, so a line becomes citable when its tag is removed and its text
confirmed. Numbers are stable; never renumber.

## State, frames, time

- MS-1 State x = [px py pz vx vy vz ax ay az] in m, m/s, m/s²; Cartesian, sensor at the origin, frame fixed and non-rotating
- MS-2 Measurement z = [azimuth elevation range] in rad, rad, m; azimuth = atan2(py, px), elevation = atan2(pz, hypot(px, py)); both angles wrapped to [-pi, pi)
- MS-3 Time base: fixed dt = 0.02 s (50 Hz); no irregular, late or missing measurements

## Phenomena modelled

- MS-4 CV: one Singer axis per Cartesian axis with tau = 0.05 s, sigma_a = 0.05 m/s², so acceleration is a fast-decaying nuisance state
- MS-5 CA: Singer / Ornstein–Uhlenbeck axis; tau = 5 s and sigma_a = 2 m/s² in `ca_model`, tau = 4 s and sigma_a = 3 m/s² inside the IMM
- MS-6 CT: exact horizontal coordinated-turn transition at a fixed nominal |omega| = 15 °/s, one mode per sign; the z axis is the Singer axis of MS-4 with the CT sigma_a
- MS-7 CV and CA (F, Q) come from van Loan discretisation of the continuous Singer axis; CT F is the closed form, CT Q is hand-typed as q = (sigma_a·dt)² on vx and vy only
- MS-8 IMM over [cv, ct+, ct−, ca]; TPM with p_stay = 0.97 and uniform off-diagonal; Blom–Bar-Shalom mixing, mode-matched EKF, combination; mode probabilities updated in log space
- MS-9 Single-model baselines EKF and UKF run the CV model of MS-4; EKF uses the Joseph-form update; UKF uses alpha = 1e-3, beta = 2, kappa = 0 and a circular mean for azimuth
- MS-10 KalmanNet and Recursive KalmanNet keep the CV f(·) and the h(·) of MS-2 and learn only the gain; they have no continuous-time model of their own, so Layer 3 does not apply to them and Layers 4 and 5 carry the load

## Noise assumptions

- MS-11 Measurement noise is Gaussian with diagonal R = diag(0.001², 0.001², 2.0²) at nominal; every filter is told this nominal R, also under the heavy-tailed regime
- MS-12 IMM process noise: CV sigma_a = 0.05, CT sigma_a = 0.3, CA sigma_a = 3.0 m/s² with tau = 4 s — are these the tuned values from `scripts/tune_imm.py` or placeholders? (needs intent)
- MS-13 The CT mode's process noise q = (0.3 m/s² · dt)² on vx, vy budgets an unmodelled lateral acceleration of 0.3 m/s², i.e. a turn-rate error |Δω| ≤ 0.3/v — 0.17 °/s at 100 m/s, 0.9 °/s at 20 m/s. It does not cover the 5–30 °/s spread of the true rate against the nominal 15 °/s; that mismatch is deliberate and is absorbed by IMM mode switching and the CA mode, not by the CT mode's Q.

## Effects deliberately neglected

- MS-14 Sensor motion and gimbal pose error: none; the sensor is a fixed origin — justification (needs intent)
- MS-15 Measurement latency, out-of-sequence and dropped measurements: none — justification (needs intent)
- MS-16 Gravity, aerodynamic drag, Earth curvature at up to 5 km range: none — order-of-magnitude margin (needs intent)
- MS-17 Range-dependent or correlated angular noise: none, R is constant and diagonal — justification (needs intent)
- MS-18 Turn rate is not a state; only the two fixed ±omega hypotheses of MS-6 — justification versus an unknown-omega CT mode (needs intent)

## Validity envelope

- MS-19 Initial slant range 500–5000 m, speed 20–250 m/s, initial elevation −10° to 60°, climb ±10°
- MS-20 Manoeuvre intensity: CT |omega| 5–30 °/s nominal and 20–60 °/s under high_maneuver; CA sigma_a 3 (8) m/s², tau 4 (2) s; segments of 1–5 s
- MS-21 Noise regimes: high_noise ×4 on angles and range, low_noise ×0.3 angles and ×0.25 range, heavy_tailed Student-t with 3 degrees of freedom
- MS-22 Trajectory length T = 750 steps = 15 s
- MS-23 Spherical singularities rho → 0 (overhead) and r → 0: the Jacobian floors rho² and r² at 1e-6, a 1 mm guard below which it is silently ~100× too small; h(·) itself is unguarded — is an overhead pass inside the envelope? (needs intent)

## Required accuracy

- MS-24 Position and velocity RMSE required per regime of MS-20 and MS-21 (needs intent)
- MS-25 Consistency: ANEES and ANIS inside the chi-square interval averaged over N Monte Carlo runs — N and the confidence level (needs intent)
- MS-26 Ranking: IMM beats the single-model EKF during a sustained turn; learned filters beat the EKF floor — by what margin (needs intent)

## Open flags

- MS-27 `data/trajectory_gen.py` imports `filters.motion_models`, so the truth generator shares its model with the filter; Layer 5 requires that it does not — accept as a known limitation or replace the generator (needs intent)
- MS-28 CT Q of MS-7 is not the van Loan Q of the same continuous model and has no position or acceleration terms — deliberate simplification, and what is the trigger to revisit? (needs intent)
- MS-29 UKF updates P as P − K S Kᵀ while EKF and IMM use the Joseph form — deliberate? (needs intent)

Flags MS-30 to MS-34 come from the blind pass of 2026-09-13 (code only, no spec; report on PR #17).

- MS-30 In the CT mode F[ax,ax] = F[ay,ay] = 1 with zero process noise and no coupling into position or velocity, so horizontal acceleration is frozen for the mode's dwell and mixed back into CV and CA by the interaction step — deliberate, or should CT carry the Singer axis for ax, ay too? (needs intent)
- MS-31 UKF alpha = 1e-3 with n = 9 puts the sigma points at ±0.003·sqrt(P) and Wm[0] ≈ −1e6, so the unscented transform evaluates h(·) as a finite-difference Jacobian and second-order effects are attenuated by alpha² = 1e-6 — is the UKF meant to differ from the EKF at all, and if so which alpha? (needs intent)
- MS-32 UKF propagates sigma points through F and adds Q to P afterwards, so the measurement sigma points, S and Pxz omit the H Q Hᵀ contribution; negligible at CV's sigma_a = 0.05 m/s², not at larger Q — deliberate simplification, and what is the trigger to revisit? (needs intent)
- MS-33 UKF adds 1e-12·I before the Cholesky factorisation; at position variances of 1e4 m² that is below double resolution and regularises nothing — is a scale-relative jitter required? (needs intent)
- MS-34 `wrap_angle` yields [−pi, pi) while the docstrings of `innovation` and `KalmanNetFilter` claim (−pi, pi]; elevation is also wrapped as if circular although it lies in [−pi/2, pi/2] by construction — which convention is the spec's, and is the elevation wrap intended? (needs intent)
