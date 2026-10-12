# KalmanNet vs. IMM

Does a learned Kalman filter beat a well-tuned classical IMM for maneuvering-target tracking
with azimuth/elevation/range measurements, on an edge tracker where the GPU is reserved for the
vision pipeline?

Short answer: not yet. A tuned IMM still wins on nominal accuracy, on calibration and on cost, but
Recursive KalmanNet now comes within 1.3x of it and beats it under hard maneuvers.

| Filter | Pos RMSE nominal (m) | Pos RMSE high_maneuver (m) | Diverged (> 200 m), nominal | High-MC ANEES (bounds 8.19-9.85) | Coverage @68/95 | Median CPU latency (ms) | Params |
|---|---|---|---|---|---|---|---|
| EKF / UKF (CV only) | 55.89* | 71.12* | 29.5% | 906,770 | 0.036 / 0.047 | 0.025 / 0.115 | 0 |
| IMM (CV/CT+/CT-/CA) | **1.17** | 2.53 | 0% | **9.53** | 0.702 / 0.862 | 0.239 | 0 |
| KalmanNet (vanilla) | 74.57* | 79.41* | 86.3% | N/A (no covariance) | N/A | 0.333 | 526,116 |
| Recursive KalmanNet (RKN) | 1.49 | **1.89** | 0% | 15.02 | 0.553 / 0.732 | 0.559 | 301,266 |
| Bayesian KalmanNet (BKN, p = 0.2, J = 20) | 72.48* | 79.30* | 87.4% | 2.6e12 | 0.000 / 0.000 | 4.183† | 526,116 |
| Unscented KalmanNet (UKN) | 1.66 | 2.35 | 0% | 50.8 | 0.256 / 0.350 | 0.644† | 130,420 |
| Cholesky-KalmanNet (CKN, β = 0.25) | 6.75 | 5.49 | 2.2% | 14,339 | 0.003 / 0.007 | 1.146† | 1,751,535 |

\* RMSE over the steps before divergence only. † Measured in a separate run: BKN serially before
its 20 passes were batched, UKN and CKN on a loaded host (write-up sec. 3.1).

What keeps RKN out: its covariance is learned from nominal-noise data and does not follow a
noisier sensor (ANEES 86 at 4x measurement noise, where IMM, which is given the new `R`, stays
at 8.8). Vanilla KalmanNet
produces no covariance at all (state dim 9 > measurement dim 3) and diverges in most runs.
The three later variants do not change that. BKN's dropout ensemble inherits KalmanNet's
divergence and is over the 2 ms budget. UKN tracks in IMM's range but is 5x overconfident. CKN's
covariance is positive definite by construction and still far too small.

## On the site

- [Test ground](https://pavelguzenfeld.github.io/kalmannet-vs-imm/web/): fly a target in the browser and watch every filter track it live.
- [Write-up](https://pavelguzenfeld.github.io/kalmannet-vs-imm/docs/view.html?doc=write-up.md): methodology, every result, the deployment
  analysis and what would flip the verdict.
- [Reproduce it](https://pavelguzenfeld.github.io/kalmannet-vs-imm/docs/view.html?doc=reproduce.md): the pipeline step by step, serving the
  test ground locally, the repo layout and the checks.
- [Source notes and references](https://pavelguzenfeld.github.io/kalmannet-vs-imm/docs/view.html?doc=source-notes.md): the papers the filters
  are built from.

![Five filters tracking one maneuvering target](docs/assets/kalmannet_motion_hud.gif)

## Quick start

Everything runs in a CPU-only Docker image that `docker_run.sh` builds:

```
./docker_run.sh pytest -q
```

MIT licensed.
