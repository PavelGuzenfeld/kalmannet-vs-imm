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

Full write-up, methodology, deployment analysis and the conditions that would flip the verdict:
[the write-up](https://pavelguzenfeld.github.io/kalmannet-vs-imm/docs/view.html?doc=write-up.md). Literature extraction the implementations are grounded in:
[the source notes](https://pavelguzenfeld.github.io/kalmannet-vs-imm/docs/view.html?doc=source-notes.md).

![Five filters tracking one maneuvering target](docs/assets/kalmannet_motion_hud.gif)

## Running it

Everything runs in a CPU-only Docker image. `docker_run.sh` builds it, mounts this repo at
`/work` and runs the command you give it. Run the steps in order; each one reads the previous
step's output from its default path. Datasets, checkpoints and results are gitignored, and every
results artifact records the `GIT_COMMIT` it was produced at.

**1. Tests**

```
./docker_run.sh pytest -q   # 724 tests
```

**2. Data.** 1,000 train, 200 validation and 200 test trajectories, the four shifted test
regimes, and the high-Monte-Carlo subset used for calibration.

```
./docker_run.sh python scripts/make_dataset.py   # -> datasets/
```

**3. Tune IMM** on the validation set.

```
./docker_run.sh python scripts/tune_imm.py       # -> results/imm_tuning.json
```

**4. KalmanNet and RKN.** Each trains, then scores itself on the test sets.

```
./docker_run.sh python scripts/train_knet.py && ./docker_run.sh python scripts/eval_knet.py
./docker_run.sh python scripts/train_rkn.py  && ./docker_run.sh python scripts/eval_rkn.py
```

**5. BKN, UKN and CKN.** Each one has a hyperparameter to choose. Train one model per candidate
value, export each to numpy weights, let the tuning script score them on the validation set,
then copy the pick to `results/<name>/`, where the evaluator looks for it.

| Filter | Trained by | Hyperparameter | Candidates | Copied to `results/<name>/` |
|---|---|---|---|---|
| BKN | `train_knet.py --dropout-p` | dropout p (and J, picked by the tuner) | 0.05, 0.1, 0.2 | p = 0.2 |
| UKN | `train_ukn.py --gain-scale` | gain scale c_K | 0.1, 0.3, 1.0 | c_K = 0.3 |
| CKN | `train_ckn.py --beta` | calibration weight β | 0.01, 0.05, 0.25 | β = 0.05, the paper's value; the tuner picks 0.25, evaluated in step 6 |

```
for p in 0.05 0.1 0.2; do
  ./docker_run.sh python scripts/train_knet.py --dropout-p $p --out results/bkn_p$p
  ./docker_run.sh python scripts/export_weights.py results/bkn_p$p/best.pt
done
./docker_run.sh python scripts/tune_bkn.py results/bkn_p*/best.npz
cp -r results/bkn_p0.2 results/bkn

for c in 0.1 0.3 1.0; do
  ./docker_run.sh python scripts/train_ukn.py --gain-scale $c --out results/ukn_ck$c
  ./docker_run.sh python scripts/export_weights.py results/ukn_ck$c/best.pt
done
./docker_run.sh python scripts/tune_ukn.py results/ukn_ck*/best.npz
cp -r results/ukn_ck0.3 results/ukn

for b in 0.01 0.05 0.25; do
  ./docker_run.sh python scripts/train_ckn.py --beta $b --out results/ckn_b$b
  ./docker_run.sh python scripts/export_weights.py results/ckn_b$b/best.pt
done
./docker_run.sh python scripts/tune_ckn.py results/ckn_b*/best.npz
cp -r results/ckn_b0.05 results/ckn
```

**6. Evaluate.** `eval.py` scores EKF, UKF and IMM, plus BKN, UKN and CKN for each
`results/<name>/best.npz` that exists. The second run scores the tuned CKN β.

```
./docker_run.sh python scripts/eval.py   # -> results/results_table.csv, results/anees_highmc.json
./docker_run.sh python scripts/eval.py --ckn-weights results/ckn_b0.25/best.npz --out results/eval_ckn_tuned
```

**7. Latency, plots and the animation**

```
./docker_run.sh python scripts/bench_latency.py --knet-checkpoint results/knet_vanilla/best.pt \
    --rkn-checkpoint results/rkn/best.pt --bench-rkn-reference-default --bkn-weights results/bkn/best.npz \
    --ukn-weights results/ukn/best.npz --ckn-weights results/ckn/best.npz
./docker_run.sh python scripts/make_plots.py       # -> results/plots/
./docker_run.sh python scripts/make_animation.py   # -> results/video/
```

## Test ground

`web/index.html` flies a fixed-wing target in the browser and runs every filter on it live. The
filters are this repo's own Python, loaded through Pyodide. You steer with keyboard or gamepad,
or hand over to heading, altitude, speed and bank holds, six named manoeuvres, or an evade mode.
Envelope presets, any of the five noise regimes and the sensor position are set on the page,
next to its controls table. Track-loss, overconfidence and raw-error flags fire per filter.

Serve it locally once the pipeline has written `results/`; each learned filter loads its
`results/<name>/best.npz`:

```
./docker_run.sh python scripts/build_site.py --tag local   # -> _site/
docker run --rm -p 8765:8765 -v "$PWD/_site:/site:ro" -w /site kalmannet-bench:cpu python -m http.server 8765
```

Then open `http://localhost:8765/web/`. Pyodide comes from jsDelivr, so the browser needs network
access. The page's flags are a preview: export a flight as JSON and
`./docker_run.sh python scripts/replay.py <file>` replays it natively. `data/scenarios/` holds the
two recorded flights the evaluation's scenarios row uses.

## Layout

```
scripts/              entrypoints, one per pipeline stage; run from the repo root
filters/              the library: EKF, UKF, IMM, motion/measurement models, metrics, flags
filters/knet/         KalmanNet, RKN, UKN and CKN (torch), and the numpy forward pass BKN and the page use
data/                 the trajectory generator, the fixed-wing flight model and regime definitions
data/scenarios/       the two recorded test-ground flights
web/                  the browser test ground
tests/                unit tests -- `./docker_run.sh pytest -q`
docs/                 the write-up, the source notes and the README animation
docker/               the CPU-only image both CI and docker_run.sh use

datasets/             generated trajectories (gitignored)
results/              every artifact the pipeline writes, plots included (gitignored)
```

## Checks

CI (`.github/workflows/ci.yml`) lints with ruff, runs the unit tests in the bench image, and
smokes the pipeline above at toy sizes on its default paths. The unit tests import the packages
directly and so never execute an entrypoint.

Pre-commit adds the diff-scoped mutation gate over `filters/` and `data/`:

```
pip install pre-commit && pre-commit install
```

## References

What the filters and the comparison are built on (extracted in [the source notes](https://pavelguzenfeld.github.io/kalmannet-vs-imm/docs/view.html?doc=source-notes.md)):

1. H. A. P. Blom, Y. Bar-Shalom, "The interacting multiple model algorithm for systems with Markovian switching coefficients," *IEEE Trans. Automatic Control* 33(8), 780–783, 1988. [doi:10.1109/9.1299](https://doi.org/10.1109/9.1299)
2. Y. Bar-Shalom, X.-R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001. [doi:10.1002/0471221279](https://doi.org/10.1002/0471221279)
3. G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun, Y. C. Eldar, "KalmanNet: Neural network aided Kalman filtering for partially known dynamics," *IEEE Trans. Signal Processing* 70, 1532–1547, 2022. [doi:10.1109/TSP.2022.3158588](https://doi.org/10.1109/TSP.2022.3158588), [arXiv:2107.10043](https://arxiv.org/abs/2107.10043)
4. H. Mortada, C. Falcon, Y. Kahil, M. Clavaud, J.-P. Michel, "Recursive KalmanNet: Deep learning-augmented Kalman filtering for state estimation with consistent uncertainty quantification," *EUSIPCO*, 885–889, 2025. [doi:10.23919/EUSIPCO63237.2025.11226444](https://doi.org/10.23919/EUSIPCO63237.2025.11226444), [arXiv:2506.11639](https://arxiv.org/abs/2506.11639)
5. M. Ko, A. Shafieezadeh, "Cholesky-KalmanNet: Model-based deep learning with positive definite error covariance structure," *IEEE Signal Processing Letters* 32, 326–330, 2025. [doi:10.1109/LSP.2024.3519265](https://doi.org/10.1109/LSP.2024.3519265)
6. Y. Dahan, G. Revach, J. Duník, N. Shlezinger, "Bayesian KalmanNet: Quantifying uncertainty in deep learning augmented Kalman filter," *IEEE Trans. Signal Processing* 73, 2558–2573, 2025. [doi:10.1109/TSP.2025.3581703](https://doi.org/10.1109/TSP.2025.3581703), [arXiv:2309.03058](https://arxiv.org/abs/2309.03058)
7. M. Mari, L. Snidaro, "Ensemble of KalmanNets for maneuvering target tracking," *FUSION*, 2024. [doi:10.23919/FUSION59988.2024.10706253](https://doi.org/10.23919/FUSION59988.2024.10706253); and "Ensemble of KalmanNets with innovation-based attention for robust target tracking," *Information Fusion* 127, 103777, 2026. [doi:10.1016/j.inffus.2025.103777](https://doi.org/10.1016/j.inffus.2025.103777)
8. N. Shlezinger, G. Revach, A. Ghosh, S. Chatterjee, S. Tang, T. Imbiriba, J. Duník, O. Straka, P. Closas, Y. C. Eldar, "Artificial intelligence-aided Kalman filters: AI-augmented designs for Kalman-type algorithms," *IEEE Signal Processing Magazine* 42(3), 52–76, 2025. [doi:10.1109/MSP.2025.3569395](https://doi.org/10.1109/MSP.2025.3569395), [arXiv:2410.12289](https://arxiv.org/abs/2410.12289)
9. M. Ko, A. Shafieezadeh, "Unscented KalmanNet: Structure-preserving deep learning with calibrated posterior uncertainty under incomplete physics and unknown noise," 2026. [arXiv:2608.04201](https://arxiv.org/abs/2608.04201)

2026 work on the same question, not yet folded into the benchmark (read-up and verdicts in #128):

- Y. Zhao, C. Yang, L. Gao, Y. Tian, T. Yuan, "IMMNet: Hybrid fusion of model-based and data-driven approaches for maneuvering target tracking," 2026. [arXiv:2607.13573](https://arxiv.org/abs/2607.13573) — IMM structure with learned components; the closest direct competitor.
- Z. Li, Y. Guo, J. Wang, "IMMnet: Learned multiple model fusion filter for intelligent vehicle pose estimation," *IEEE Sensors J.* 26(4), 5769–5776, 2026. [doi:10.1109/JSEN.2025.3646621](https://doi.org/10.1109/JSEN.2025.3646621) — learns the IMM fusion step.
- C.-B. Lim, D.-H. Paek, S.-H. Kong, "IMM-based multiple object tracking using a state prediction neural network," 2026. [arXiv:2609.13307](https://arxiv.org/abs/2609.13307) — adds a learned predictor as an extra IMM mode.
- W. Zhang, X. Ni, N. Shlezinger, Z. Wang, "Change-aware self-adaptive AI-aided Kalman filters with neural change point detection," 2026. [arXiv:2607.13387](https://arxiv.org/abs/2607.13387) — online change detection, the role IMM mode probabilities play.
- O. Cohen, N. Shlezinger, T. Routtenberg, "EM-KalmanNet: Learned expectation-maximization for adaptive tracking in partially known, block-wise time-varying state-space models," 2026. [arXiv:2608.24404](https://arxiv.org/abs/2608.24404) — regime switching, close to IMM's assumption.
- A. Mehrfard, B. Duraisamy, S. Haag, F. Geiss, M. Mählisch, "Adaptive learned state estimation based on KalmanNet," 2026. [arXiv:2604.02441v1](https://arxiv.org/abs/2604.02441v1) (v1; v2 withdrawn) — scenario-adaptive KalmanNet with covariance estimation.
- J. Song, W. Mei, Y. Xu, Q. Fu, R. Kou, L. Bu, Y. Long, "Motion estimation for multi-object tracking using KalmanNet with semantic-independent encoding," *Information Fusion* 136, 104513, 2026. [doi:10.1016/j.inffus.2026.104513](https://doi.org/10.1016/j.inffus.2026.104513) — KalmanNet under mismatched KF parameters in MOT.
