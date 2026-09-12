# KalmanNet vs. IMM

Does a learned Kalman filter beat a well-tuned classical IMM for maneuvering-target tracking
with azimuth/elevation/range measurements, on an edge tracker where the GPU is reserved for the
vision pipeline?

Short answer: no. A tuned IMM wins on accuracy, on calibration, and on cost.

| | IMM | KalmanNet (vanilla) | RKN |
|---|---|---|---|
| Position RMSE, nominal (m) | **1.17** | 576.75 | 798.13 |
| High-MC ANEES (bounds 8.19-9.85) | 11.06 | N/A (no covariance) | 23.21 |
| Coverage @68/95 | 0.691 / 0.829 | N/A | 0.453 / 0.603 |
| Median CPU latency (ms) | 0.235 | 0.333 | 0.391 |
| Params | 0 | 526,116 | 301,266 |

Recursive KalmanNet's one genuine win: it produces a usable, better-calibrated covariance where
vanilla KalmanNet produces none (state dim 9 > measurement dim 3).

Full write-up, methodology, deployment analysis and the conditions that would flip the verdict:
[`REPORT.md`](REPORT.md). Literature extraction the implementations are grounded in:
[`NOTES.md`](NOTES.md).

![Five filters tracking one maneuvering target](docs/assets/kalmannet_motion_hud.gif)

## Running it

Everything runs in a CPU-only Docker image; `docker_run.sh` builds it and mounts this repo at
`/work`.

```
./docker_run.sh pytest -q          # 38 tests
./docker_run.sh python data/make_dataset.py
./docker_run.sh python tune_imm.py
./docker_run.sh python eval.py     # classical filters -> results/
./docker_run.sh python train_knet.py && ./docker_run.sh python eval_knet.py
./docker_run.sh python train_rkn.py  && ./docker_run.sh python eval_rkn.py
./docker_run.sh python bench_latency.py
./docker_run.sh python make_plots.py
./docker_run.sh python make_animation.py
```

Generated datasets, checkpoints and results are gitignored; regenerate them with the commands
above. Every results artifact records the host `GIT_COMMIT` it was produced at.
