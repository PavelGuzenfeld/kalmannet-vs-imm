# Reproducing the benchmark

How to rerun every number in [the write-up](write-up.md), serve the test ground locally, and
find your way around the repo.

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
