# KalmanNet vs. IMM

Does a learned Kalman filter beat a well-tuned classical IMM for maneuvering-target tracking
with azimuth/elevation/range measurements, on an edge tracker where the GPU is reserved for the
vision pipeline?

Short answer: not yet. A tuned IMM still wins on accuracy, on honest uncertainty and on cost.
Recursive KalmanNet comes close and beats it on hard turns, but its uncertainty does not follow a
noisier sensor. Every number is in the write-up.

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
