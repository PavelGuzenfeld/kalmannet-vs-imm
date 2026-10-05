export const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

export const toWorker = Object.freeze({
  load: () => ({ kind: "load" }),
  start: (setup) => ({ kind: "start", setup }),
  advance: (controls) => ({ kind: "advance", controls }),
  setThresholds: (thresholds) => ({ kind: "setThresholds", thresholds }),
  exportScenario: () => ({ kind: "exportScenario" }),
});

export const fromWorker = Object.freeze({
  ready: (options) => ({ kind: "ready", options }),
  started: () => ({ kind: "started" }),
  frame: (frame, stepCostMs) => ({ kind: "frame", frame, stepCostMs }),
  scenario: (json) => ({ kind: "scenario", json }),
  failed: (message) => ({ kind: "failed", message }),
});
