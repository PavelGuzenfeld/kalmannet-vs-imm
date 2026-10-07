import { PYODIDE_URL, fromWorker } from "./protocol.js";

const REPO_ROOT = new URL("../", import.meta.url);
const SITE_DIR = "/repo";
const PYTHON_SOURCES = [
  "data/__init__.py",
  "data/config.py",
  "data/fixed_wing.py",
  "data/trajectory_gen.py",
  "filters/__init__.py",
  "filters/state.py",
  "filters/discretize.py",
  "filters/motion_models.py",
  "filters/measurement_model.py",
  "filters/ekf.py",
  "filters/ukf.py",
  "filters/imm.py",
  "filters/flags.py",
  "filters/knet/__init__.py",
  "filters/knet/numpy_forward.py",
  "filters/metrics.py",
  "filters/plot_data.py",
  "scripts/eval.py",
  "web/holds.py",
  "web/live_flight.py",
  "web/manoeuvres.py",
];

async function fetched(path) {
  const response = await fetch(new URL(path, REPO_ROOT));
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response;
}

function writeSiteFile(pyodide, path, content) {
  pyodide.FS.mkdirTree(`${SITE_DIR}/${path.slice(0, path.lastIndexOf("/"))}`);
  pyodide.FS.writeFile(`${SITE_DIR}/${path}`, content);
}

async function boot() {
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  await pyodide.loadPackage(["numpy", "scipy"]);
  for (const path of PYTHON_SOURCES) writeSiteFile(pyodide, path, await (await fetched(path)).text());
  pyodide.runPython(`import sys; sys.path.insert(0, "${SITE_DIR}")`);
  const liveFlight = pyodide.pyimport("web.live_flight");
  for (const path of Object.values(toJs(liveFlight.page_options()).learned_weights)) {
    const response = await fetch(new URL(path, REPO_ROOT));
    if (response.ok) writeSiteFile(pyodide, path, new Uint8Array(await response.arrayBuffer()));
  }
  const immBest = (await (await fetched("web/imm_tuning.json")).json()).best;
  const weightsRoot = pyodide.pyimport("pathlib").Path(SITE_DIR);
  return { pyodide, liveFlight, immBest, weightsRoot };
}

const python = boot();
let flight = null;

function toJs(proxy) {
  const value = proxy.toJs({ dict_converter: Object.fromEntries });
  proxy.destroy();
  return value;
}

const handlers = {
  async load() {
    const { liveFlight } = await python;
    return fromWorker.ready(toJs(liveFlight.page_options()));
  },
  async start({ setup }) {
    const { pyodide, liveFlight, immBest, weightsRoot } = await python;
    flight?.destroy();
    flight = liveFlight.LiveFlight(pyodide.toPy(setup), pyodide.toPy(immBest), weightsRoot);
    return fromWorker.started();
  },
  async advance({ controls, manualAxes, envelope }) {
    const { pyodide } = await python;
    if (envelope) {
      const fields = pyodide.toPy(envelope);
      flight.set_envelope(fields);
      fields.destroy();
    }
    const begin = performance.now();
    for (const control of controls) flight.advance(control, manualAxes);
    const stepCostMs = (performance.now() - begin) / controls.length;
    return fromWorker.frame(toJs(flight.frame()), stepCostMs);
  },
  async manoeuvre({ name }) {
    if (name) flight.start_manoeuvre(name);
    else flight.cancel_manoeuvre();
    return null;
  },
  async setThresholds({ thresholds }) {
    const { pyodide } = await python;
    flight.thresholds = pyodide.toPy(thresholds);
    return flight.controls.length ? fromWorker.frame(toJs(flight.frame()), null) : null;
  },
  async toggleHold({ name }) {
    flight?.toggle_hold(name);
    return null;
  },
  async exportScenario() {
    return fromWorker.scenario(flight.scenario_json());
  },
};

self.onmessage = async ({ data }) => {
  try {
    const reply = await handlers[data.kind](data);
    if (reply) self.postMessage(reply);
  } catch (error) {
    self.postMessage(fromWorker.failed(String(error)));
  }
};
