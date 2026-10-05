import { toWorker } from "./protocol.js";

const FLAGS = ["track_loss", "overconfidence", "raw_error"];
const THRESHOLD_INPUTS = ["innovation_gate_nis", "nees_upper", "overconfident_steps", "raw_error_m"];
const INTEGER_THRESHOLDS = new Set(["overconfident_steps"]);
const ENVELOPE_SLIDERS = {
  min_speed_m_s: { toSlider: (value) => value, fromSlider: (value) => value },
  max_speed_m_s: { toSlider: (value) => value, fromSlider: (value) => value },
  max_load_factor: { toSlider: (value) => value, fromSlider: (value) => value },
  max_roll_rate_rad_s: { id: "max_roll_rate_deg_s", toSlider: (rad) => (rad * 180) / Math.PI, fromSlider: (deg) => (deg * Math.PI) / 180 },
};
const STEERING_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "KeyW", "KeyS"]);
const MAX_STEPS_PER_MESSAGE = 10;
const PUSH_LOAD_FACTOR = -1;
const CRUISE_THROTTLE = 0.6;
const THROTTLE_RATE_PER_S = 0.5;
const STICK_DEADZONE = 0.08;
const STANDARD_LEFT_TRIGGER = 6;
const STANDARD_RIGHT_TRIGGER = 7;
const STEP_COST_WINDOW = 100;
const STEP_BUDGET_MS = 20;
const TRAIL_LENGTH = 3000;
const PLAN_MARGIN = 0.1;
const MIN_PLAN_SPAN_M = 200;
const TRAIL_COLORS = { truth: "#000000", EKF: "#d62728", UKF: "#ff7f0e", IMM: "#2ca02c", KalmanNet: "#1f77b4", RKN: "#9467bd" };

const byId = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const fixed = (value, digits) => (value === null || value === undefined ? "—" : value.toFixed(digits));

const worker = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
const heldKeys = new Set();
let options = null;
let envelope = null;
let flightSetup = null;
let script = null;
let stepsSent = 0;
let speed = 0;
let owedSteps = 0;
let awaitingWorker = true;
let lastTick = performance.now();
let throttle = CRUISE_THROTTLE;
let latestControl = [0, 1, CRUISE_THROTTLE];
let trails = {};
let stepCosts = [];

const setStatus = (text) => (byId("status").textContent = text);

function sliderOf(key) {
  return byId(ENVELOPE_SLIDERS[key].id ?? key);
}

function showEnvelope() {
  for (const [key, slider] of Object.entries(ENVELOPE_SLIDERS)) {
    const input = sliderOf(key);
    input.value = slider.toSlider(envelope[key]);
    input.nextElementSibling.textContent = Number(input.value).toFixed(1);
  }
}

function applyScenarioToUi(scenario) {
  envelope = { ...scenario.envelope };
  showEnvelope();
  byId("seed").value = scenario.seed;
  byId("noise_regime").value = scenario.noise_regime;
  for (const key of THRESHOLD_INPUTS) byId(key).value = String(scenario.thresholds[key]);
}

function populate() {
  for (const name of Object.keys(options.presets)) byId("preset").add(new Option(name, name));
  for (const name of options.noise_regimes) byId("noise_regime").add(new Option(name, name));
  byId("preset").value = options.default_preset;
  applyScenarioToUi({
    envelope: options.presets[options.default_preset],
    seed: Math.floor(Math.random() * 2 ** 31),
    noise_regime: options.noise_regimes[0],
    thresholds: options.default_thresholds,
  });
}

function thresholds() {
  return Object.fromEntries(
    THRESHOLD_INPUTS.map((key) => [key, INTEGER_THRESHOLDS.has(key) ? parseInt(byId(key).value, 10) : parseFloat(byId(key).value)]),
  );
}

function startFlight(scriptControls) {
  flightSetup = {
    seed: parseInt(byId("seed").value, 10),
    envelope: { ...envelope },
    noise_regime: byId("noise_regime").value,
    thresholds: thresholds(),
  };
  script = scriptControls;
  stepsSent = 0;
  owedSteps = 0;
  trails = {};
  stepCosts = [];
  speed = script ? 1 : 0;
  awaitingWorker = true;
  worker.postMessage(toWorker.start(flightSetup));
}

async function startFromUrlOrUi() {
  const scenarioUrl = new URLSearchParams(location.search).get("scenario");
  if (!scenarioUrl) return startFlight(null);
  const scenario = await (await fetch(scenarioUrl)).json();
  applyScenarioToUi(scenario);
  startFlight(scenario.controls);
}

function stickInput() {
  const pad = [...(navigator.getGamepads?.() ?? [])].find((candidate) => candidate?.mapping === "standard");
  const axis = (index) => (pad && Math.abs(pad.axes[index]) > STICK_DEADZONE ? pad.axes[index] : 0);
  const trigger = (index) => pad?.buttons[index]?.value ?? 0;
  const key = (code) => (heldKeys.has(code) ? 1 : 0);
  return {
    roll: clamp(axis(0) + key("ArrowRight") - key("ArrowLeft"), -1, 1),
    pull: clamp(axis(1) + key("ArrowDown") - key("ArrowUp"), -1, 1),
    throttleRate: trigger(STANDARD_RIGHT_TRIGGER) - trigger(STANDARD_LEFT_TRIGGER) + key("KeyW") - key("KeyS"),
  };
}

function readControl(elapsedS) {
  const stick = stickInput();
  const { max_roll_rate_rad_s, max_load_factor } = flightSetup.envelope;
  throttle = clamp(throttle + stick.throttleRate * THROTTLE_RATE_PER_S * elapsedS, 0, 1);
  const loadFactor = stick.pull >= 0 ? 1 + stick.pull * (max_load_factor - 1) : 1 + stick.pull * (1 - PUSH_LOAD_FACTOR);
  return [stick.roll * max_roll_rate_rad_s, loadFactor, throttle];
}

function advance(steps) {
  const controls = script ? script.slice(stepsSent, stepsSent + steps) : Array(steps).fill(latestControl);
  if (!controls.length) {
    speed = 0;
    return;
  }
  stepsSent += controls.length;
  awaitingWorker = true;
  worker.postMessage(toWorker.advance(controls));
}

function tick(now) {
  const elapsedS = (now - lastTick) / 1000;
  lastTick = now;
  if (flightSetup) {
    latestControl = readControl(elapsedS);
    const [rollRate, loadFactor, throttleNow] = latestControl;
    byId("controls").textContent =
      `roll rate ${((rollRate * 180) / Math.PI).toFixed(0)} °/s, load factor ${loadFactor.toFixed(2)} g, throttle ${throttleNow.toFixed(2)}`;
    owedSteps = speed > 0 ? Math.min(owedSteps + (elapsedS * speed) / options.dt_s, MAX_STEPS_PER_MESSAGE) : 0;
    const steps = Math.floor(owedSteps);
    if (steps > 0 && !awaitingWorker) {
      owedSteps -= steps;
      advance(steps);
    }
  }
  requestAnimationFrame(tick);
}

function renderReadouts(frame) {
  byId("readouts").replaceChildren(
    ...Object.entries(frame.flags).map(([name, flags]) => {
      const row = document.createElement("tr");
      const readout = frame.filters[name] ?? {};
      const cells = [name, fixed(readout.error_m, 1), fixed(readout.nees, 2), fixed(readout.nis, 2), ...FLAGS.map((flag) => flags[flag])];
      row.replaceChildren(
        ...cells.map((text, column) => {
          const cell = document.createElement("td");
          cell.textContent = text;
          if (column >= 4) Object.assign(cell.dataset, { filter: name, flag: FLAGS[column - 4] });
          return cell;
        }),
      );
      return row;
    }),
  );
}

function renderPlan(frame) {
  const positions = { truth: frame.truth_m, ...Object.fromEntries(Object.entries(frame.filters).map(([name, readout]) => [name, readout.estimate_m])) };
  for (const [name, [x, y]] of Object.entries(positions)) {
    trails[name] = [...(trails[name] ?? []).slice(1 - TRAIL_LENGTH), [x, y]];
  }
  const points = Object.values(trails).flat();
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), MIN_PLAN_SPAN_M) * (1 + 2 * PLAN_MARGIN);
  const centre = [(Math.max(...xs) + Math.min(...xs)) / 2, (Math.max(...ys) + Math.min(...ys)) / 2];
  const canvas = byId("plan");
  const context = canvas.getContext("2d");
  const toCanvas = ([x, y]) => [((x - centre[0]) / span + 0.5) * canvas.width, (0.5 - (y - centre[1]) / span) * canvas.height];
  context.clearRect(0, 0, canvas.width, canvas.height);
  for (const [name, trail] of Object.entries(trails)) {
    context.strokeStyle = TRAIL_COLORS[name];
    context.beginPath();
    trail.map(toCanvas).forEach(([x, y], index) => (index ? context.lineTo(x, y) : context.moveTo(x, y)));
    context.stroke();
    context.fillStyle = TRAIL_COLORS[name];
    context.fillText(name, ...toCanvas(trail.at(-1)));
  }
  context.fillStyle = "#666666";
  context.fillText(`plan view, x east, y north, ${span.toFixed(0)} m across`, 8, canvas.height - 8);
}

function renderClock(frame) {
  const cost = stepCosts.length ? `${median(stepCosts).toFixed(2)} ms per step (budget ${STEP_BUDGET_MS} ms)` : "";
  byId("clock").textContent = `step ${frame.step}, t = ${(frame.step * options.dt_s).toFixed(2)} s, ${speed}×, ${cost}`;
}

const replies = {
  ready({ options: workerOptions }) {
    options = workerOptions;
    populate();
    startFromUrlOrUi();
  },
  started() {
    awaitingWorker = false;
    setStatus(script ? `flying the scripted ${script.length} steps` : "ready: pick a speed to fly");
  },
  frame({ frame, stepCostMs }) {
    awaitingWorker = false;
    if (stepCostMs !== null) stepCosts = [...stepCosts.slice(1 - STEP_COST_WINDOW), stepCostMs];
    if (script && stepsSent === script.length) setStatus(`script finished at step ${stepsSent}`);
    renderReadouts(frame);
    renderPlan(frame);
    renderClock(frame);
  },
  scenario({ json }) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    link.download = `scenario-seed${flightSetup.seed}-step${stepsSent}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  },
  failed({ message }) {
    awaitingWorker = false;
    speed = 0;
    setStatus(`worker error: ${message}`);
  },
};

worker.onmessage = ({ data }) => replies[data.kind](data);
worker.postMessage(toWorker.load());

byId("preset").addEventListener("change", () => {
  envelope = { ...options.presets[byId("preset").value] };
  showEnvelope();
});
for (const [key, slider] of Object.entries(ENVELOPE_SLIDERS)) {
  sliderOf(key).addEventListener("input", (event) => {
    envelope[key] = slider.fromSlider(Number(event.target.value));
    event.target.nextElementSibling.textContent = Number(event.target.value).toFixed(1);
  });
}
for (const key of THRESHOLD_INPUTS) {
  byId(key).addEventListener("change", () => flightSetup && worker.postMessage(toWorker.setThresholds(thresholds())));
}
for (const button of document.querySelectorAll("[data-speed]")) {
  button.addEventListener("click", () => (speed = Number(button.dataset.speed)));
}
byId("single_step").addEventListener("click", () => {
  speed = 0;
  if (!awaitingWorker) advance(1);
});
byId("new_flight").addEventListener("click", () => startFlight(null));
byId("export").addEventListener("click", () => stepsSent && worker.postMessage(toWorker.exportScenario()));
addEventListener("keydown", (event) => {
  if (!STEERING_KEYS.has(event.code) || event.target.matches("input, select")) return;
  heldKeys.add(event.code);
  event.preventDefault();
});
addEventListener("keyup", (event) => heldKeys.delete(event.code));
requestAnimationFrame(tick);
