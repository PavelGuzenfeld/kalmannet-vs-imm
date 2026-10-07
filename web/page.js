import { toWorker } from "./protocol.js";
import { DIVERGED_LABEL, FILTER_COLORS, TRUTH_COLOR, planExtent, readoutText, renderTimePlots } from "./plots.js";
import { createScene } from "./scene.js";

const FLAGS = ["track_loss", "overconfidence", "raw_error"];
const THRESHOLD_INPUTS = ["innovation_gate_nis", "nees_upper", "overconfident_steps", "raw_error_m"];
const INTEGER_THRESHOLDS = new Set(["overconfident_steps"]);
const ENVELOPE_SLIDERS = {
  min_speed_m_s: { toSlider: (value) => value, fromSlider: (value) => value },
  max_speed_m_s: { toSlider: (value) => value, fromSlider: (value) => value },
  max_load_factor: { toSlider: (value) => value, fromSlider: (value) => value },
  max_side_load_factor: { toSlider: (value) => value, fromSlider: (value) => value },
  max_negative_load_factor: { toSlider: (value) => value, fromSlider: (value) => value },
  brake_drag_multiplier: { toSlider: (value) => value, fromSlider: (value) => value },
  afterburner_thrust_multiplier: { toSlider: (value) => value, fromSlider: (value) => value },
  max_roll_rate_rad_s: { id: "max_roll_rate_deg_s", toSlider: (rad) => (rad * 180) / Math.PI, fromSlider: (deg) => (deg * Math.PI) / 180 },
};
const STEERING_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "KeyW", "KeyS", "KeyA", "KeyD", "Space", "ShiftLeft", "ShiftRight"]);
const MAX_STEPS_PER_MESSAGE = 10;
const CRUISE_THROTTLE = 0.6;
const THROTTLE_RATE_PER_S = 0.5;
const KEY_RAMP_PER_S = 0.5;
const STICK_DEADZONE = 0.08;
const STANDARD_LEFT_TRIGGER = 6;
const STANDARD_RIGHT_TRIGGER = 7;
const STANDARD_A = 0;
const STANDARD_B = 1;
const STANDARD_LEFT_BUMPER = 4;
const STANDARD_RIGHT_BUMPER = 5;
const LEGACY_ENVELOPE = { max_side_load_factor: 0, brake_drag_multiplier: 1, afterburner_thrust_multiplier: 1 };
const ROW_ROLL_RATE = 0;
const ROW_LOAD_FACTOR = 1;
const ROW_THROTTLE = 2;
const ROW_SPEED_BRAKE = 4;
const ROW_AFTERBURNER = 5;
const HOLD_KEYS = { KeyH: "heading", KeyJ: "altitude", KeyK: "speed", KeyL: "bank" };
const STEP_COST_WINDOW = 100;
const STEP_BUDGET_MS = 20;
const TRAIL_LENGTH = 3000;
const TRAIL_COLORS = { truth: TRUTH_COLOR, ...FILTER_COLORS };

const byId = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const divergedFilters = (frame) => new Set(Object.keys(frame.filters).filter((name) => frame.filters[name].diverged));

const worker = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
const heldKeys = new Set();
let options = null;
let envelope = null;
let flightSetup = null;
let script = null;
let stepsSent = 0;
let speed = 0;
let flying = false;
let owedSteps = 0;
let awaitingWorker = true;
let lastTick = performance.now();
let throttle = CRUISE_THROTTLE;
let latestControl = [0, 1, CRUISE_THROTTLE, 0, 0, 0];
const keyAxes = { roll: 0, pull: 0, rudder: 0 };
let steering = { roll: 0, pull: 0, rudder: 0, brake: 0, afterburner: 0, manualAxes: [] };
let trails = {};
let stepCosts = [];
let history = [];
let scene = null;

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
  envelope = { ...LEGACY_ENVELOPE, ...scenario.envelope };
  envelope.max_negative_load_factor ??= -envelope.max_load_factor;
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

function startFlight(scriptControls, recordedStart) {
  flightSetup = {
    seed: parseInt(byId("seed").value, 10),
    envelope: { ...envelope },
    noise_regime: byId("noise_regime").value,
    thresholds: thresholds(),
    ...(recordedStart && { start: recordedStart }),
  };
  script = scriptControls;
  stepsSent = 0;
  owedSteps = 0;
  trails = {};
  history = [];
  scene?.reset();
  stepCosts = [];
  speed = 1;
  flying = true;
  awaitingWorker = true;
  worker.postMessage(toWorker.start(flightSetup));
}

async function startFromUrlOrUi() {
  const scenarioUrl = new URLSearchParams(location.search).get("scenario");
  if (!scenarioUrl) return setStatus("paused: set the parameters and press Start");
  const scenario = await (await fetch(scenarioUrl)).json();
  applyScenarioToUi(scenario);
  startFlight(scenario.controls, scenario.start);
}

const rampToward = (value, target, elapsedS) => value + clamp(target - value, -KEY_RAMP_PER_S * elapsedS, KEY_RAMP_PER_S * elapsedS);

function stickInput(elapsedS) {
  const pad = [...(navigator.getGamepads?.() ?? [])].find((candidate) => candidate?.mapping === "standard");
  const axis = (index) => (pad && Math.abs(pad.axes[index]) > STICK_DEADZONE ? pad.axes[index] : 0);
  const trigger = (index) => pad?.buttons[index]?.value ?? 0;
  const key = (code) => (heldKeys.has(code) ? 1 : 0);
  keyAxes.roll = rampToward(keyAxes.roll, key("ArrowRight") - key("ArrowLeft"), elapsedS);
  keyAxes.pull = rampToward(keyAxes.pull, key("ArrowDown") - key("ArrowUp"), elapsedS);
  keyAxes.rudder = rampToward(keyAxes.rudder, key("KeyD") - key("KeyA"), elapsedS);
  const button = (index) => (pad?.buttons[index]?.pressed ? 1 : 0);
  const throttleRate = trigger(STANDARD_RIGHT_TRIGGER) - trigger(STANDARD_LEFT_TRIGGER) + key("KeyW") - key("KeyS");
  const brake = Math.max(button(STANDARD_B), key("Space"));
  const afterburner = Math.max(button(STANDARD_A), key("ShiftLeft"), key("ShiftRight"));
  const touched = {
    [ROW_ROLL_RATE]: axis(0) || key("ArrowRight") || key("ArrowLeft"),
    [ROW_LOAD_FACTOR]: axis(1) || key("ArrowDown") || key("ArrowUp"),
    [ROW_THROTTLE]: throttleRate,
    [ROW_SPEED_BRAKE]: brake,
    [ROW_AFTERBURNER]: afterburner,
  };
  return {
    manualAxes: Object.keys(touched).filter((column) => touched[column]).map(Number),
    rudder: clamp(button(STANDARD_RIGHT_BUMPER) - button(STANDARD_LEFT_BUMPER) + keyAxes.rudder, -1, 1),
    brake,
    afterburner,
    roll: clamp(axis(0) + keyAxes.roll, -1, 1),
    pull: clamp(axis(1) + keyAxes.pull, -1, 1),
    throttleRate,
  };
}

function readControl(elapsedS) {
  const stick = stickInput(elapsedS);
  steering = stick;
  const { max_roll_rate_rad_s, max_load_factor, max_side_load_factor, max_negative_load_factor } = flightSetup.envelope;
  throttle = clamp(throttle + stick.throttleRate * THROTTLE_RATE_PER_S * elapsedS, 0, 1);
  const loadFactor = stick.pull >= 0 ? 1 + stick.pull * (max_load_factor - 1) : 1 + stick.pull * (1 - max_negative_load_factor);
  return [stick.roll * max_roll_rate_rad_s, loadFactor, throttle, stick.rudder * max_side_load_factor, stick.brake, stick.afterburner];
}

function renderSteeringHud() {
  for (const code of STEERING_KEYS) byId("steering-hud").querySelector(`[data-hud-key="${code}"]`).dataset.held = String(heldKeys.has(code));
  const percents = { roll: steering.roll * 100, pull: steering.pull * 100, rudder: steering.rudder * 100, brake: steering.brake * 100, afterburner: steering.afterburner * 100, throttle: throttle * 100 };
  for (const [axis, percent] of Object.entries(percents)) {
    const row = byId("steering-hud").querySelector(`[data-hud-axis="${axis}"]`);
    row.dataset.percent = percent.toFixed(1);
    row.querySelector("meter").value = percent;
    row.querySelector("output").textContent = `${percent.toFixed(0)}%`;
  }
}

function renderHolds(frame) {
  for (const name of Object.values(HOLD_KEYS)) {
    byId("steering-hud").querySelector(`[data-hud-hold="${name}"]`).dataset.engaged = String(name in frame.holds);
  }
  if ("speed" in frame.holds) throttle = frame.control[ROW_THROTTLE];
}

function advance(steps) {
  const controls = script ? script.slice(stepsSent, stepsSent + steps) : Array(steps).fill(latestControl);
  if (!controls.length) {
    speed = 0;
    return;
  }
  stepsSent += controls.length;
  awaitingWorker = true;
  worker.postMessage(toWorker.advance(controls, script ? [] : steering.manualAxes));
}

function tick(now) {
  const elapsedS = (now - lastTick) / 1000;
  lastTick = now;
  if (flightSetup) {
    latestControl = readControl(elapsedS);
    renderSteeringHud();
    const [rollRate, loadFactor, throttleNow] = latestControl;
    byId("controls").textContent =
      `roll rate ${((rollRate * 180) / Math.PI).toFixed(0)} °/s, load factor ${loadFactor.toFixed(2)} g, throttle ${throttleNow.toFixed(2)}`;
    owedSteps = flying && speed > 0 ? Math.min(owedSteps + (elapsedS * speed) / options.dt_s, MAX_STEPS_PER_MESSAGE) : 0;
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
      const cells = [name, readoutText(readout.error_m, 1), readoutText(readout.nees, 2), readoutText(readout.nis, 2), ...FLAGS.map((flag) => flags[flag])];
      row.replaceChildren(
        ...cells.map((text, column) => {
          const cell = document.createElement("td");
          cell.textContent = text;
          if (column === 0 && readout.diverged) cell.append(divergedBadge(name, readout.diverged_step));
          if (column >= 4) Object.assign(cell.dataset, { filter: name, flag: FLAGS[column - 4] });
          return cell;
        }),
      );
      return row;
    }),
  );
}

function divergedBadge(name, step) {
  const badge = document.createElement("span");
  badge.className = "badge";
  Object.assign(badge.dataset, { divergedFilter: name, divergedStep: step });
  badge.textContent = `${DIVERGED_LABEL} at step ${step}`;
  return badge;
}

function renderLegend(frame) {
  byId("legend").replaceChildren(
    ...["truth", ...Object.keys(frame.filters)].map((name) => {
      const entry = document.createElement("span");
      entry.style.color = TRAIL_COLORS[name];
      entry.textContent = name;
      if (frame.filters[name]?.diverged) entry.append(divergedBadge(name, frame.filters[name].diverged_step));
      return entry;
    }),
  );
}

function renderPlan(frame) {
  const positions = { truth: frame.truth_m, ...Object.fromEntries(Object.entries(frame.filters).map(([name, readout]) => [name, readout.estimate_m ?? []])) };
  for (const [name, [x, y]] of Object.entries(positions)) {
    if (Number.isFinite(x) && Number.isFinite(y)) trails[name] = [...(trails[name] ?? []).slice(1 - TRAIL_LENGTH), [x, y]];
  }
  const { centre, span } = planExtent(trails, divergedFilters(frame));
  const canvas = byId("plan");
  canvas.dataset.spanM = span;
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

function renderViews(frame) {
  history = history.filter((entry) => entry.t < frame.step - 1);
  history.push({
    t: frame.step - 1,
    truthAltitude: frame.truth_m[2],
    filters: Object.fromEntries(
      Object.entries(frame.filters).map(([name, readout]) => [
        name,
        { error: readout.error_m, nees: readout.nees, altitude: readout.estimate_m?.[2], modes: readout.mode_probabilities },
      ]),
    ),
  });
  renderTimePlots(history, frame.trips, [options.nees_lower, thresholds().nees_upper], divergedFilters(frame));
  scene?.update(frame);
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
    setStatus(script ? `flying the scripted ${script.length} steps` : "flying: steer with the keyboard or a gamepad");
  },
  frame({ frame, stepCostMs }) {
    awaitingWorker = false;
    if (stepCostMs !== null) stepCosts = [...stepCosts.slice(1 - STEP_COST_WINDOW), stepCostMs];
    if (script && stepsSent === script.length) setStatus(`script finished at step ${stepsSent}`);
    renderReadouts(frame);
    renderLegend(frame);
    renderPlan(frame);
    renderViews(frame);
    renderClock(frame);
    renderHolds(frame);
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
createScene(byId("scene")).then(
  (created) => (scene = created),
  (error) => (byId("scene").textContent = `3D view unavailable: ${error.message}`),
);
byId("view_mode").addEventListener("change", (event) => scene?.setMode(event.target.value));

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
  if (flying && !awaitingWorker) advance(1);
});
byId("start").addEventListener("click", () => options && startFlight(null));
byId("stop").addEventListener("click", () => {
  if (!flying) return;
  flying = false;
  speed = 0;
  setStatus(`stopped at step ${stepsSent}: press Start for a new flight`);
});
for (const tab of document.querySelectorAll("[data-tab]")) {
  tab.addEventListener("click", () => {
    for (const other of document.querySelectorAll("[data-tab]")) {
      other.setAttribute("aria-selected", String(other === tab));
      byId(`tab-${other.dataset.tab}`).hidden = other !== tab;
    }
  });
}
byId("export").addEventListener("click", () => stepsSent && worker.postMessage(toWorker.exportScenario()));
addEventListener("keydown", (event) => {
  if (event.code in HOLD_KEYS && flying && !script && !event.repeat && !event.target.matches("input, select")) {
    worker.postMessage(toWorker.toggleHold(HOLD_KEYS[event.code]));
    event.preventDefault();
    return;
  }
  if (!STEERING_KEYS.has(event.code) || event.target.matches("input, select")) return;
  heldKeys.add(event.code);
  event.preventDefault();
});
addEventListener("keyup", (event) => heldKeys.delete(event.code));
requestAnimationFrame(tick);
