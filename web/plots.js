export const FILTER_COLORS = { EKF: "#d62728", UKF: "#ff7f0e", IMM: "#2ca02c", KalmanNet: "#1f77b4", RKN: "#9467bd", BKN: "#8c564b", UKN: "#e377c2", CKN: "#17becf" };
export const TRUTH_COLOR = "#000000";
export const MODE_NAMES = ["CV", "CT+", "CT−", "CA"];
const MODE_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"];
const FLAG_DASHES = { track_loss: "", overconfidence: "6 3", raw_error: "2 3" };
const SVG_NS = "http://www.w3.org/2000/svg";
const WIDTH = 640;
const HEIGHT = 220;
const PAD = { left: 48, right: 8, top: 32, bottom: 18 };
const MAX_PLOT_POINTS = 500;
const MIN_LOG_VALUE = 0.01;
const LEGEND_CHAR_PX = 6.5;
const LEGEND_GAP_PX = 14;
const PLAN_MARGIN = 0.1;
const MIN_PLAN_SPAN_M = 200;
export const DIVERGED_LABEL = "diverged";

const finite = ([, y]) => Number.isFinite(y);
const validReadout = (value) => Number.isFinite(value) && value >= 0;

export function readoutText(value, digits) {
  if (value === null || value === undefined) return "—";
  return validReadout(value) ? value.toFixed(digits) : "invalid";
}

export function planExtent(trails, diverged) {
  const points = Object.entries(trails).flatMap(([name, trail]) => (diverged.has(name) ? [] : trail));
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  return {
    centre: [(Math.max(...xs) + Math.min(...xs)) / 2, (Math.max(...ys) + Math.min(...ys)) / 2],
    span: Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), MIN_PLAN_SPAN_M) * (1 + 2 * PLAN_MARGIN),
  };
}

function svgElement(tag, attributes, text) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}

function sampled(history, read) {
  const stride = Math.max(1, Math.ceil(history.length / MAX_PLOT_POINTS));
  return history.filter((_, index) => index % stride === 0 || index === history.length - 1).map((entry) => [entry.t, read(entry)]);
}

function axisFor(values, log, band) {
  const all = [...values, ...(band ?? [])].filter(Number.isFinite);
  const low = all.length ? Math.min(...all) : 0;
  const high = all.length ? Math.max(...all) : 1;
  if (log) {
    const bottom = Math.log10(Math.max(low, MIN_LOG_VALUE));
    const top = Math.max(Math.log10(Math.max(high, MIN_LOG_VALUE)), bottom + 1);
    return { low: 10 ** bottom, high: 10 ** top, toUnit: (value) => (Math.log10(Math.max(value, MIN_LOG_VALUE)) - bottom) / (top - bottom), label: (unit) => (10 ** (bottom + unit * (top - bottom))).toPrecision(2) };
  }
  const span = high - low || 1;
  return { low, high: low + span, toUnit: (value) => (value - low) / span, label: (unit) => (low + unit * span).toPrecision(4) };
}

function drawPanel(svg, { title, series, log = false, band = null, bounds = null, lastT, markers }) {
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const scaled = series.filter(({ diverged }) => !diverged);
  const axis = axisFor(bounds ?? scaled.flatMap(({ points }) => points.map(([, y]) => y)), log, band);
  Object.assign(svg.dataset, { axisLow: axis.low, axisHigh: axis.high });
  const toX = (t) => PAD.left + (t / Math.max(lastT, 1)) * plotW;
  const toY = (value) => PAD.top + (1 - axis.toUnit(value)) * plotH;
  const parts = [
    svgElement("rect", { x: PAD.left, y: PAD.top, width: plotW, height: plotH, fill: "none", stroke: "#ccc" }),
    svgElement("text", { x: PAD.left, y: 12, "font-size": 12, fill: "#333" }, title),
    svgElement("text", { x: PAD.left - 4, y: PAD.top + 10, "font-size": 10, "text-anchor": "end", fill: "#666" }, axis.label(1)),
    svgElement("text", { x: PAD.left - 4, y: PAD.top + plotH, "font-size": 10, "text-anchor": "end", fill: "#666" }, axis.label(0)),
    svgElement("text", { x: PAD.left + plotW, y: HEIGHT - 4, "font-size": 10, "text-anchor": "end", fill: "#666" }, `step ${lastT}`),
    svgElement("text", { x: PAD.left, y: HEIGHT - 4, "font-size": 10, fill: "#666" }, "0"),
  ];
  if (band) {
    const [lower, upper] = band.map(toY);
    parts.push(svgElement("rect", { x: PAD.left, y: Math.min(lower, upper), width: plotW, height: Math.abs(lower - upper), fill: "#2ca02c", opacity: 0.12 }));
  }
  let legendX = PAD.left;
  for (const { name, color, diverged } of series) {
    const label = diverged ? `${name} (${DIVERGED_LABEL})` : name;
    parts.push(svgElement("text", { x: legendX, y: 26, "font-size": 11, fill: color }, label));
    legendX += label.length * LEGEND_CHAR_PX + LEGEND_GAP_PX;
  }
  for (const { name, color, points } of series) {
    const path = points.filter(finite).map(([t, value]) => `${toX(t).toFixed(1)},${toY(value).toFixed(1)}`).join(" ");
    parts.push(svgElement("polyline", { points: path, fill: "none", stroke: color, "stroke-width": 1.2, "data-series": name }));
  }
  for (const { filter, flag, step } of markers) {
    parts.push(
      svgElement("line", {
        class: "trip-marker",
        x1: toX(step),
        x2: toX(step),
        y1: PAD.top,
        y2: PAD.top + plotH,
        stroke: FILTER_COLORS[filter],
        "stroke-dasharray": FLAG_DASHES[flag],
        "stroke-width": 1.5,
        "data-panel": svg.dataset.panel,
        "data-filter": filter,
        "data-flag": flag,
        "data-trip-step": step,
      }),
    );
  }
  svg.replaceChildren(...parts);
}

function tripMarkers(trips) {
  return Object.entries(trips).flatMap(([filter, flags]) =>
    Object.entries(flags).filter(([, step]) => Number.isInteger(step)).map(([flag, step]) => ({ filter, flag, step })),
  );
}

export function renderTimePlots(history, trips, neesBand, diverged) {
  const markers = tripMarkers(trips);
  const lastT = history.at(-1).t;
  const names = Object.keys(history.at(-1).filters);
  const perFilter = (read) =>
    names.map((name) => ({ name, color: FILTER_COLORS[name], diverged: diverged.has(name), points: sampled(history, (entry) => read(entry.filters[name])) }));
  const panel = (id) => document.querySelector(`svg[data-panel="${id}"]`);
  drawPanel(panel("altitude"), {
    title: "altitude (m)",
    series: [{ name: "truth", color: TRUTH_COLOR, points: sampled(history, (entry) => entry.truthAltitude) }, ...perFilter((readout) => readout.altitude)],
    lastT,
    markers,
  });
  drawPanel(panel("error"), { title: "position error (m, log)", series: perFilter((readout) => readout.error), log: true, lastT, markers });
  drawPanel(panel("nees"), {
    title: `NEES (log); shaded is the chi-square 95% band, ${neesBand.map((value) => value.toFixed(2)).join(" to ")}`,
    series: perFilter((readout) => (validReadout(readout.nees) ? readout.nees : NaN)),
    log: true,
    band: neesBand,
    lastT,
    markers,
  });
  drawPanel(panel("modes"), {
    title: "IMM mode probabilities",
    series: MODE_NAMES.map((name, index) => ({ name, color: MODE_COLORS[index], points: sampled(history, (entry) => entry.filters.IMM?.modes?.[index]) })),
    bounds: [0, 1],
    lastT,
    markers,
  });
}
