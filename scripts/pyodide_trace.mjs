import { readFileSync, readdirSync, mkdirSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

const [pyodideModule, outPath, ...scenarioPaths] = process.argv.slice(2);
const { loadPyodide } = await import(pyodideModule);
const SITE_DIR = "/repo";
const PYTHON_TREES = ["data", "filters"];
const PYTHON_FILES = ["scripts/eval.py", "web/holds.py", "web/live_flight.py", "web/manoeuvres.py"];

function sourceFiles() {
  const trees = PYTHON_TREES.flatMap((tree) => walk(tree));
  return [...trees, ...PYTHON_FILES];
}

function walk(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return entry.name === "__pycache__" ? [] : walk(path);
    return path.endsWith(".py") ? [path] : [];
  });
}

const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "scipy"]);
for (const path of sourceFiles()) {
  pyodide.FS.mkdirTree(`${SITE_DIR}/${path.slice(0, Math.max(path.lastIndexOf("/"), 0))}`);
  pyodide.FS.writeFile(`${SITE_DIR}/${path}`, readFileSync(path));
}
pyodide.runPython(`import sys; sys.path.insert(0, "${SITE_DIR}")`);
const liveFlight = pyodide.pyimport("web.live_flight");
const immBest = JSON.parse(readFileSync("web/imm_tuning.json", "utf8")).best;
const weightsRoot = pyodide.pyimport("pathlib").Path("/nonexistent");

const traces = {};
for (const path of scenarioPaths) {
  const scenario = JSON.parse(readFileSync(path, "utf8"));
  const steps = liveFlight.trace(pyodide.toPy(scenario), pyodide.toPy(immBest), weightsRoot);
  traces[basename(path, ".json")] = steps.toJs({ dict_converter: Object.fromEntries });
}
mkdirSync(join(outPath, ".."), { recursive: true });
writeFileSync(outPath, JSON.stringify(traces));
