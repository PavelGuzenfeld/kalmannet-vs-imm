import { FILTER_COLORS, TRUTH_COLOR } from "./plots.js";

const THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js";
const VIEW_WIDTH = 960;
const VIEW_HEIGHT = 600;
const FIELD_OF_VIEW_DEG = 55;
const NEAR_M = 1;
const FAR_M = 40000;
const TRAIL_POINTS = 600;
const TRUTH_SIZE_M = 6;
const ESTIMATE_RADIUS_M = 2;
const GRID_SIZE_M = 6000;
const GRID_CELLS = 60;
const GRID_CELL_M = GRID_SIZE_M / GRID_CELLS;
const DEFAULT_DISTANCE_M = 180;
const MIN_DISTANCE_M = 20;
const MAX_DISTANCE_M = 4000;
const DEFAULT_ELEVATION_RAD = 0.35;
const MAX_ELEVATION_RAD = 1.5;
const DRAG_RAD_PER_PX = 0.008;
const ZOOM_PER_WHEEL_UNIT = 0.001;
const ELLIPSOID_OPACITY = 0.18;
const SCALE_BAR_MAX_PX = 120;
const SCALE_BAR_STEPS = [5, 2, 1];
const KM_IN_M = 1000;
const SCALE_BAR_MARGIN_PX = 8;
const SCALE_BAR_COLOR = "#1d1d1f";

const toThree = ([east, north, up]) => [east, up, -north];

export function scaleBar(metresPerPixel, maxBarPx) {
  const limit = metresPerPixel * maxBarPx;
  const decade = 10 ** Math.floor(Math.log10(limit));
  const metres = Number((SCALE_BAR_STEPS.find((step) => step * decade <= limit) * decade).toPrecision(6));
  const label = metres >= KM_IN_M ? `${metres / KM_IN_M} km` : `${metres} m`;
  return { metres, pixels: metres / metresPerPixel, label };
}

export async function createScene(container) {
  const THREE = await import(THREE_URL);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(VIEW_WIDTH, VIEW_HEIGHT);
  renderer.setClearColor(0xf2f5fa);
  const scaleBarLabel = document.createElement("div");
  const scaleBarLine = document.createElement("div");
  const scaleBarBox = document.createElement("div");
  scaleBarBox.dataset.scaleBar = "";
  scaleBarBox.style.cssText = `position:absolute;left:${SCALE_BAR_MARGIN_PX}px;bottom:${SCALE_BAR_MARGIN_PX}px;font:11px monospace;color:${SCALE_BAR_COLOR};pointer-events:none`;
  scaleBarLine.style.cssText = `border:2px solid ${SCALE_BAR_COLOR};border-top:none;height:5px`;
  scaleBarBox.append(scaleBarLabel, scaleBarLine);
  container.style.position = "relative";
  container.style.width = "fit-content";
  container.replaceChildren(renderer.domElement, scaleBarBox);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FIELD_OF_VIEW_DEG, VIEW_WIDTH / VIEW_HEIGHT, NEAR_M, FAR_M);
  const grid = new THREE.GridHelper(GRID_SIZE_M, GRID_CELLS, 0x888888, 0xbbbbbb);
  scene.add(grid);

  const truth = new THREE.Mesh(new THREE.ConeGeometry(TRUTH_SIZE_M / 2, TRUTH_SIZE_M * 2, 12), new THREE.MeshBasicMaterial({ color: TRUTH_COLOR }));
  scene.add(truth);
  const trailOf = (color) => {
    const line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color }));
    line.frustumCulled = false;
    scene.add(line);
    return line;
  };
  const trails = { truth: trailOf(TRUTH_COLOR) };
  const markers = {};
  const trailPoints = { truth: [] };

  function markerFor(name) {
    const color = FILTER_COLORS[name];
    const estimate = new THREE.Mesh(new THREE.SphereGeometry(ESTIMATE_RADIUS_M, 12, 8), new THREE.MeshBasicMaterial({ color }));
    const ellipsoid = new THREE.Mesh(
      new THREE.SphereGeometry(1, 20, 14),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: ELLIPSOID_OPACITY, depthWrite: false }),
    );
    ellipsoid.matrixAutoUpdate = false;
    scene.add(estimate, ellipsoid);
    trails[name] = trailOf(color);
    trailPoints[name] = [];
    markers[name] = { estimate, ellipsoid };
  }

  const view = { mode: "chase", azimuth: 0, elevation: DEFAULT_ELEVATION_RAD, distance: DEFAULT_DISTANCE_M };
  let heading = 0;
  let focus = new THREE.Vector3();

  function placeCamera() {
    const azimuth = view.azimuth + (view.mode === "chase" ? heading : 0);
    container.dataset.cameraDistance = view.distance;
    const offset = new THREE.Vector3(
      -Math.cos(view.elevation) * Math.cos(azimuth),
      Math.sin(view.elevation),
      Math.cos(view.elevation) * Math.sin(azimuth),
    ).multiplyScalar(view.distance);
    container.dataset.cameraOffset = offset.toArray().join(" ");
    camera.position.copy(focus).add(offset);
    camera.lookAt(focus);
    placeScaleBar();
    renderer.render(scene, camera);
  }

  function placeScaleBar() {
    const viewHeightPx = renderer.domElement.clientHeight;
    if (!viewHeightPx) return;
    const metresPerPixel = (2 * camera.position.distanceTo(focus) * Math.tan((FIELD_OF_VIEW_DEG * Math.PI) / 360)) / viewHeightPx;
    const bar = scaleBar(metresPerPixel, SCALE_BAR_MAX_PX);
    scaleBarLabel.textContent = bar.label;
    scaleBarLine.style.width = `${bar.pixels}px`;
    container.dataset.scaleBarLabel = bar.label;
    container.dataset.scaleBarPx = bar.pixels;
  }

  function pushTrail(name, position) {
    trailPoints[name] = [...trailPoints[name].slice(1 - TRAIL_POINTS), position];
    trails[name].geometry.setFromPoints(trailPoints[name]);
  }

  function placeEllipsoid({ ellipsoid }, shape, centre) {
    ellipsoid.visible = Boolean(shape);
    if (!shape) return;
    const [a, b, c] = shape.axes.map((axis, index) => new THREE.Vector3(...toThree(axis)).multiplyScalar(shape.semi_axes_m[index]));
    ellipsoid.matrix.makeBasis(a, b, c).setPosition(centre);
  }

  const canvas = renderer.domElement;
  let dragFrom = null;
  canvas.addEventListener("pointerdown", (event) => {
    dragFrom = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", () => (dragFrom = null));
  canvas.addEventListener("pointermove", (event) => {
    if (!dragFrom) return;
    view.azimuth += (event.clientX - dragFrom[0]) * DRAG_RAD_PER_PX;
    view.elevation = Math.min(MAX_ELEVATION_RAD, Math.max(-MAX_ELEVATION_RAD, view.elevation + (event.clientY - dragFrom[1]) * DRAG_RAD_PER_PX));
    dragFrom = [event.clientX, event.clientY];
    placeCamera();
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      view.distance = Math.min(MAX_DISTANCE_M, Math.max(MIN_DISTANCE_M, view.distance * (1 + event.deltaY * ZOOM_PER_WHEEL_UNIT)));
      event.preventDefault();
      placeCamera();
    },
    { passive: false },
  );

  new ResizeObserver(placeCamera).observe(renderer.domElement);

  return {
    setMode(mode) {
      view.mode = mode;
      view.azimuth = 0;
      placeCamera();
    },
    update(frame) {
      const truthPosition = new THREE.Vector3(...toThree(frame.truth_m));
      const [east, north, up] = frame.truth_velocity_m_s;
      heading = Math.atan2(north, east);
      truth.position.copy(truthPosition);
      truth.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...toThree([east, north, up])).normalize());
      pushTrail("truth", truthPosition);
      for (const [name, readout] of Object.entries(frame.filters)) {
        if (!markers[name]) markerFor(name);
        markers[name].estimate.visible = Boolean(readout.estimate_m);
        if (!readout.estimate_m) {
          placeEllipsoid(markers[name], null);
          continue;
        }
        const centre = new THREE.Vector3(...toThree(readout.estimate_m));
        markers[name].estimate.position.copy(centre);
        placeEllipsoid(markers[name], readout.diverged ? null : readout.ellipsoid, centre);
        pushTrail(name, centre);
      }
      container.dataset.shownEllipsoids = Object.keys(markers).filter((name) => markers[name].ellipsoid.visible).join(" ");
      grid.position.set(Math.round(truthPosition.x / GRID_CELL_M) * GRID_CELL_M, 0, Math.round(truthPosition.z / GRID_CELL_M) * GRID_CELL_M);
      focus = truthPosition;
      placeCamera();
    },
    reset() {
      for (const name of Object.keys(trailPoints)) {
        trailPoints[name] = [];
        trails[name].geometry.setFromPoints([]);
      }
    },
  };
}
