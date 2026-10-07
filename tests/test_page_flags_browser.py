import json
import math
import os
from pathlib import Path

import pytest

PAGE_BASE_URL = os.environ.get("PAGE_BASE_URL")
if not PAGE_BASE_URL:
    pytest.skip("browser test: set PAGE_BASE_URL to a served repo root", allow_module_level=True)

from playwright import sync_api  # noqa: E402

CASES_DIR = Path(os.environ.get("PAGE_CASES_DIR", "build/browser"))
CASES_URL_PATH = "/build/browser"
FLIGHT_TIMEOUT_MS = 240_000
LEARNED_FILTERS = ("KalmanNet", "RKN")
LARGE_RADIUS_CASE = "break_turn_9g"
LARGE_RADIUS_CASE_STEPS = 296
DEFAULT_CAMERA_DISTANCE_M = 180.0
UNAVAILABLE_PREFIXES = ("not available", "n/a")
CASE_NAMES = sorted(path.name.removesuffix(".expected.json") for path in CASES_DIR.glob("*.expected.json"))

assert CASE_NAMES, f"no cases in {CASES_DIR}; run scripts/page_parity.py cases first"


@pytest.fixture()
def browser_context():
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser.new_context()
        browser.close()


def finished_page(context, case_name, steps):
    page = context.new_page()
    page.goto(f"{PAGE_BASE_URL}/web/index.html?scenario={CASES_URL_PATH}/{case_name}.json")
    page.wait_for_function(
        "() => /^(script finished|worker error)/.test(document.getElementById('status').textContent)",
        timeout=FLIGHT_TIMEOUT_MS,
    )
    assert page.locator("#status").text_content() == f"script finished at step {steps}"
    return page


def page_flags(page):
    cells = page.eval_on_selector_all(
        "td[data-filter][data-flag]",
        "cells => cells.map(c => [c.dataset.filter, c.dataset.flag, c.textContent])",
    )
    flags = {}
    for filter_name, flag, text in cells:
        flags.setdefault(filter_name, {})[flag] = text
    return flags


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_page_flags_at_the_end_of_the_script_equal_native_replay(browser_context, case_name):
    expected = json.loads((CASES_DIR / f"{case_name}.expected.json").read_text())

    page = finished_page(browser_context, case_name, expected["steps"])
    flags = page_flags(page)

    assert {name: flags[name] for name in expected["flags"]} == expected["flags"]
    for name in LEARNED_FILTERS:
        assert all(text.startswith(UNAVAILABLE_PREFIXES) for text in flags[name].values())


def camera_distance(page):
    return float(page.locator("#scene").get_attribute("data-camera-distance"))


def test_chase_camera_keeps_the_user_distance_when_the_ellipsoid_is_large(browser_context):
    page = finished_page(browser_context, LARGE_RADIUS_CASE, LARGE_RADIUS_CASE_STEPS)

    assert camera_distance(page) == DEFAULT_CAMERA_DISTANCE_M


def test_chase_camera_sits_behind_the_truth_along_its_velocity(browser_context):
    page = module_page(browser_context)

    offset = page.evaluate(
        """async () => {
          const container = document.createElement("div");
          document.body.append(container);
          const scene = await (await import("/web/scene.js")).createScene(container);
          scene.update({ truth_m: [0, 0, 400], truth_velocity_m_s: [100, 0, 0], truth_lift_direction: [0, 0, 1], filters: {} });
          return container.dataset.cameraOffset.split(" ").map(Number);
        }"""
    )

    assert offset[0] < 0
    assert offset[2] == 0


def test_orbit_camera_keeps_the_user_distance_when_the_ellipsoid_is_large(browser_context):
    page = finished_page(browser_context, LARGE_RADIUS_CASE, LARGE_RADIUS_CASE_STEPS)

    page.select_option("#view_mode", "orbit")

    assert camera_distance(page) == DEFAULT_CAMERA_DISTANCE_M


BOOT_TIMEOUT_MS = 240_000
PAUSED_WATCH_MS = 1_500
KNET_WEIGHTS_URL_PATH = "/results/knet_vanilla/best.npz"
DIVERGING_KNET_WEIGHTS = CASES_DIR / "diverging_knet.npz"
DIVERGING_KNET_DIVERGENCE = "diverging_knet_divergence.json"
KEPT_SCALE_BOUND_M = 3e5
STOP_SETTLE_MS = 500
ANALYSIS_PANELS = ("altitude", "error", "nees", "modes")


def loaded_page(context):
    page = context.new_page()
    page.goto(f"{PAGE_BASE_URL}/web/index.html")
    page.wait_for_function(
        "() => /^(paused|worker error)/.test(document.getElementById('status').textContent)",
        timeout=BOOT_TIMEOUT_MS,
    )
    return page


def test_page_without_a_scenario_waits_for_start_and_flies_no_step_before_it(browser_context):
    page = loaded_page(browser_context)

    page.wait_for_timeout(PAUSED_WATCH_MS)

    assert page.locator("#clock").text_content() == ""


def test_start_flies_steps_from_the_chosen_parameters(browser_context):
    page = loaded_page(browser_context)

    page.click("#start")

    page.wait_for_function("() => /^step [1-9]/.test(document.getElementById('clock').textContent)")


def test_analysis_tab_holds_the_plan_view_time_plots_and_flags_table_and_opens_on_click(browser_context):
    page = loaded_page(browser_context)
    analysis = page.locator("#tab-analysis")
    assert analysis.is_hidden()

    page.click("button[data-tab='analysis']")

    assert analysis.is_visible()
    panels = analysis.locator("svg[data-panel]").evaluate_all("svgs => svgs.map(s => s.dataset.panel)")
    assert tuple(panels) == ANALYSIS_PANELS
    assert analysis.locator("#plan").count() == 1
    assert analysis.locator("#readouts").count() == 1


def module_page(context):
    page = context.new_page()
    page.goto(f"{PAGE_BASE_URL}/")
    return page


def test_plan_extent_ignores_the_trail_of_a_diverged_filter(browser_context):
    page = module_page(browser_context)
    trails = {"truth": [[0, 0], [1000, 400]], "IMM": [[10, -50]], "RKN": [[0, 0], [1.0e8, -2.0e8]]}

    with_diverged, without = page.evaluate(
        """async (trails) => {
          const { planExtent } = await import("/web/plots.js");
          const { RKN, ...kept } = trails;
          return [planExtent(trails, new Set(["RKN"])), planExtent(kept, new Set())];
        }""",
        trails,
    )

    assert with_diverged == without == {"centre": [500, 175], "span": 1000 * 1.2}


@pytest.mark.parametrize(
    ("value", "text"),
    [(-438.12, "invalid"), (float("nan"), "invalid"), (float("inf"), "invalid"), (None, "—"), (5.4889, "5.49"), (0.0, "0.00")],
)
def test_readout_shows_a_negative_or_non_finite_quadratic_form_as_invalid(browser_context, value, text):
    page = module_page(browser_context)

    shown = page.evaluate(
        """async (value) => (await import("/web/plots.js")).readoutText(value, 2)""", value
    )

    assert shown == text


def test_nees_plot_leaves_out_a_negative_quadratic_form(browser_context):
    page = module_page(browser_context)

    plotted = page.evaluate(
        """async (panels) => {
          document.body.innerHTML = panels.map((panel) => `<svg data-panel="${panel}"></svg>`).join("");
          const { renderTimePlots } = await import("/web/plots.js");
          const history = [1.5, -438.12, 2.5].map((nees, t) => ({
            t,
            truthAltitude: 400,
            filters: { EKF: { error: 1, nees, altitude: 400, modes: null } },
          }));
          renderTimePlots(history, {}, [0.5, 20], new Set());
          return document.querySelector('svg[data-panel="nees"] polyline[data-series="EKF"]').getAttribute("points").split(" ");
        }""",
        list(ANALYSIS_PANELS),
    )

    assert len(plotted) == 2


def test_scene_hides_the_ellipsoid_of_a_diverged_filter(browser_context):
    page = module_page(browser_context)

    shown = page.evaluate(
        """async () => {
          const container = document.createElement("div");
          document.body.append(container);
          const scene = await (await import("/web/scene.js")).createScene(container);
          const ellipsoid = { semi_axes_m: [5, 4, 3], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] };
          const readout = (diverged) => ({ estimate_m: [10, 0, 400], ellipsoid, diverged });
          scene.update({
            truth_m: [0, 0, 400],
            truth_velocity_m_s: [100, 0, 0],
            truth_lift_direction: [0, 0, 1],
            filters: { EKF: readout(false), UKF: readout(true) },
          });
          return container.dataset.shownEllipsoids;
        }"""
    )

    assert shown == "EKF"


def test_stop_halts_the_flight_clock(browser_context):
    page = loaded_page(browser_context)
    page.click("#start")
    page.wait_for_function("() => /^step [1-9]/.test(document.getElementById('clock').textContent)")

    page.click("#stop")
    page.wait_for_function("() => document.getElementById('status').textContent.startsWith('stopped')")
    page.wait_for_timeout(STOP_SETTLE_MS)
    stopped_at = page.locator("#clock").text_content()
    page.wait_for_timeout(PAUSED_WATCH_MS)

    assert page.locator("#clock").text_content() == stopped_at


def test_diverged_filter_gets_a_badge_at_the_native_divergence_step_and_leaves_the_plot_scales(browser_context):
    """KalmanNet on the diverging weights passes 200 m in the break turn, at the step replay.py's run records, and
    is stepped no further. Truth's break-turn altitude spans under 100 m, so a kept-only scale spans under 3e5 m."""
    native_step = json.loads((CASES_DIR / DIVERGING_KNET_DIVERGENCE).read_text())["KalmanNet"]
    page = browser_context.new_page()
    page.route(f"**{KNET_WEIGHTS_URL_PATH}", lambda route: route.fulfill(path=DIVERGING_KNET_WEIGHTS))
    page.goto(f"{PAGE_BASE_URL}/web/index.html?scenario={CASES_URL_PATH}/{LARGE_RADIUS_CASE}.json")
    page.wait_for_function(
        "() => /^(script finished|worker error)/.test(document.getElementById('status').textContent)",
        timeout=FLIGHT_TIMEOUT_MS,
    )
    assert page.locator("#status").text_content() == f"script finished at step {LARGE_RADIUS_CASE_STEPS}"

    badges = page.eval_on_selector_all(
        "#readouts [data-diverged-filter]", "spans => spans.map(s => [s.dataset.divergedFilter, Number(s.dataset.divergedStep)])"
    )
    altitude_axis = page.locator('svg[data-panel="altitude"]').evaluate("svg => [Number(svg.dataset.axisLow), Number(svg.dataset.axisHigh)]")
    plan_span = float(page.locator("#plan").get_attribute("data-span-m"))

    assert badges == [["KalmanNet", native_step]]
    assert 0 < altitude_axis[1] - altitude_axis[0] < KEPT_SCALE_BOUND_M
    assert plan_span < KEPT_SCALE_BOUND_M


FIELD_OF_VIEW_DEG = 55
SCALE_BAR_TOLERANCE_PX = 1
KM_IN_M = 1000
ONE_TWO_FIVE_MANTISSAS = ("1", "2", "5")


def bar_metres(label):
    value, unit = label.split()
    return float(value) * (KM_IN_M if unit == "km" else 1)


def is_one_two_five(metres):
    digits = f"{metres:.6g}".replace(".", "").strip("0")
    return digits in ONE_TWO_FIVE_MANTISSAS


@pytest.mark.parametrize("mode", ["chase", "orbit"])
def test_scale_bar_is_a_one_two_five_value_whose_length_matches_metres_per_pixel_at_two_zoom_distances(browser_context, mode):
    page = module_page(browser_context)

    readings = page.evaluate(
        """async (mode) => {
          const container = document.createElement("div");
          document.body.append(container);
          const scene = await (await import("/web/scene.js")).createScene(container);
          scene.update({
            truth_m: [0, 0, 400],
            truth_velocity_m_s: [100, 0, 0],
            truth_lift_direction: [0, 0, 1],
            filters: {},
          });
          scene.setMode(mode);
          const canvas = container.querySelector("canvas");
          const read = () => ({
            label: container.dataset.scaleBarLabel,
            pixels: Number(container.dataset.scaleBarPx),
            distance: Number(container.dataset.cameraDistance),
            viewHeightPx: canvas.clientHeight,
          });
          const near = read();
          canvas.dispatchEvent(new WheelEvent("wheel", { deltaY: 1500 }));
          return [near, read()];
        }""",
        mode,
    )

    near, far = readings
    assert far["distance"] > near["distance"]
    for reading in readings:
        metres_per_pixel = 2 * reading["distance"] * math.tan(math.radians(FIELD_OF_VIEW_DEG) / 2) / reading["viewHeightPx"]
        metres = bar_metres(reading["label"])
        assert is_one_two_five(metres)
        assert metres / metres_per_pixel == pytest.approx(reading["pixels"], abs=SCALE_BAR_TOLERANCE_PX)


def test_scale_bar_sits_in_the_bottom_left_corner_of_the_view(browser_context):
    page = module_page(browser_context)

    corner = page.evaluate(
        """async () => {
          const container = document.createElement("div");
          document.body.append(container);
          await (await import("/web/scene.js")).createScene(container);
          const box = container.getBoundingClientRect();
          const bar = container.querySelector("[data-scale-bar]").getBoundingClientRect();
          return { left: bar.left - box.left, fromBottom: box.bottom - bar.bottom, top: bar.top - box.top, right: box.right - bar.right };
        }"""
    )

    assert corner["left"] < corner["right"]
    assert corner["fromBottom"] < corner["top"]


@pytest.mark.parametrize(
    ("metres_per_pixel", "max_bar_px", "metres", "label"),
    [
        (1, 1, 1, "1 m"),
        (1, 1.99, 1, "1 m"),
        (1, 2, 2, "2 m"),
        (1, 4.99, 2, "2 m"),
        (1, 5, 5, "5 m"),
        (1, 9.99, 5, "5 m"),
        (1, 10, 10, "10 m"),
        (1, 999, 500, "500 m"),
        (1, 1000, 1000, "1 km"),
        (1, 2000, 2000, "2 km"),
        (1, 5000, 5000, "5 km"),
        (1, 10000, 10000, "10 km"),
        (1, 0.5, 0.5, "0.5 m"),
        (1, 0.2, 0.2, "0.2 m"),
        (1, 0.1, 0.1, "0.1 m"),
        (1, 0.99, 0.5, "0.5 m"),
        (4, 5, 20, "20 m"),
        (0.5, 150, 50, "50 m"),
    ],
)
def test_scale_bar_rounds_down_to_a_one_two_five_value_and_switches_to_km_at_a_thousand_metres(
    browser_context, metres_per_pixel, max_bar_px, metres, label
):
    page = module_page(browser_context)

    bar = page.evaluate(
        """async ([metresPerPixel, maxBarPx]) => (await import("/web/scene.js")).scaleBar(metresPerPixel, maxBarPx)""",
        [metres_per_pixel, max_bar_px],
    )

    assert bar == {"metres": metres, "pixels": metres / metres_per_pixel, "label": label}


WING_AXIS_ATOL = 1e-6
BANKED_WING_Z_MIN = 0.3
SCENE_FRAME_CASES = {
    "level_east": ([100, 0, 0], [0, 0, 1]),
    "banked_right_45deg_east": ([100, 0, 0], [0, -0.5**0.5, 0.5**0.5]),
    "inverted_east": ([100, 0, 0], [0, 0, -1]),
    "vertical_climb_lift_north": ([0, 0, 100], [0, 1, 0]),
    "loop_top_inverted_west": ([-100, 0, 0], [0, 0, -1]),
}


def scene_wing_axis(page, velocity, lift):
    return page.evaluate(
        """async ([velocity, lift]) => {
          const container = document.createElement("div");
          document.body.append(container);
          const scene = await (await import("/web/scene.js")).createScene(container);
          scene.update({ truth_m: [0, 0, 400], truth_velocity_m_s: velocity, truth_lift_direction: lift, filters: {} });
          return container.dataset.truthWingAxis.split(" ").map(Number);
        }""",
        [velocity, lift],
    )


@pytest.mark.parametrize("case", SCENE_FRAME_CASES)
def test_truth_wing_axis_is_perpendicular_to_the_frames_lift_direction_and_velocity(browser_context, case):
    velocity, lift = SCENE_FRAME_CASES[case]
    page = module_page(browser_context)

    wing_axis = scene_wing_axis(page, velocity, lift)

    assert sum(w * v for w, v in zip(wing_axis, velocity)) == pytest.approx(0.0, abs=WING_AXIS_ATOL * 100)
    assert sum(w * up for w, up in zip(wing_axis, lift)) == pytest.approx(0.0, abs=WING_AXIS_ATOL)
    assert sum(w * w for w in wing_axis) == pytest.approx(1.0, abs=WING_AXIS_ATOL)


def test_truth_wings_are_horizontal_in_level_flight(browser_context):
    velocity, lift = SCENE_FRAME_CASES["level_east"]

    wing_axis = scene_wing_axis(module_page(browser_context), velocity, lift)

    assert wing_axis[2] == pytest.approx(0.0, abs=WING_AXIS_ATOL)


def wing_axis_z(page):
    axis = page.locator("#scene").get_attribute("data-truth-wing-axis")
    return float(axis.split()[2]) if axis else None


def test_flying_a_banked_turn_tilts_the_truth_wings_off_the_horizontal(browser_context):
    page = loaded_page(browser_context)
    page.click("#start")
    page.wait_for_function("() => /^step [1-9]/.test(document.getElementById('clock').textContent)")
    assert abs(wing_axis_z(page)) < WING_AXIS_ATOL

    page.keyboard.down("ArrowRight")
    page.wait_for_function(
        f"() => Math.abs(Number(document.getElementById('scene').dataset.truthWingAxis.split(' ')[2])) > {BANKED_WING_Z_MIN}",
        timeout=FLIGHT_TIMEOUT_MS,
    )
