import json
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
FRAMING_DISTANCE_PER_RADIUS = 2.5
DEFAULT_CAMERA_DISTANCE_M = 180.0
UNAVAILABLE_PREFIXES = ("not available", "n/a")
CASE_NAMES = sorted(path.name.removesuffix(".expected.json") for path in CASES_DIR.glob("*.expected.json"))

assert CASE_NAMES, f"no cases in {CASES_DIR}; run scripts/page_parity.py cases first"


@pytest.fixture(scope="module")
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


def camera_state(page):
    dataset = page.locator("#scene").evaluate("element => ({...element.dataset})")
    return float(dataset["cameraDistance"]), float(dataset["framingRadius"])


def test_chase_camera_backs_off_to_the_framing_distance_when_the_ellipsoid_is_large(browser_context):
    page = finished_page(browser_context, LARGE_RADIUS_CASE, LARGE_RADIUS_CASE_STEPS)

    distance, framing_radius = camera_state(page)

    assert framing_radius * FRAMING_DISTANCE_PER_RADIUS > DEFAULT_CAMERA_DISTANCE_M
    assert distance == framing_radius * FRAMING_DISTANCE_PER_RADIUS


def test_orbit_camera_keeps_the_user_distance_when_the_ellipsoid_is_large(browser_context):
    page = finished_page(browser_context, LARGE_RADIUS_CASE, LARGE_RADIUS_CASE_STEPS)

    page.select_option("#view_mode", "orbit")

    distance, framing_radius = camera_state(page)
    assert framing_radius * FRAMING_DISTANCE_PER_RADIUS > DEFAULT_CAMERA_DISTANCE_M
    assert distance == DEFAULT_CAMERA_DISTANCE_M


BOOT_TIMEOUT_MS = 240_000
PAUSED_WATCH_MS = 1_500
DIVERGED_CASE_URL_PATH = "/build/browser/diverged-case.json"
DIVERGING_RAW_ERROR_M = 1.0
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


def test_diverged_filters_get_a_badge_and_leave_the_chase_camera_framing(browser_context):
    scenario = json.loads((CASES_DIR / f"{LARGE_RADIUS_CASE}.json").read_text())
    scenario["thresholds"]["raw_error_m"] = DIVERGING_RAW_ERROR_M
    page = browser_context.new_page()
    page.route(f"{PAGE_BASE_URL}{DIVERGED_CASE_URL_PATH}", lambda route: route.fulfill(json=scenario))
    page.goto(f"{PAGE_BASE_URL}/web/index.html?scenario={DIVERGED_CASE_URL_PATH}")
    page.wait_for_function(
        "() => /^(script finished|worker error)/.test(document.getElementById('status').textContent)",
        timeout=FLIGHT_TIMEOUT_MS,
    )

    badges = page.eval_on_selector_all("#readouts [data-diverged-filter]", "spans => spans.map(s => s.dataset.divergedFilter)")
    distance, framing_radius = camera_state(page)

    assert sorted(badges) == ["EKF", "UKF"]
    assert framing_radius * FRAMING_DISTANCE_PER_RADIUS < DEFAULT_CAMERA_DISTANCE_M
    assert distance == DEFAULT_CAMERA_DISTANCE_M
