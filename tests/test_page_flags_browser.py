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
