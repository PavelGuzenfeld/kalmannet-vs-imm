import json
from pathlib import Path

import pytest

from tests.test_steering_hud_browser import (  # noqa: F401
    BOOT_TIMEOUT_MS,
    PAGE_BASE_URL,
    boot_paused,
    browser,
    run_until,
    start_flight,
)

START_FIELDS = {
    "start_range_m": "2000",
    "start_bearing_deg": "90",
    "start_altitude_m": "300",
    "start_heading_deg": "270",
    "start_speed_m_s": "150",
}
QUARTER_TURN_COSINE_RESIDUE_M = 1e-9

FLOWN_A_STEP = "/^step [1-9]/.test(document.getElementById('clock').textContent)"


def exported_scenario(page):
    assert run_until(page, FLOWN_A_STEP)
    with page.expect_download() as download:
        page.click("#export")
    return json.loads(Path(download.value.path()).read_text())


def test_set_range_bearing_altitude_heading_and_speed_export_that_start_east_north_up(browser):  # noqa: F811
    page = boot_paused(browser)
    for field, value in START_FIELDS.items():
        page.fill(f"#{field}", value)

    start_flight(page)
    start = exported_scenario(page)["start"]

    assert start["position"] == pytest.approx([2000.0, 0.0, 300.0], abs=QUARTER_TURN_COSINE_RESIDUE_M)
    assert start["velocity"] == pytest.approx([-150.0, 0.0, 0.0], abs=QUARTER_TURN_COSINE_RESIDUE_M)


def test_all_five_on_random_exports_no_start(browser):  # noqa: F811
    page = boot_paused(browser)

    start_flight(page)

    assert "start" not in exported_scenario(page)


def test_speed_over_the_envelope_is_refused_with_a_message_naming_the_limits(browser):  # noqa: F811
    page = boot_paused(browser)
    page.fill("#start_speed_m_s", "400")

    page.click("#start")
    page.wait_for_function(
        "document.getElementById('status').textContent.startsWith('worker error')",
        polling=200,
        timeout=BOOT_TIMEOUT_MS,
    )

    assert "speed 400 m/s is outside the envelope's 20 to 250 m/s" in page.inner_text("#status")


def test_start_stays_disabled_while_the_filters_load_so_an_early_click_is_not_dropped(browser):  # noqa: F811
    page = browser.new_context().new_page()
    page.goto(f"{PAGE_BASE_URL}/web/index.html", wait_until="domcontentloaded")
    assert page.is_disabled("#start")
    page.wait_for_function(
        "document.getElementById('status').textContent.startsWith('paused')", polling=200, timeout=BOOT_TIMEOUT_MS
    )
    assert page.is_enabled("#start")
