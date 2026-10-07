import json
import os
from pathlib import Path

import pytest

PAGE_BASE_URL = os.environ.get("PAGE_BASE_URL")
if not PAGE_BASE_URL:
    pytest.skip("browser test: set PAGE_BASE_URL to a served repo root", allow_module_level=True)

from playwright import sync_api  # noqa: E402

BOOT_TIMEOUT_MS = 240_000
FRAME_MS = 16
RAMP_S = 2.0
HOLD_MS = 1000
KEYDOWN_POLL_MS = 100
KEYDOWN_POLLS = 100
FRAME_STEP_PERCENT = 100 * FRAME_MS / 1000 / RAMP_S


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        yield chromium
        chromium.close()


def boot_paused(browser):
    page = browser.new_context().new_page()
    page.clock.install(time=0)
    page.clock.pause_at(FRAME_MS)
    page.goto(f"{PAGE_BASE_URL}/web/index.html")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('paused')", polling=200, timeout=BOOT_TIMEOUT_MS)
    return page


def start_flight(page):
    page.click("#start")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('flying')", polling=200, timeout=BOOT_TIMEOUT_MS)
    page.bring_to_front()


def flying_page(browser):
    page = boot_paused(browser)
    start_flight(page)
    return page


def press_and_wait_for_keydown(page, key):
    delivered = []
    page.expose_function("reportKeydown", lambda: delivered.append(key))
    page.evaluate("addEventListener('keydown', () => window.reportKeydown(), { once: true })")
    page.keyboard.down(key)
    for _ in range(KEYDOWN_POLLS):
        if delivered:
            return
        page.wait_for_timeout(KEYDOWN_POLL_MS)
    pytest.fail(f"keydown {key} never reached the page")


def hud_percent(page, axis):
    return float(page.locator(f'[data-hud-axis="{axis}"]').get_attribute("data-percent"))


def key_lit(page, code):
    return page.locator(f'[data-hud-key="{code}"]').get_attribute("data-held") == "true"


def test_holding_right_arrow_for_one_second_reads_half_roll_with_the_key_lit(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "ArrowRight")
    page.clock.run_for(HOLD_MS)

    assert hud_percent(page, "roll") == pytest.approx(50, abs=FRAME_STEP_PERCENT)
    assert key_lit(page, "ArrowRight")


def test_releasing_right_arrow_unlights_the_key_and_ramps_roll_back_to_zero_within_one_second(browser):
    page = flying_page(browser)
    press_and_wait_for_keydown(page, "ArrowRight")
    page.clock.run_for(HOLD_MS)

    page.keyboard.up("ArrowRight")
    page.clock.run_for(HOLD_MS + 2 * FRAME_MS)

    assert not key_lit(page, "ArrowRight")
    assert hud_percent(page, "roll") == pytest.approx(0, abs=FRAME_STEP_PERCENT)


def test_holding_d_for_one_second_reads_half_rudder_with_the_key_lit(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyD")
    page.clock.run_for(HOLD_MS)

    assert hud_percent(page, "rudder") == pytest.approx(50, abs=FRAME_STEP_PERCENT)
    assert key_lit(page, "KeyD")


def test_holding_a_reads_negative_rudder(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyA")
    page.clock.run_for(HOLD_MS)

    assert hud_percent(page, "rudder") == pytest.approx(-50, abs=FRAME_STEP_PERCENT)


def test_holding_space_reads_full_brake_and_releasing_clears_it(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "Space")
    page.clock.run_for(2 * FRAME_MS)
    assert hud_percent(page, "brake") == 100
    assert key_lit(page, "Space")

    page.keyboard.up("Space")
    page.clock.run_for(2 * FRAME_MS)
    assert hud_percent(page, "brake") == 0


def test_holding_shift_reads_full_afterburner(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "ShiftLeft")
    page.clock.run_for(2 * FRAME_MS)

    assert hud_percent(page, "afterburner") == 100
    assert key_lit(page, "ShiftLeft")


ROW_LOAD_FACTOR, ROW_THROTTLE = 1, 2
ROW_RUDDER, ROW_BRAKE, ROW_AFTERBURNER = 3, 4, 5
BENCHMARK_MAX_SIDE_G = 2.0
EXPORT_LAG_FRAMES = 8
SLIDER_IDS = ("max_side_load_factor", "max_negative_load_factor", "brake_drag_multiplier", "afterburner_thrust_multiplier")
PRESET_SLIDERS = {
    "benchmark": (2.0, -4.5, 3.0, 1.5),
    "small aerial object": (0.5, -2.0, 2.0, 1.0),
    "light aircraft": (0.3, -1.5, 2.0, 1.0),
    "jet": (1.0, -3.0, 3.0, 1.5),
}


def recorded_rows(page):
    with page.expect_download() as download:
        page.click("#export")
    return json.loads(Path(download.value.path()).read_text())["controls"]


def test_holding_d_for_one_second_records_half_of_max_side_g_in_the_rudder_column(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyD")
    page.clock.run_for(HOLD_MS)

    rudder_fraction = recorded_rows(page)[-1][ROW_RUDDER] / BENCHMARK_MAX_SIDE_G
    assert rudder_fraction == pytest.approx(0.5, abs=EXPORT_LAG_FRAMES * FRAME_STEP_PERCENT / 100)


def test_holding_space_records_brake_one_and_afterburner_zero(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "Space")
    page.clock.run_for(HOLD_MS)

    row = recorded_rows(page)[-1]
    assert row[ROW_BRAKE] == 1
    assert row[ROW_AFTERBURNER] == 0


def test_holding_shift_records_afterburner_one(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "ShiftLeft")
    page.clock.run_for(HOLD_MS)

    assert recorded_rows(page)[-1][ROW_AFTERBURNER] == 1


def test_full_push_records_minus_the_preset_max_negative_g(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "ArrowUp")
    page.clock.run_for(int(2 * RAMP_S * 1000))

    assert recorded_rows(page)[-1][1] == -4.5


def test_selecting_each_preset_sets_the_four_new_sliders(browser):
    page = boot_paused(browser)

    shown = {}
    for preset in PRESET_SLIDERS:
        page.select_option("#preset", preset)
        shown[preset] = tuple(float(page.input_value(f"#{slider}")) for slider in SLIDER_IDS)

    assert shown == PRESET_SLIDERS


def test_holding_w_reaches_exactly_full_throttle_in_the_recorded_row(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyW")
    page.clock.run_for(2 * HOLD_MS)

    assert recorded_rows(page)[-1][ROW_THROTTLE] == 1


def test_holding_shift_at_full_throttle_keeps_the_afterburner_column_set(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyW")
    page.keyboard.down("ShiftLeft")
    page.clock.run_for(2 * HOLD_MS)

    row = recorded_rows(page)[-1]
    assert row[ROW_THROTTLE] == 1
    assert row[ROW_AFTERBURNER] == 1


def test_moving_the_max_negative_g_slider_changes_the_full_push_row(browser):
    page = boot_paused(browser)
    page.fill("#max_negative_load_factor", "-6")
    start_flight(page)

    press_and_wait_for_keydown(page, "ArrowUp")
    page.clock.run_for(int(2 * RAMP_S * 1000))

    assert recorded_rows(page)[-1][ROW_LOAD_FACTOR] == -6
