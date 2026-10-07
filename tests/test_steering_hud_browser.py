import json
import os
import re
from pathlib import Path

import pytest

PAGE_BASE_URL = os.environ.get("PAGE_BASE_URL")
if not PAGE_BASE_URL:
    pytest.skip("browser test: set PAGE_BASE_URL to a served repo root", allow_module_level=True)

from playwright import sync_api  # noqa: E402

BOOT_TIMEOUT_MS = 240_000
FRAME_MS = 16
RAMP_S = 0.5
HOLD_MS = 1000
HALF_RAMP_MS = int(RAMP_S * 500)
KEYDOWN_POLL_MS = 100
KEYDOWN_POLLS = 100
FRAME_STEP_PERCENT = 100 * FRAME_MS / 1000 / RAMP_S


@pytest.fixture()
def browser():
    with sync_api.sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        yield chromium
        chromium.close()


def boot_paused(browser, init_script=None):
    page = browser.new_context().new_page()
    if init_script:
        page.add_init_script(init_script)
    page.clock.install(time=0)
    page.clock.pause_at(FRAME_MS)
    page.goto(f"{PAGE_BASE_URL}/web/index.html")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('paused')", polling=200, timeout=BOOT_TIMEOUT_MS)
    return page


def start_flight(page):
    page.click("#start")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('flying')", polling=200, timeout=BOOT_TIMEOUT_MS)
    page.bring_to_front()


def flying_page(browser, init_script=None):
    page = boot_paused(browser, init_script)
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
    return page.locator(f'[data-hud-key~="{code}"]').get_attribute("data-held") == "true"


def test_holding_right_arrow_for_a_quarter_second_reads_half_roll_with_the_key_lit(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "ArrowRight")
    page.clock.run_for(HALF_RAMP_MS)

    assert hud_percent(page, "roll") == pytest.approx(50, abs=FRAME_STEP_PERCENT)
    assert key_lit(page, "ArrowRight")


def test_releasing_right_arrow_unlights_the_key_and_ramps_roll_back_to_zero_within_half_a_second(browser):
    page = flying_page(browser)
    press_and_wait_for_keydown(page, "ArrowRight")
    page.clock.run_for(HOLD_MS)

    page.keyboard.up("ArrowRight")
    page.clock.run_for(int(RAMP_S * 1000) + 2 * FRAME_MS)

    assert not key_lit(page, "ArrowRight")
    assert hud_percent(page, "roll") == pytest.approx(0, abs=FRAME_STEP_PERCENT)


def test_holding_d_for_a_quarter_second_reads_half_rudder_with_the_key_lit(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyD")
    page.clock.run_for(HALF_RAMP_MS)

    assert hud_percent(page, "rudder") == pytest.approx(50, abs=FRAME_STEP_PERCENT)
    assert key_lit(page, "KeyD")


def test_holding_a_reads_negative_rudder(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyA")
    page.clock.run_for(HALF_RAMP_MS)

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


def test_holding_d_for_a_quarter_second_records_half_of_max_side_g_in_the_rudder_column(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyD")
    page.clock.run_for(HALF_RAMP_MS)

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


def test_holding_shift_right_records_afterburner_one(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "ShiftRight")
    page.clock.run_for(HOLD_MS)

    assert recorded_rows(page)[-1][ROW_AFTERBURNER] == 1


def test_max_side_g_slider_scales_the_rudder_column_at_full_deflection(browser):
    page = boot_paused(browser)
    page.fill("#max_side_load_factor", "1.5")
    start_flight(page)

    press_and_wait_for_keydown(page, "KeyD")
    page.clock.run_for(int(2 * RAMP_S * 1000))

    assert recorded_rows(page)[-1][ROW_RUDDER] == 1.5


def test_rudder_column_climbs_the_ramp_in_exact_frame_steps(browser):
    page = flying_page(browser)

    press_and_wait_for_keydown(page, "KeyD")
    page.clock.run_for(HALF_RAMP_MS)

    fractions = [row[ROW_RUDDER] / BENCHMARK_MAX_SIDE_G for row in recorded_rows(page)]
    frame_steps = sorted({fraction / (FRAME_STEP_PERCENT / 100) for fraction in fractions})
    gaps = [later - earlier for earlier, later in zip(frame_steps, frame_steps[1:])]
    assert all(gap == pytest.approx(round(gap), abs=1e-6) for gap in gaps)
    assert 0.4 <= max(fractions) <= 0.5 + 1e-9


LEGACY_SCENARIO_URL = "../data/scenarios/legacy_break_turn_9g.json"
COMMITTED_SCENARIO = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "break_turn_9g.json"
LEGACY_MAX_LOAD_FACTOR = 9.0
LEGACY_CONTROL_COLUMNS = 3


def legacy_form_of_committed_scenario():
    scenario = json.loads(COMMITTED_SCENARIO.read_text())
    scenario["envelope"] = {key: value for key, value in scenario["envelope"].items() if key not in SLIDER_IDS}
    scenario["controls"] = [row[:LEGACY_CONTROL_COLUMNS] for row in scenario["controls"]]
    return json.dumps(scenario)


def test_a_legacy_scenario_gets_neutral_defaults_for_the_new_limits(browser):
    page = browser.new_context().new_page()
    page.route(f"{PAGE_BASE_URL}/data/scenarios/legacy_break_turn_9g.json", lambda route: route.fulfill(content_type="application/json", body=legacy_form_of_committed_scenario()))
    page.clock.install(time=0)
    page.clock.pause_at(FRAME_MS)
    page.goto(f"{PAGE_BASE_URL}/web/index.html?scenario={LEGACY_SCENARIO_URL}")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('flying')", polling=200, timeout=BOOT_TIMEOUT_MS)
    page.clock.run_for(HOLD_MS)

    shown = tuple(float(page.input_value(f"#{slider}")) for slider in SLIDER_IDS)
    with page.expect_download() as download:
        page.click("#export")
    flown = json.loads(Path(download.value.path()).read_text())["envelope"]
    assert shown == (0.0, -LEGACY_MAX_LOAD_FACTOR, 1.0, 1.0)
    assert tuple(flown[key] for key in SLIDER_IDS) == shown


WORKER_REPLY_POLLS = 600
WORKER_REPLY_POLL_MS = 50


def run_until(page, condition):
    for _ in range(WORKER_REPLY_POLLS):
        page.clock.run_for(4 * FRAME_MS)
        if page.evaluate(condition):
            return True
        page.wait_for_timeout(WORKER_REPLY_POLL_MS)
    return False


def hold_engaged_js(name, engaged):
    return f"document.querySelector('[data-hud-hold=\"{name}\"]').dataset.engaged === '{str(engaged).lower()}'"


def test_j_engages_altitude_hold_and_pulling_the_stick_disengages_it(browser):
    page = flying_page(browser)

    page.keyboard.press("KeyJ")
    assert run_until(page, hold_engaged_js("altitude", True))

    press_and_wait_for_keydown(page, "ArrowDown")
    assert run_until(page, hold_engaged_js("altitude", False))
    page.clock.run_for(HOLD_MS)
    assert recorded_rows(page)[-1][ROW_LOAD_FACTOR] > 1


BUTTON_A, BUTTON_B, LEFT_BUMPER, RIGHT_BUMPER = 0, 1, 4, 5
STANDARD_BUTTON_COUNT = 17


def gamepad_script(*held):
    return f"""
        const held = new Set({list(held)});
        const buttons = Array.from({{ length: {STANDARD_BUTTON_COUNT} }}, (_, index) => ({{ pressed: held.has(index), value: held.has(index) ? 1 : 0 }}));
        navigator.getGamepads = () => [{{ mapping: "standard", axes: [0, 0, 0, 0], buttons }}];
    """


def test_right_bumper_reads_full_right_rudder_and_records_full_side_g(browser):
    page = flying_page(browser, gamepad_script(RIGHT_BUMPER))

    page.clock.run_for(HOLD_MS)

    assert hud_percent(page, "rudder") == 100
    assert recorded_rows(page)[-1][ROW_RUDDER] == BENCHMARK_MAX_SIDE_G


def test_left_bumper_reads_full_left_rudder_and_records_negative_side_g(browser):
    page = flying_page(browser, gamepad_script(LEFT_BUMPER))

    page.clock.run_for(HOLD_MS)

    assert hud_percent(page, "rudder") == -100
    assert recorded_rows(page)[-1][ROW_RUDDER] == -BENCHMARK_MAX_SIDE_G


def test_both_bumpers_cancel_to_zero_rudder(browser):
    page = flying_page(browser, gamepad_script(LEFT_BUMPER, RIGHT_BUMPER))

    page.clock.run_for(HOLD_MS)

    assert hud_percent(page, "rudder") == 0


def test_gamepad_b_reads_full_brake_and_records_brake_one_afterburner_zero(browser):
    page = flying_page(browser, gamepad_script(BUTTON_B))

    page.clock.run_for(HOLD_MS)

    row = recorded_rows(page)[-1]
    assert hud_percent(page, "brake") == 100
    assert row[ROW_BRAKE] == 1
    assert row[ROW_AFTERBURNER] == 0


def test_gamepad_a_reads_full_afterburner_and_records_afterburner_one_brake_zero(browser):
    page = flying_page(browser, gamepad_script(BUTTON_A))

    page.clock.run_for(HOLD_MS)

    row = recorded_rows(page)[-1]
    assert hud_percent(page, "afterburner") == 100
    assert row[ROW_AFTERBURNER] == 1
    assert row[ROW_BRAKE] == 0


PAGE_JS = Path(__file__).resolve().parents[1] / "web" / "page.js"
BINDING_DECLARATIONS = tuple(f"const {name} " for name in ("STEERING_KEYS", "MANOEUVRE_KEYS", "HOLD_KEYS"))
KEY_CODE = re.compile(r"\b(?:Arrow[A-Z]\w+|Key[A-Z]|Digit\d|Space|Shift(?:Left|Right))\b")


def bound_key_codes():
    declarations = [line for line in PAGE_JS.read_text().splitlines() if line.startswith(BINDING_DECLARATIONS)]
    assert len(declarations) == len(BINDING_DECLARATIONS)
    return {code for line in declarations for code in KEY_CODE.findall(line)}


def test_every_key_bound_in_page_js_appears_in_the_controls_table(browser):
    page = boot_paused(browser)

    listed = set(page.eval_on_selector_all("#controls-table [data-code]", "cells => cells.map(cell => cell.dataset.code)"))

    assert bound_key_codes() - listed == set()


@pytest.mark.parametrize("code", ["ShiftLeft", "ShiftRight"])
def test_holding_either_shift_lights_the_one_afterburner_chip(browser, code):
    page = flying_page(browser)
    assert page.locator('[data-hud-key~="ShiftLeft"], [data-hud-key~="ShiftRight"]').count() == 1

    press_and_wait_for_keydown(page, code)
    page.clock.run_for(2 * FRAME_MS)

    assert key_lit(page, code)
    assert key_lit(page, "ShiftRight" if code == "ShiftLeft" else "ShiftLeft")


BENCHMARK_MAX_LOAD_FACTOR = 9.0
LOWERED_MAX_LOAD_FACTOR = 3.0


def exported_scenario(page):
    with page.expect_download() as download:
        page.click("#export")
    return json.loads(Path(download.value.path()).read_text())


def test_lowering_max_g_mid_pull_records_the_new_max_from_the_step_the_slider_moved(browser):
    page = flying_page(browser)
    press_and_wait_for_keydown(page, "ArrowDown")
    page.clock.run_for(int(2 * RAMP_S * 1000))
    moved_at_step = len(exported_scenario(page)["controls"])

    page.fill("#max_load_factor", f"{LOWERED_MAX_LOAD_FACTOR:g}")
    page.clock.run_for(HOLD_MS)

    scenario = exported_scenario(page)
    rows = scenario["controls"]
    assert rows[-1][ROW_LOAD_FACTOR] == LOWERED_MAX_LOAD_FACTOR
    assert (rows[moved_at_step - 1][ROW_LOAD_FACTOR], rows[moved_at_step][ROW_LOAD_FACTOR]) == (
        BENCHMARK_MAX_LOAD_FACTOR,
        LOWERED_MAX_LOAD_FACTOR,
    )
    assert [[step, envelope["max_load_factor"]] for step, envelope in scenario["envelope_changes"]] == [
        [moved_at_step, LOWERED_MAX_LOAD_FACTOR]
    ]
    assert scenario["envelope"]["max_load_factor"] == BENCHMARK_MAX_LOAD_FACTOR


def test_max_g_moved_while_paused_applies_from_the_first_step_after_resume(browser):
    page = flying_page(browser)
    page.clock.run_for(HOLD_MS)
    page.click('[data-speed="0"]')
    paused_at_step = len(exported_scenario(page)["controls"])

    page.fill("#max_load_factor", f"{LOWERED_MAX_LOAD_FACTOR:g}")
    page.clock.run_for(HOLD_MS)
    assert len(exported_scenario(page)["controls"]) == paused_at_step
    page.click('[data-speed="1"]')
    page.clock.run_for(HOLD_MS)

    changes = exported_scenario(page)["envelope_changes"]
    assert [[step, envelope["max_load_factor"]] for step, envelope in changes] == [
        [paused_at_step, LOWERED_MAX_LOAD_FACTOR]
    ]
