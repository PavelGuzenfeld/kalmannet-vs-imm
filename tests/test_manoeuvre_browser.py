import math

import pytest

from tests.test_steering_hud_browser import (  # noqa: F401
    boot_paused,
    browser,
    flying_page,
    hold_engaged_js,
    press_and_wait_for_keydown,
    run_until,
    start_flight,
)
from playwright.sync_api import TimeoutError as PlaywrightTimeout  # noqa: E402

SETTLE_STEP_MS = 200
SETTLE_STEPS = 60
HUD_POLL_TIMEOUT_MS = 500


def hud_manoeuvre(page):
    return page.locator("#manoeuvre-hud").get_attribute("data-manoeuvre")


def run_until_hud_names(page, name):
    for _ in range(SETTLE_STEPS):
        page.clock.run_for(SETTLE_STEP_MS)
        try:
            page.wait_for_function(
                "name => document.getElementById('manoeuvre-hud').dataset.manoeuvre === name",
                arg=name,
                timeout=HUD_POLL_TIMEOUT_MS,
            )
            return
        except PlaywrightTimeout:
            continue
    pytest.fail(f"HUD never read {name!r}, last read {hud_manoeuvre(page)!r}")


def test_pressing_1_names_the_break_turn_on_the_hud_and_an_arrow_key_cancels_it(browser):  # noqa: F811
    page = flying_page(browser)
    assert hud_manoeuvre(page) == ""

    press_and_wait_for_keydown(page, "Digit1")
    run_until_hud_names(page, "break turn")
    assert page.locator("#manoeuvre-hud").inner_text() == "break turn"

    page.keyboard.down("ArrowDown")
    run_until_hud_names(page, "")
    assert page.locator("#manoeuvre-hud").inner_text() == ""


HEAD_ON_SEED = 4240
DT_S = 0.02
EVADE_WITHIN_S = 5.0
BEAM_DEG = 90.0
BEAM_TOLERANCE_DEG = 15.0
HEAD_ON_MIN_DEG = 175.0
FRAME_RECORDER = """
window.flownFrames = [];
const PageWorker = window.Worker;
window.Worker = class extends PageWorker {
  set onmessage(handler) {
    super.onmessage = (event) => {
      if (event.data.kind === "frame") window.flownFrames.push(event.data.frame);
      handler(event);
    };
  }
};
"""
EVADE_HUD = "document.querySelector('[data-hud-evade]').dataset.engaged === '{}'"


def angle_to_line_of_sight_deg(frame):
    position, velocity = frame["truth_m"], frame["truth_velocity_m_s"]
    cosine = sum(p * v for p, v in zip(position, velocity)) / (math.dist(position, (0, 0, 0)) * math.dist(velocity, (0, 0, 0)))
    return math.degrees(math.acos(cosine))


def head_on_flight(browser):  # noqa: F811
    page = boot_paused(browser, FRAME_RECORDER)
    page.fill("#seed", str(HEAD_ON_SEED))
    start_flight(page)
    return page


def test_e_turns_a_head_on_target_across_the_line_of_sight_within_5_s_and_an_arrow_key_takes_it_back(browser):  # noqa: F811
    page = head_on_flight(browser)
    assert run_until(page, "window.flownFrames.length > 0")

    press_and_wait_for_keydown(page, "KeyE")
    assert run_until(page, EVADE_HUD.format("true"))
    frames = page.evaluate("window.flownFrames")
    engaged_at = max(frame["step"] for frame in frames if not frame["evade"])
    deadline = engaged_at + round(EVADE_WITHIN_S / DT_S)
    assert run_until(page, f"window.flownFrames.at(-1).step >= {deadline}")
    frames = page.evaluate("window.flownFrames")
    at_engage = next(frame for frame in frames if frame["step"] == engaged_at)
    by_deadline = [frame for frame in frames if frame["step"] <= deadline][-1]

    assert angle_to_line_of_sight_deg(at_engage) > HEAD_ON_MIN_DEG
    assert abs(angle_to_line_of_sight_deg(by_deadline) - BEAM_DEG) <= BEAM_TOLERANCE_DEG
    assert page.locator("[data-hud-evade]").inner_text() == "E evade"

    page.keyboard.down("ArrowLeft")
    assert run_until(page, EVADE_HUD.format("false"))
    assert not page.evaluate("window.flownFrames.at(-1).evade")


def test_the_flight_panel_checkbox_engages_evade_and_unticks_when_a_hold_takes_over(browser):  # noqa: F811
    page = head_on_flight(browser)

    page.check("#evade")
    assert run_until(page, EVADE_HUD.format("true"))
    page.keyboard.press("KeyH")
    assert run_until(page, EVADE_HUD.format("false"))

    assert not page.is_checked("#evade")
    assert run_until(page, hold_engaged_js("heading", True))


def test_pressing_e_a_second_time_turns_evade_off(browser):  # noqa: F811
    page = head_on_flight(browser)

    press_and_wait_for_keydown(page, "KeyE")
    assert run_until(page, EVADE_HUD.format("true"))
    page.keyboard.up("KeyE")
    page.keyboard.press("KeyE")

    assert run_until(page, EVADE_HUD.format("false"))
    assert not page.evaluate("window.flownFrames.at(-1).evade")
