import pytest

from tests.test_steering_hud_browser import browser, flying_page, press_and_wait_for_keydown  # noqa: F401
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
