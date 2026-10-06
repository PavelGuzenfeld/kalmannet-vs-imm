import os

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


def flying_page(browser):
    page = browser.new_context().new_page()
    page.clock.install(time=0)
    page.clock.pause_at(FRAME_MS)
    page.goto(f"{PAGE_BASE_URL}/web/index.html")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('paused')", polling=200, timeout=BOOT_TIMEOUT_MS)
    page.click("#start")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('flying')", polling=200, timeout=BOOT_TIMEOUT_MS)
    page.bring_to_front()
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
