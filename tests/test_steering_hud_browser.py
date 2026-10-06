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
FRAME_STEP_PERCENT = 100 * FRAME_MS / 1000 / RAMP_S


@pytest.fixture(scope="module")
def browser_context():
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser.new_context()
        browser.close()


def flying_page(context):
    page = context.new_page()
    page.clock.install()
    page.goto(f"{PAGE_BASE_URL}/web/index.html")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('paused')", polling=200, timeout=BOOT_TIMEOUT_MS)
    page.click("#start")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('flying')", polling=200, timeout=BOOT_TIMEOUT_MS)
    page.bring_to_front()
    page.clock.pause_at(page.evaluate("Date.now()") + FRAME_MS)
    return page


def hud_percent(page, axis):
    return float(page.locator(f'[data-hud-axis="{axis}"]').get_attribute("data-percent"))


def key_lit(page, code):
    return page.locator(f'[data-hud-key="{code}"]').get_attribute("data-held") == "true"


def test_holding_right_arrow_for_one_second_reads_half_roll_with_the_key_lit(browser_context):
    page = flying_page(browser_context)

    page.keyboard.down("ArrowRight")
    page.clock.run_for(HOLD_MS)

    assert hud_percent(page, "roll") == pytest.approx(50, abs=FRAME_STEP_PERCENT)
    assert key_lit(page, "ArrowRight")


def test_releasing_right_arrow_unlights_the_key_and_ramps_roll_back_to_zero_within_one_second(browser_context):
    page = flying_page(browser_context)
    page.keyboard.down("ArrowRight")
    page.clock.run_for(HOLD_MS)

    page.keyboard.up("ArrowRight")
    page.clock.run_for(HOLD_MS + 2 * FRAME_MS)

    assert not key_lit(page, "ArrowRight")
    assert hud_percent(page, "roll") == pytest.approx(0, abs=FRAME_STEP_PERCENT)
