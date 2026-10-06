import os

import pytest

PAGE_BASE_URL = os.environ.get("SITE_BASE_URL")
if not PAGE_BASE_URL:
    pytest.skip("browser test: set SITE_BASE_URL to a served _site", allow_module_level=True)

from playwright import sync_api  # noqa: E402

RENDER_TIMEOUT_MS = 30_000
MATH_FIXTURE = "Energy is $E = mc^2$ here.\n"


@pytest.fixture(scope="module")
def browser_context():
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser.new_context()
        browser.close()


def open_doc_from_landing(context, link_text):
    page = context.new_page()
    page.goto(f"{PAGE_BASE_URL}/index.html")
    page.get_by_role("link", name=link_text).click()
    return page


def test_write_up_results_table_renders_as_a_table(browser_context):
    page = open_doc_from_landing(browser_context, "Write-up")

    page.wait_for_selector("table", timeout=RENDER_TIMEOUT_MS)

    headers = page.eval_on_selector_all("table th", "cells => cells.map(c => c.textContent)")
    assert "Pos RMSE (m)" in headers
    assert page.locator("table td", has_text="112.59").count() > 0


def test_source_notes_link_from_write_up_opens_source_notes(browser_context):
    page = open_doc_from_landing(browser_context, "Write-up")
    page.wait_for_selector("table", timeout=RENDER_TIMEOUT_MS)

    page.locator('a[href*="source-notes"]').first.click()

    page.wait_for_url("**/view.html?doc=source-notes.md", timeout=RENDER_TIMEOUT_MS)
    page.get_by_text("Research notes for the benchmark").wait_for(timeout=RENDER_TIMEOUT_MS)


def test_inline_math_renders(browser_context):
    page = browser_context.new_page()
    page.route("**/docs/math-fixture.md", lambda route: route.fulfill(body=MATH_FIXTURE, content_type="text/markdown"))

    page.goto(f"{PAGE_BASE_URL}/docs/view.html?doc=math-fixture.md")

    page.wait_for_selector(".katex", timeout=RENDER_TIMEOUT_MS)
