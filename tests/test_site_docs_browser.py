import os
from urllib.parse import urlsplit

import pytest

PAGE_BASE_URL = os.environ.get("SITE_BASE_URL")
if not PAGE_BASE_URL:
    pytest.skip("browser test: set SITE_BASE_URL to a served _site", allow_module_level=True)

from playwright import sync_api  # noqa: E402

RENDER_TIMEOUT_MS = 30_000


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


@pytest.mark.parametrize(
    "query",
    ["?doc=../README.md", "?doc=https://example.com/x.md", "?doc=//example.com/x.md", "?doc=%2e%2e%2fREADME.md", "?doc=", ""],
)
def test_unsafe_or_absent_doc_name_is_refused_without_a_foreign_request(browser_context, query):
    page = browser_context.new_page()
    foreign_requests = []
    page.on("request", lambda request: foreign_requests.append(request.url) if urlsplit(request.url).hostname == "example.com" else None)

    page.goto(f"{PAGE_BASE_URL}/docs/view.html{query}")

    page.get_by_text("Unknown document.").wait_for(timeout=RENDER_TIMEOUT_MS)
    assert foreign_requests == []


def test_well_formed_but_missing_doc_name_says_not_found(browser_context):
    page = browser_context.new_page()

    page.goto(f"{PAGE_BASE_URL}/docs/view.html?doc=no-such-doc.md")

    page.get_by_text("Document not found.").wait_for(timeout=RENDER_TIMEOUT_MS)


def test_home_link_from_the_source_notes_lands_on_the_landing_page_under_the_pages_subpath(browser_context):
    page = open_doc_from_landing(browser_context, "Write-up")
    page.wait_for_selector("table", timeout=RENDER_TIMEOUT_MS)
    page.locator('a[href*="source-notes"]').first.click()
    page.get_by_text("Research notes for the benchmark").wait_for(timeout=RENDER_TIMEOUT_MS)

    page.get_by_role("link", name="Home").click()

    page.wait_for_url(f"{PAGE_BASE_URL}/", timeout=RENDER_TIMEOUT_MS)
    page.get_by_role("link", name="Tracking test ground").wait_for(timeout=RENDER_TIMEOUT_MS)
