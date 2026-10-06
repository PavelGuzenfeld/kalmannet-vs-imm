import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FLAGS = ("track_loss", "overconfidence", "raw_error")
PANELS = ("altitude", "error", "nees", "modes")
FLAG_PATTERN = re.compile(rf"(\w+): .*?\b({'|'.join(FLAGS)})=(\d+|-)")
FINISHED_PATTERN = re.compile(r"script finished at step (\d+)")
BOOT_AND_FLIGHT_TIMEOUT_MS = 300_000


def replay_trips(replay_output):
    trips = {}
    for line in replay_output.splitlines():
        name = line.split(":")[0]
        pairs = re.findall(rf"({'|'.join(FLAGS)})=(\d+|-)", line)
        if pairs:
            trips[name] = {flag: None if step == "-" else int(step) for flag, step in pairs}
    return trips


def plotted_markers(page):
    return page.eval_on_selector_all(
        ".trip-marker",
        "markers => markers.map(m => [m.dataset.panel, m.dataset.filter, m.dataset.flag, Number(m.dataset.tripStep)])",
    )


def check_scenario(page, base_url, scenario, expected):
    page.goto(f"{base_url}/web/index.html?scenario=../data/scenarios/{scenario}.json")
    page.wait_for_function(
        "document.getElementById('status').textContent.startsWith('script finished')",
        timeout=BOOT_AND_FLIGHT_TIMEOUT_MS,
    )
    markers = plotted_markers(page)
    failures = []
    for name, flags in expected.items():
        for flag, step in flags.items():
            plotted = {(panel, s) for panel, n, f, s in markers if (n, f) == (name, flag)}
            wanted = {(panel, step) for panel in PANELS} if step is not None else set()
            if plotted != wanted:
                failures.append(f"{scenario} {name} {flag}: replay {step}, plotted {sorted(plotted)}")
    tracked = [m for m in markers if m[2] == "track_loss"]
    if not tracked:
        failures.append(f"{scenario}: no track-loss marker plotted")
    return failures


def main(base_url, replay_dir):
    failures = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for output in sorted(Path(replay_dir).glob("*.txt")):
            page = browser.new_page()
            page.on("pageerror", lambda error: failures.append(f"page error: {error}"))
            failures += check_scenario(page, base_url, output.stem, replay_trips(output.read_text()))
            page.close()
        browser.close()
    print("\n".join(failures) or "page views match replay.py")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
