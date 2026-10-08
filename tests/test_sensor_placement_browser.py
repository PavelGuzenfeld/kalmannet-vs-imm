import json
from pathlib import Path

from tests.test_manoeuvre_browser import FRAME_RECORDER
from tests.test_steering_hud_browser import (  # noqa: F401
    BOOT_TIMEOUT_MS,
    FRAME_MS,
    PAGE_BASE_URL,
    boot_paused,
    browser,
    run_until,
    start_flight,
)
from tests.test_target_start_browser import exported_scenario

BREAK_TURN = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "break_turn_9g.json"
SENSOR_M = [1000.0, -500.0, 0.0]
PLACED_START = {"position": [3123.456, 987.654, 300.0], "velocity": [-150.0, 0.0, 0.0]}
SHIFTED_START = {
    "position": [p - s for p, s in zip(PLACED_START["position"], SENSOR_M)],
    "velocity": PLACED_START["velocity"],
}
SCRIPTED_STEPS = 120
FRAME_POLL_MS = 20
SENSOR_FIELDS = {"sensor_east_m": "1000", "sensor_north_m": "-500", "sensor_altitude_m": "0"}


def break_turn_with(start, sensor=None):
    scenario = json.loads(BREAK_TURN.read_text())
    scenario["controls"] = scenario["controls"][:SCRIPTED_STEPS]
    scenario["start"] = start
    if sensor is not None:
        scenario["sensor"] = {"position": sensor}
    return scenario


def single_stepped_frames(browser, name, scenario):  # noqa: F811
    page = browser.new_context().new_page()
    page.add_init_script(FRAME_RECORDER)
    url = f"{PAGE_BASE_URL}/build/{name}.json"
    page.route(url, lambda route: route.fulfill(content_type="application/json", body=json.dumps(scenario)))
    page.clock.install(time=0)
    page.clock.pause_at(FRAME_MS)
    page.goto(f"{PAGE_BASE_URL}/web/index.html?scenario={url}")
    page.wait_for_function(
        "document.getElementById('status').textContent.startsWith('flying the scripted')",
        polling=200,
        timeout=BOOT_TIMEOUT_MS,
    )
    for step in range(1, SCRIPTED_STEPS + 1):
        page.click("#single_step")
        page.wait_for_function(
            f"window.flownFrames.length >= {step}", polling=FRAME_POLL_MS, timeout=BOOT_TIMEOUT_MS
        )
    with page.expect_download() as download:
        page.click("#export")
    frames = page.evaluate("window.flownFrames")
    assert [frame["step"] for frame in frames] == list(range(1, SCRIPTED_STEPS + 1))
    return page, frames, json.loads(Path(download.value.path()).read_text())


def shifted(position, sensor):
    return [p + s for p, s in zip(position, sensor)]


def test_placed_sensor_flies_every_filter_as_the_origin_sensor_flies_the_start_shifted_by_minus_the_sensor(
    browser,  # noqa: F811
):
    placed_page, placed, exported = single_stepped_frames(
        browser, "sensor_placed", break_turn_with(PLACED_START, SENSOR_M)
    )
    _, origin, _ = single_stepped_frames(browser, "sensor_origin", break_turn_with(SHIFTED_START))

    assert (exported["sensor"], exported["start"]) == ({"position": SENSOR_M}, PLACED_START)
    for at_placed, at_origin in zip(placed, origin, strict=True):
        assert at_placed["truth_m"] == shifted(at_origin["truth_m"], SENSOR_M)
        assert at_placed["trips"] == at_origin["trips"]
        assert at_placed["filters"].keys() == at_origin["filters"].keys() >= {"EKF", "UKF", "IMM"}
        for name, readout in at_placed["filters"].items():
            other = at_origin["filters"][name]
            assert readout["estimate_m"] == shifted(other["estimate_m"], SENSOR_M)
            assert {**readout, "estimate_m": None} == {**other, "estimate_m": None}
    assert placed_page.locator("#scene").get_attribute("data-sensor-position") == "1000 -500 0"
    assert placed_page.locator("#plan").get_attribute("data-sensor-m") == "1000 -500"


def test_sensor_placed_from_the_flight_panel_is_exported_and_an_untouched_panel_exports_none(browser):  # noqa: F811
    page = boot_paused(browser)
    for field, value in SENSOR_FIELDS.items():
        page.fill(f"#{field}", value)
    start_flight(page)
    untouched = boot_paused(browser)
    start_flight(untouched)

    assert exported_scenario(page)["sensor"] == {"position": SENSOR_M}
    assert "sensor" not in exported_scenario(untouched)
