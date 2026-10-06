import json
import re
import sys
from pathlib import Path

import pytest

from scripts.build_site import main as build_site_main
from web.live_flight import LEARNED_WEIGHTS

REPO = Path(__file__).resolve().parents[1]
TAG = "v9.8.7"
OTHER_TAG = "v0.0.1"
WORKER_FETCHED_SOURCES = [
    "data/fixed_wing.py",
    "filters/knet/numpy_forward.py",
    "scripts/eval.py",
    "web/live_flight.py",
    "web/index.html",
    "data/scenarios/overhead_pass.json",
]


def build(tmp_path, monkeypatch, tag=TAG):
    results = tmp_path / "results"
    for relative in LEARNED_WEIGHTS.values():
        weights = results / Path(relative).relative_to("results")
        weights.parent.mkdir(parents=True)
        weights.write_bytes(relative.encode())
    out = tmp_path / "_site"
    monkeypatch.setattr(sys, "argv", ["build_site.py", "--tag", tag, "--results", str(results), "--out", str(out)])
    build_site_main()
    return out


@pytest.fixture
def site(tmp_path, monkeypatch):
    stray_bytecode = REPO / "data" / "__pycache__" / "stray.pyc"
    stray_bytecode.parent.mkdir(exist_ok=True)
    stray_bytecode.write_bytes(b"")
    yield build(tmp_path, monkeypatch)
    stray_bytecode.unlink()


@pytest.mark.parametrize("relative", WORKER_FETCHED_SOURCES)
def test_site_serves_the_commits_own_copy_of_each_file_the_worker_fetches(site, relative):
    assert (site / relative).read_bytes() == (REPO / relative).read_bytes()


@pytest.mark.parametrize("relative", LEARNED_WEIGHTS.values())
def test_site_serves_each_filters_own_weights_where_the_worker_looks(site, relative):
    assert (site / relative).read_bytes() == relative.encode()


@pytest.mark.parametrize("tag", [TAG, OTHER_TAG])
def test_site_stamps_the_tag_it_was_given(tmp_path, monkeypatch, tag):
    out = build(tmp_path, monkeypatch, tag)
    assert json.loads((out / "web" / "version.json").read_text()) == {"tag": tag}


def test_page_reads_the_tag_stamp(site):
    assert 'fetch("version.json")' in (site / "web" / "index.html").read_text()


def landing_links(site):
    return re.findall(r'href="([^"]+)"', (site / "index.html").read_text())


@pytest.mark.parametrize("target", ["web/", "docs/view.html?doc=write-up.md", "docs/view.html?doc=source-notes.md"])
def test_landing_page_links_the_test_ground_and_both_docs(site, target):
    assert target in landing_links(site)


def test_landing_page_does_not_redirect(site):
    assert "http-equiv" not in (site / "index.html").read_text()


@pytest.mark.parametrize("tag", [TAG, OTHER_TAG])
def test_landing_page_shows_the_tag_it_was_given(tmp_path, monkeypatch, tag):
    out = build(tmp_path, monkeypatch, tag)
    assert f"built from {tag}" in (out / "index.html").read_text()


@pytest.mark.parametrize("doc", ["write-up.md", "source-notes.md"])
def test_site_serves_the_commits_own_copy_of_each_doc(site, doc):
    assert (site / "docs" / doc).read_bytes() == (REPO / "docs" / doc).read_bytes()


@pytest.mark.parametrize("doc", ["write-up.md", "source-notes.md"])
def test_docs_do_not_cross_reference_by_issue_number_or_issue_link(doc):
    text = (REPO / "docs" / doc).read_text()
    assert not re.search(r"(?<![\w\[])#1[45]\b", text)
    assert not re.search(r"issues/1[45]\b", text)


def test_readme_points_at_the_site_not_at_the_two_issues():
    text = (REPO / "README.md").read_text()
    assert "https://pavelguzenfeld.github.io/kalmannet-vs-imm/" in text
    assert not re.search(r"issues/1[45]\b", text)


def test_write_up_links_the_source_notes_by_relative_path():
    assert "[#14](source-notes.md)" in (REPO / "docs" / "write-up.md").read_text()


def test_site_excludes_python_bytecode(site):
    assert not list(site.rglob("__pycache__"))
