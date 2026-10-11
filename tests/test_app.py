"""Headless UI smoke tests with Streamlit's AppTest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from ferna_cv.pipeline import ScreeningConfig, analyze_file  # noqa: E402

APP = str(ROOT / "app.py")


@pytest.fixture(scope="module")
def jd():
    return (ROOT / "samples" / "job_description.txt").read_text(encoding="utf-8")


def test_app_loads():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("No CVs analysed yet" in i.value for i in at.info)


def test_autofill_and_leaderboard(jd):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["jd_text"] = jd
    recs = {}
    for p in sorted((ROOT / "samples" / "cvs").iterdir()):
        r = analyze_file(p.name, p.read_bytes(), jd, ScreeningConfig())
        recs[r.file_hash] = r
    at.session_state["records"] = recs
    at.run()
    assert not at.exception
    # click "Analyse JD & auto-fill checklist"
    btn = next(b for b in at.button if "Analyse JD" in b.label)
    btn.click().run()
    assert not at.exception
    assert at.session_state["exp_min"] == 4.0
    assert "python" in at.session_state["skill_list"]
    assert any("Checklist filled" in s.value for s in at.success)
    # leaderboard metrics rendered
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("Candidates") == "7"
    assert labels.get("✅ Qualified") == "2"
    # custom weights that do not total 100 block scoring
    at.radio(key="weight_mode").set_value("Custom weight matrix").run()
    at.number_input(key="w_mand").set_value(70).run()
    assert any("total" in e.value and "100" in e.value for e in at.error)
    at.number_input(key="w_mand").set_value(50).run()
    assert not at.exception
    # strict location rule changes outcomes live (no re-processing)
    at.radio(key="loc_enf").set_value("Hard").run()
    assert not at.exception


def test_deep_dive_dialog(jd):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["jd_text"] = jd
    recs = {}
    for p in sorted((ROOT / "samples" / "cvs").iterdir()):
        r = analyze_file(p.name, p.read_bytes(), jd, ScreeningConfig())
        recs[r.file_hash] = r
    at.session_state["records"] = recs
    at.run()
    next(b for b in at.button if "Analyse JD" in b.label).click().run()
    next(b for b in at.button if "deep-dive" in b.label).click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Stage 1" in text and "Stage 2" in text
