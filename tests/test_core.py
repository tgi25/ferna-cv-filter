"""Unit tests: extraction, weights, heuristic analysis, scoring and (mocked) LLM providers.

    python samples/make_samples.py && pytest -q
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ferna_cv.dates import candidate_age, parse_date, total_experience_years  # noqa: E402
from ferna_cv.extraction import ExtractionError, extract_text, find_phones, redact_pii  # noqa: E402
from ferna_cv.heuristic import analyze_cv, analyze_jd  # noqa: E402
from ferna_cv.llm import LLMClient, LLMConfig, LLMError, analyze_cv_llm, analyze_jd_llm, schema_for  # noqa: E402
from ferna_cv.matching import match_skill  # noqa: E402
from ferna_cv.pipeline import ScreeningConfig, analyze_batch, analyze_file  # noqa: E402
from ferna_cv.schemas import (  # noqa: E402
    CandidateProfile, CandidateStatus, CertificationRule, CVAnalysis, DegreeLevel, Enforcement, ExperienceRule,
    LocationRule, MandatoryCriteria, QualificationRule, SkillsRule, WorkExperience, coerce_level, AgeRule,
)
from ferna_cv.scoring import rank_results, score_candidate  # noqa: E402
from ferna_cv.weights import DEFAULT_RANGES, fit_to_ranges, plan_from_analysis  # noqa: E402

SAMPLES = ROOT / "samples"
CVS = SAMPLES / "cvs"
TODAY = date(2026, 10, 11)


@pytest.fixture(scope="session", autouse=True)
def samples():
    if not (CVS / "Nimal_Perera_CV.docx").exists():
        import subprocess
        subprocess.run([sys.executable, str(SAMPLES / "make_samples.py")], check=True)


@pytest.fixture(scope="session")
def jd():
    return (SAMPLES / "job_description.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------- extraction
def test_extract_docx_and_pdf():
    t1, _ = extract_text("a.docx", (CVS / "Nimal_Perera_CV.docx").read_bytes())
    t2, _ = extract_text("b.pdf", (CVS / "Kasun_Silva_CV.pdf").read_bytes())
    assert "NIMAL PERERA" in t1 and "Kasun Silva" in t2


@pytest.mark.parametrize("name,data,needle", [
    ("corrupted_cv.pdf", None, "corrupted"),
    ("scanned_cv.pdf", None, "OCR"),
    ("empty.pdf", b"", "empty"),
    ("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100, ".doc"),
    ("x.png", b"\x89PNG....", "Unsupported"),
    ("broken.docx", b"PK\x03\x04garbage", "ZIP"),
])
def test_extraction_errors(name, data, needle):
    data = data if data is not None else (CVS / name).read_bytes()
    with pytest.raises(ExtractionError) as exc:
        extract_text(name, data)
    assert needle.lower() in str(exc.value).lower()


def test_phone_detection_ignores_year_ranges():
    assert find_phones("2019 - 2021 and 2015 - 2018") == []
    assert find_phones("Call +94 77 123 4567") == ["+94 77 123 4567"]
    red = redact_pii("me@x.com | 071 234 5678 | linkedin.com/in/me | 2019 - 2021")
    assert "[EMAIL]" in red and "[PHONE]" in red and "[URL]" in red and "2019 - 2021" in red


# ---------------------------------------------------------------- dates
def test_dates_and_experience():
    assert parse_date("14/03/1991")[0] == date(1991, 3, 14)
    assert parse_date("Mar 2018")[1] == "month"
    roles = [WorkExperience(start="2018-01", end="2020-12"), WorkExperience(start="2020-06", end="2021-12"),
             WorkExperience(start="2015", end="2016")]
    # 2018-01..2021-12 merged = 48 months + 12 months = 60 months
    assert total_experience_years(roles, TODAY) == 5.0
    assert candidate_age(CandidateProfile(date_of_birth="1991-03-14"), TODAY) == 35
    assert candidate_age(CandidateProfile(), TODAY) is None


def test_degree_levels():
    assert coerce_level("BSc (Hons) in Statistics") == DegreeLevel.BACHELOR
    assert coerce_level("Master's degree") == DegreeLevel.MASTER
    assert coerce_level("PhD in NLP") == DegreeLevel.DOCTORATE
    assert coerce_level("HND in IT") == DegreeLevel.DIPLOMA
    assert coerce_level("MS Excel and MS Office") == DegreeLevel.NONE
    assert coerce_level("Master data management") == DegreeLevel.NONE


# ---------------------------------------------------------------- weights
def test_fit_to_ranges_random():
    rnd = random.Random(7)
    for _ in range(500):
        raw = {k: rnd.uniform(-10, 120) for k in DEFAULT_RANGES}
        w = fit_to_ranges(raw)
        assert sum(w.values()) == 100
        for k, (lo, hi) in DEFAULT_RANGES.items():
            assert lo <= w[k] <= hi


def test_jd_heuristic(jd):
    a = analyze_jd(jd)
    assert a.min_years_experience == 4
    assert a.min_degree_level == DegreeLevel.BACHELOR
    assert "python" in a.core_skills and "pytorch" in a.nice_to_have_skills
    assert a.locations == ["Colombo"] and a.related_fields_ok
    plan = plan_from_analysis(a)
    assert sum(plan.modules.as_dict().values()) == 100


# ---------------------------------------------------------------- matching
def test_skill_matching():
    assert match_skill("Python", ["python 3.11"]).matched
    assert match_skill("AWS", ["Amazon Web Services"]).matched
    assert not match_skill("Java", ["JavaScript"]).matched
    assert match_skill("Kubernetes", [], "Deployed services on k8s clusters").matched


# ---------------------------------------------------------------- end-to-end scoring (offline)
def _criteria(a):
    return MandatoryCriteria(
        location=LocationRule(enforcement=Enforcement.HARD, accepted_locations=a.locations),
        experience=ExperienceRule(min_years=a.min_years_experience),
        qualification=QualificationRule(min_level=a.min_degree_level, fields=a.degree_fields,
                                        related_fields_ok=a.related_fields_ok),
        certification=CertificationRule(required=a.required_certifications),
        skills=SkillsRule(required=a.core_skills),
    )


def test_pipeline_offline(jd):
    files = [(p.name, p.read_bytes()) for p in sorted(CVS.iterdir())]
    recs = analyze_batch(files, jd, ScreeningConfig())
    a = analyze_jd(jd)
    res = {r.file_name: r for r in rank_results(
        [score_candidate(r, _criteria(a), plan_from_analysis(a), jd, today=TODAY) for r in recs])}
    assert res["Nimal_Perera_CV.docx"].status == CandidateStatus.QUALIFIED
    assert res["Nimal_Perera_CV.docx"].rank == 1
    assert res["Ruwan_Jayasinghe_CV.pdf"].status == CandidateStatus.DISQUALIFIED       # location, no relocation
    assert res["Kasun_Silva_CV.pdf"].status == CandidateStatus.FLAGGED                 # related field + soft cert
    assert res["Dilini_Wickramasinghe_CV.docx"].status == CandidateStatus.DISQUALIFIED  # HND < bachelor
    assert res["corrupted_cv.pdf"].status == CandidateStatus.ERROR
    for r in res.values():
        if r.status != CandidateStatus.ERROR:
            assert 0 <= r.total <= 100
            assert abs(sum(m.points for m in r.modules) - r.total) < 0.11
            assert sum(m.weight for m in r.modules) == 100


def test_unknown_age_is_flagged_not_disqualified(jd):
    rec = analyze_file("Kasun_Silva_CV.pdf", (CVS / "Kasun_Silva_CV.pdf").read_bytes(), jd, ScreeningConfig())
    crit = MandatoryCriteria(age=AgeRule(enforcement=Enforcement.HARD, max_age=40),
                             experience=ExperienceRule(min_years=0))
    r = score_candidate(rec, crit, plan_from_analysis(analyze_jd(jd)), jd, today=TODAY)
    assert r.status == CandidateStatus.FLAGGED
    assert any("Age not stated" in x for x in r.reasons)


# ---------------------------------------------------------------- LLM providers (mocked SDKs)
GOOD = {
    "profile": {
        "full_name": "Test Person", "email": "[EMAIL]", "current_location": "Colombo",
        "total_years_experience": "6+ years", "skills": "Python, SQL; Docker",
        "education": [{"degree": "BSc in Computer Science", "level": "Bachelor's", "end_year": "2018"}],
        "work_experience": [{"title": "DS", "start": "2019-01", "end": "present"}],
        "willing_to_relocate": "unknown", "stated_age": None,
    },
    "assessment": {"academic_professional": {"score": 12, "justification": "x", "evidence": "a; b"},
                   "highlights": {"score": "7/10"}, "extracurricular": {"score": 3}},
}


class FakeOpenAI:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self.payloads.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])


class FakeAnthropic:
    def __init__(self, data):
        self.data = data
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason="tool_use",
                               content=[SimpleNamespace(type="tool_use", name=kw["tool_choice"]["name"], input=self.data)])


def test_openai_structured_extraction_with_coercion():
    fake = FakeOpenAI(["```json\n" + json.dumps(GOOD) + "\n```"])
    client = LLMClient(LLMConfig(provider="openai", api_key="k", model="m"), client=fake)
    out = analyze_cv_llm(client, "cv", "jd")
    p = out.profile
    assert p.total_years_experience == 6 and p.skills == ["Python", "SQL", "Docker"]
    assert p.education[0].level == DegreeLevel.BACHELOR and p.education[0].end_year == 2018
    assert p.willing_to_relocate is None
    assert out.assessment.academic_professional.score == 10  # clamped
    assert out.assessment.highlights.score == 7
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_openai_drops_unsupported_temperature_and_repairs_invalid_json():
    bad = json.dumps({"profile": "not an object", "assessment": [1, 2]})
    fake = FakeOpenAI([Exception("Unsupported parameter: 'temperature'"), bad, json.dumps(GOOD)])
    client = LLMClient(LLMConfig(provider="openai", api_key="k", model="m"), client=fake)
    out = analyze_cv_llm(client, "cv", "jd")
    assert out.profile.full_name == "Test Person"
    assert "temperature" not in fake.calls[1]
    assert "failed schema validation" in fake.calls[2]["messages"][1]["content"]


def test_openai_garbage_raises_llmerror():
    fake = FakeOpenAI(["no json here", "still none"])
    client = LLMClient(LLMConfig(provider="openai", api_key="k", model="m"), client=fake)
    with pytest.raises(LLMError):
        analyze_cv_llm(client, "cv", "jd")


def test_anthropic_tool_use():
    fake = FakeAnthropic({"job_title": "DS", "min_degree_level": "Bachelors", "core_skills": ["Python"],
                          "weights": {"mandatory_compliance": 70, "academic_professional": 10,
                                      "highlights": 10, "extracurricular": 10}, "complexity": "4"})
    client = LLMClient(LLMConfig(provider="anthropic", api_key="k", model="claude-sonnet-5-5"), client=fake)
    a = analyze_jd_llm(client, "jd")
    assert a.min_degree_level == DegreeLevel.BACHELOR and a.complexity == 4
    call = fake.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "record_jd_analysis"}
    assert "$defs" not in json.dumps(call["tools"][0]["input_schema"])
    plan = plan_from_analysis(a)
    assert plan.modules.mandatory_compliance == 60 and sum(plan.modules.as_dict().values()) == 100


def test_schema_inlining_keeps_title_property():
    sch = schema_for(CVAnalysis)
    wexp = sch["properties"]["profile"]["properties"]["work_experience"]["items"]["properties"]
    assert "title" in wexp and "$ref" not in json.dumps(sch)


def test_missing_key_rejected():
    with pytest.raises(LLMError):
        LLMClient(LLMConfig(provider="anthropic", api_key="", model="x"))


def test_pipeline_llm_path_restores_redacted_contacts(jd, monkeypatch):
    data = (CVS / "Nimal_Perera_CV.docx").read_bytes()
    fake = FakeOpenAI([json.dumps(GOOD)])
    client = LLMClient(LLMConfig(provider="openai", api_key="k", model="m"), client=fake)
    rec = analyze_file("Nimal_Perera_CV.docx", data, jd,
                       ScreeningConfig(llm=LLMConfig(provider="openai", api_key="k", model="m")), client)
    assert not rec.error
    sent = fake.calls[0]["messages"][1]["content"]
    assert "nimal.perera@example.com" not in sent and "[EMAIL]" in sent
    assert rec.analysis.profile.email == "nimal.perera@example.com"
