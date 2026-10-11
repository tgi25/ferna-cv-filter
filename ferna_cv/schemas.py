"""Pydantic schemas.

Two families of models live here:

* LLM-facing models (``LenientModel`` subclasses) — what the model must return. They are
  deliberately forgiving: unknown keys are ignored, ``null`` means "use the default", single
  strings are accepted where lists are expected, and numbers wrapped in text ("5+ years") are
  coerced. Validation failures still trigger one automatic repair round-trip in ``llm.py``.
* Configuration / result models — employer criteria, weight plans and scored candidates.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
class LenientModel(BaseModel):
    """Base for LLM-facing schemas: ignores unknown keys and treats null as 'use default'."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


def as_str_list(value: Any) -> List[str]:
    """Accept a list, a single delimited string, or list of dicts and return a clean list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n;]|,(?![^()]*\))", value)
        return [p.strip(" -•\t") for p in parts if p.strip(" -•\t")]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):  # e.g. {"name": "Python", "level": "advanced"}
                item = item.get("name") or item.get("text") or item.get("title") or " ".join(
                    str(v) for v in item.values() if v
                )
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(value)]


def first_number(value: Any) -> Optional[float]:
    """'5+ years' -> 5.0, '3.5' -> 3.5, 'unknown' -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def tri_state_bool(value: Any) -> Optional[bool]:
    if value is None or isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "yes", "y", "1", "willing", "open"}:
        return True
    if s in {"false", "no", "n", "0", "not willing", "unwilling"}:
        return False
    return None


# --------------------------------------------------------------------------------------------
# Degree levels
# --------------------------------------------------------------------------------------------
class DegreeLevel(str, Enum):
    NONE = "none"
    CERTIFICATE = "certificate"
    DIPLOMA = "diploma"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"

    @property
    def rank(self) -> int:
        return _DEGREE_ORDER.index(self)

    @property
    def label(self) -> str:
        return _DEGREE_LABELS[self]


_DEGREE_ORDER = [
    DegreeLevel.NONE,
    DegreeLevel.CERTIFICATE,
    DegreeLevel.DIPLOMA,
    DegreeLevel.ASSOCIATE,
    DegreeLevel.BACHELOR,
    DegreeLevel.MASTER,
    DegreeLevel.DOCTORATE,
]
_DEGREE_LABELS = {
    DegreeLevel.NONE: "No requirement / none",
    DegreeLevel.CERTIFICATE: "Certificate",
    DegreeLevel.DIPLOMA: "Diploma / HND",
    DegreeLevel.ASSOCIATE: "Associate degree",
    DegreeLevel.BACHELOR: "Bachelor's degree",
    DegreeLevel.MASTER: "Master's degree",
    DegreeLevel.DOCTORATE: "Doctorate (PhD)",
}

# Checked top-down; first hit wins. Patterns run on lower-cased text.
_LEVEL_PATTERNS = [
    (DegreeLevel.DOCTORATE, r"\bph\.?\s?d\b|\bd\.?\s?phil\b|\bdoctorate\b|\bdoctor of\b|\bdoctoral\b"),
    (
        DegreeLevel.MASTER,
        r"\bm\.?\s?sc\b|\bmasters?\b\s*(?:of|in|degree)|\bmaster'?s\b|\bmba\b|\bm\.?\s?phil\b"
        r"|\bm\.?\s?eng\b|\bm\.?\s?tech\b|\bm\.\s?a\.|\bm\.\s?s\.|\bma\s*(?:\(|in\b)|\bmca\b|\bllm\b"
        r"|\bpostgraduate degree\b",
    ),
    (
        DegreeLevel.BACHELOR,
        r"\bb\.?\s?sc\b|\bbachelor|\bb\.?\s?eng\b|\bb\.?\s?tech\b|\bb\.\s?a\.|\bb\.\s?s\.|\bba\b"
        r"|\bbba\b|\bb\.?\s?com\b|\bbcs\b|\bllb\b|\bhonou?rs degree\b|\bundergraduate degree\b",
    ),
    (DegreeLevel.ASSOCIATE, r"\bassociate'?s?\s+degree\b"),
    (DegreeLevel.DIPLOMA, r"\bdiploma\b|\bhnd\b|\bhigher national\b|\bnvq\s*(?:level)?\s*[56]\b"),
    (DegreeLevel.CERTIFICATE, r"\bcertificate\b"),
]


def coerce_level(value: Any) -> DegreeLevel:
    """Map free text ("BSc (Hons)", "Master's", "phd") onto a DegreeLevel."""
    if isinstance(value, DegreeLevel):
        return value
    s = str(value or "").strip().lower()
    for lvl in DegreeLevel:
        if s == lvl.value:
            return lvl
    for lvl, pattern in _LEVEL_PATTERNS:
        if re.search(pattern, s):
            return lvl
    return DegreeLevel.NONE


# --------------------------------------------------------------------------------------------
# LLM-facing: candidate profile + qualitative assessment
# --------------------------------------------------------------------------------------------
class Education(LenientModel):
    degree: str = Field("", description="Qualification title as written, e.g. 'BSc (Hons) in Computer Science'")
    level: DegreeLevel = Field(DegreeLevel.NONE, description="Normalised level of the qualification")
    field: str = Field("", description="Field / major, e.g. 'Computer Science'")
    institution: str = ""
    end_year: Optional[int] = Field(None, description="Year completed (or expected)")
    grade: str = Field("", description="Class / GPA as written, e.g. 'First Class', 'GPA 3.72/4.0'")

    @field_validator("level", mode="before")
    @classmethod
    def _lvl(cls, v: Any) -> DegreeLevel:
        return coerce_level(v)

    @field_validator("end_year", mode="before")
    @classmethod
    def _year(cls, v: Any) -> Optional[int]:
        if v is None or isinstance(v, int):
            return v
        m = re.search(r"(19|20)\d{2}", str(v))
        return int(m.group()) if m else None

    @model_validator(mode="after")
    def _infer_level(self) -> "Education":
        if self.level == DegreeLevel.NONE and self.degree:
            self.level = coerce_level(self.degree)
        return self


class WorkExperience(LenientModel):
    title: str = ""
    organization: str = ""
    start: str = Field("", description="Start date as YYYY-MM or YYYY")
    end: str = Field("", description="End date as YYYY-MM, YYYY or 'present'")
    description: str = Field("", description="One-sentence summary of responsibilities/impact")


_LIST_FIELDS = (
    "certifications", "skills", "academic_achievements", "professional_achievements",
    "teamwork_leadership", "special_accomplishments", "sports", "societies",
    "volunteer_work", "languages",
)


class CandidateProfile(LenientModel):
    full_name: str = ""
    email: str = ""
    phone: str = ""
    date_of_birth: str = Field("", description="Only if explicitly stated; YYYY-MM-DD, YYYY-MM or YYYY")
    stated_age: Optional[int] = Field(None, description="Only if the CV explicitly states an age")
    current_location: str = Field("", description="City/country where the candidate currently lives")
    willing_to_relocate: Optional[bool] = Field(None, description="true/false only if explicitly stated")
    total_years_experience: Optional[float] = Field(
        None, description="Total professional experience in years (exclude internships unless stated as work)"
    )
    education: List[Education] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list, description="Professional certifications/licences")
    skills: List[str] = Field(default_factory=list, description="Technical and professional skills, one per item")
    academic_achievements: List[str] = Field(
        default_factory=list, description="Scholarships, awards, dean's list, publications, patents, medals"
    )
    professional_achievements: List[str] = Field(
        default_factory=list, description="Work awards, promotions, quantified impact"
    )
    teamwork_leadership: List[str] = Field(
        default_factory=list, description="Leadership roles, team leading, mentoring, collaboration evidence"
    )
    special_accomplishments: List[str] = Field(
        default_factory=list, description="Competitions won, hackathons, notable initiatives"
    )
    sports: List[str] = Field(default_factory=list)
    societies: List[str] = Field(default_factory=list, description="Clubs, societies, professional bodies")
    volunteer_work: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    summary: str = Field("", description="Two-sentence neutral summary of the candidate")

    @field_validator(*_LIST_FIELDS, mode="before")
    @classmethod
    def _listify(cls, v: Any) -> List[str]:
        return as_str_list(v)

    @field_validator("stated_age", mode="before")
    @classmethod
    def _age(cls, v: Any) -> Optional[int]:
        n = first_number(v)
        return int(n) if n is not None and 14 <= n <= 90 else None

    @field_validator("total_years_experience", mode="before")
    @classmethod
    def _years(cls, v: Any) -> Optional[float]:
        n = first_number(v)
        return n if n is not None and 0 <= n <= 60 else None

    @field_validator("willing_to_relocate", mode="before")
    @classmethod
    def _reloc(cls, v: Any) -> Optional[bool]:
        return tri_state_bool(v)

    @field_validator("education", "work_experience", mode="before")
    @classmethod
    def _objlist(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return []


class ModuleAssessment(LenientModel):
    score: float = Field(0, ge=0, le=10, description="0-10 using the rubric")
    justification: str = Field("", description="2-3 sentences explaining the score")
    evidence: List[str] = Field(default_factory=list, description="Short items from the CV that support the score")

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        n = first_number(v)
        return 0.0 if n is None else max(0.0, min(10.0, n))

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v: Any) -> List[str]:
        return as_str_list(v)


class QualitativeAssessment(LenientModel):
    academic_professional: ModuleAssessment = Field(default_factory=ModuleAssessment)
    highlights: ModuleAssessment = Field(default_factory=ModuleAssessment)
    extracurricular: ModuleAssessment = Field(default_factory=ModuleAssessment)


class CVAnalysis(LenientModel):
    """Single structured output requested from the LLM for each CV."""

    profile: CandidateProfile = Field(default_factory=CandidateProfile)
    assessment: QualitativeAssessment = Field(default_factory=QualitativeAssessment)


# --------------------------------------------------------------------------------------------
# Employer criteria (Stage 1)
# --------------------------------------------------------------------------------------------
class Enforcement(str, Enum):
    HARD = "hard"  # failure -> Disqualified
    SOFT = "soft"  # failure -> Flagged for human review
    OFF = "off"    # ignored


class AgeRule(BaseModel):
    enforcement: Enforcement = Enforcement.OFF
    min_age: Optional[int] = None
    max_age: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return self.enforcement != Enforcement.OFF and bool(self.min_age or self.max_age)


class LocationRule(BaseModel):
    enforcement: Enforcement = Enforcement.OFF
    accepted_locations: List[str] = Field(default_factory=list)
    accept_relocation: bool = True

    @property
    def is_active(self) -> bool:
        return self.enforcement != Enforcement.OFF and bool(self.accepted_locations)


class ExperienceRule(BaseModel):
    enforcement: Enforcement = Enforcement.HARD
    min_years: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.enforcement != Enforcement.OFF and self.min_years > 0


class QualificationRule(BaseModel):
    enforcement: Enforcement = Enforcement.HARD
    min_level: DegreeLevel = DegreeLevel.NONE
    fields: List[str] = Field(default_factory=list, description="Accepted fields of study (any)")
    related_fields_ok: bool = Field(True, description="Other fields -> flag for review instead of failing")

    @property
    def is_active(self) -> bool:
        return self.enforcement != Enforcement.OFF and (self.min_level != DegreeLevel.NONE or bool(self.fields))


class CertificationRule(BaseModel):
    enforcement: Enforcement = Enforcement.SOFT
    required: List[str] = Field(default_factory=list)
    min_required: int = Field(0, ge=0, description="0 = all listed certifications are required")

    @property
    def is_active(self) -> bool:
        return self.enforcement != Enforcement.OFF and bool(self.required)

    @property
    def needed(self) -> int:
        n = len(self.required)
        return n if self.min_required <= 0 else min(self.min_required, n)


class SkillsRule(BaseModel):
    enforcement: Enforcement = Enforcement.HARD
    required: List[str] = Field(default_factory=list)
    min_coverage: float = Field(0.7, ge=0, le=1, description="Fraction of core skills that must be present")

    @property
    def is_active(self) -> bool:
        return self.enforcement != Enforcement.OFF and bool(self.required)


class MandatoryCriteria(BaseModel):
    age: AgeRule = Field(default_factory=AgeRule)
    location: LocationRule = Field(default_factory=LocationRule)
    experience: ExperienceRule = Field(default_factory=ExperienceRule)
    qualification: QualificationRule = Field(default_factory=QualificationRule)
    certification: CertificationRule = Field(default_factory=CertificationRule)
    skills: SkillsRule = Field(default_factory=SkillsRule)
    nice_to_have_skills: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Weights (Stage 2)
# --------------------------------------------------------------------------------------------
MODULE_KEYS = ("mandatory_compliance", "academic_professional", "highlights", "extracurricular")
MODULE_LABELS = {
    "mandatory_compliance": "Mandatory compliance",
    "academic_professional": "Academic & professional achievements",
    "highlights": "Other highlights (teamwork, leadership, accomplishments)",
    "extracurricular": "Extra-curricular activities",
}
MODULE_SHORT = {
    "mandatory_compliance": "Compliance",
    "academic_professional": "Achievements",
    "highlights": "Highlights",
    "extracurricular": "Extra-curricular",
}
COMPLIANCE_KEYS = ("skills", "experience", "qualification", "certification", "location")
COMPLIANCE_LABELS = {
    "skills": "Core skills",
    "experience": "Experience",
    "qualification": "Qualifications",
    "certification": "Certifications",
    "location": "Location",
}


class ModuleWeights(BaseModel):
    mandatory_compliance: int = Field(50, ge=0, le=100)
    academic_professional: int = Field(20, ge=0, le=100)
    highlights: int = Field(15, ge=0, le=100)
    extracurricular: int = Field(15, ge=0, le=100)

    @model_validator(mode="after")
    def _total(self) -> "ModuleWeights":
        total = sum(self.as_dict().values())
        if total != 100:
            raise ValueError(f"Module weights must total 100 (got {total}).")
        return self

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in MODULE_KEYS}


class ComplianceSubWeights(BaseModel):
    """Relative importance of each mandatory criterion inside the compliance module."""

    skills: float = Field(35, ge=0)
    experience: float = Field(25, ge=0)
    qualification: float = Field(20, ge=0)
    certification: float = Field(10, ge=0)
    location: float = Field(10, ge=0)

    def as_dict(self) -> dict:
        return {k: float(getattr(self, k)) for k in COMPLIANCE_KEYS}


class WeightPlan(BaseModel):
    mode: str = "autonomous"  # "autonomous" | "custom"
    modules: ModuleWeights = Field(default_factory=ModuleWeights)
    compliance: ComplianceSubWeights = Field(default_factory=ComplianceSubWeights)
    rationale: str = ""


# --------------------------------------------------------------------------------------------
# LLM-facing: job description analysis (auto-fill + autonomous weights)
# --------------------------------------------------------------------------------------------
class SuggestedWeights(LenientModel):
    mandatory_compliance: float = 50
    academic_professional: float = 20
    highlights: float = 15
    extracurricular: float = 15

    @field_validator("*", mode="before")
    @classmethod
    def _num(cls, v: Any) -> float:
        n = first_number(v)
        return 0.0 if n is None else max(0.0, n)


class SuggestedSubWeights(LenientModel):
    skills: float = 35
    experience: float = 25
    qualification: float = 20
    certification: float = 10
    location: float = 10

    @field_validator("*", mode="before")
    @classmethod
    def _num(cls, v: Any) -> float:
        n = first_number(v)
        return 0.0 if n is None else max(0.0, n)


class JDAnalysis(LenientModel):
    job_title: str = ""
    seniority: str = Field("mid", description="entry | mid | senior | lead | executive")
    complexity: int = Field(3, description="1 (simple) to 5 (highly specialised)")
    min_years_experience: Optional[float] = None
    min_degree_level: DegreeLevel = DegreeLevel.NONE
    degree_fields: List[str] = Field(default_factory=list)
    related_fields_ok: bool = Field(False, description="true if the JD accepts 'a related field' / equivalent")
    required_certifications: List[str] = Field(default_factory=list)
    core_skills: List[str] = Field(default_factory=list, description="Must-have skills only")
    nice_to_have_skills: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    relocation_acceptable: Optional[bool] = None
    min_age: Optional[int] = Field(None, description="Only if the JD explicitly states an age limit")
    max_age: Optional[int] = Field(None, description="Only if the JD explicitly states an age limit")
    weights: SuggestedWeights = Field(default_factory=SuggestedWeights)
    compliance_weights: SuggestedSubWeights = Field(default_factory=SuggestedSubWeights)
    rationale: str = Field("", description="2-4 sentences explaining the weight allocation")

    @field_validator("min_degree_level", mode="before")
    @classmethod
    def _lvl(cls, v: Any) -> DegreeLevel:
        return coerce_level(v)

    @field_validator("degree_fields", "required_certifications", "core_skills", "nice_to_have_skills", "locations", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> List[str]:
        return as_str_list(v)

    @field_validator("complexity", mode="before")
    @classmethod
    def _cx(cls, v: Any) -> int:
        n = first_number(v)
        return 3 if n is None else int(max(1, min(5, round(n))))

    @field_validator("min_years_experience", mode="before")
    @classmethod
    def _yrs(cls, v: Any) -> Optional[float]:
        return first_number(v)

    @field_validator("min_age", "max_age", mode="before")
    @classmethod
    def _ages(cls, v: Any) -> Optional[int]:
        n = first_number(v)
        return int(n) if n is not None and 14 <= n <= 90 else None

    @field_validator("related_fields_ok", mode="before")
    @classmethod
    def _rf(cls, v: Any) -> bool:
        return bool(tri_state_bool(v))

    @field_validator("relocation_acceptable", mode="before")
    @classmethod
    def _rel(cls, v: Any) -> Optional[bool]:
        return tri_state_bool(v)


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------
class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class CandidateStatus(str, Enum):
    QUALIFIED = "Qualified"
    FLAGGED = "Flagged"
    DISQUALIFIED = "Disqualified"
    ERROR = "Error"


class GateCheck(BaseModel):
    criterion: str
    enforcement: Enforcement
    status: CheckStatus
    detail: str


class ModuleScore(BaseModel):
    key: str
    label: str
    weight: int
    fraction: float  # 0..1
    points: float
    justification: str = ""
    evidence: List[str] = Field(default_factory=list)


class ComplianceDetail(BaseModel):
    key: str
    label: str
    share: float  # share of the compliance module (0..1)
    fraction: float  # 0..1 achieved
    note: str = ""


class CVRecord(BaseModel):
    """Expensive per-CV output (extraction + analysis). Cached; re-scored cheaply."""

    file_name: str
    file_hash: str
    cache_key: str = ""
    jd_hash: str = ""
    source: str = ""
    text: str = ""
    analysis: Optional[CVAnalysis] = None
    warnings: List[str] = Field(default_factory=list)
    error: str = ""
    seconds: float = 0.0


class CandidateResult(BaseModel):
    rank: int = 0
    file_name: str
    file_hash: str = ""
    candidate_name: str
    status: CandidateStatus
    total: float = 0.0
    modules: List[ModuleScore] = Field(default_factory=list)
    compliance: List[ComplianceDetail] = Field(default_factory=list)
    gate: List[GateCheck] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    years_experience: Optional[float] = None
    years_source: str = ""
    age: Optional[int] = None
    location: str = ""
    highest_qualification: str = ""
    skills_matched: List[str] = Field(default_factory=list)
    skills_missing: List[str] = Field(default_factory=list)
    certs_matched: List[str] = Field(default_factory=list)
    certs_missing: List[str] = Field(default_factory=list)
    nice_matched: List[str] = Field(default_factory=list)
    jd_similarity: Optional[float] = None
    source: str = ""
    warnings: List[str] = Field(default_factory=list)
    error: str = ""
    stale: bool = False
    profile: Optional[CandidateProfile] = None
    cv_text: str = Field("", exclude=True)
