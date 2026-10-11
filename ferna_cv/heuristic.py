"""Offline, rule-based analyser for CVs and job descriptions.

Used when no API key is configured (demo / air-gapped mode) and as a safety net. It is
deliberately conservative: it extracts what it can recognise from section headings, date
ranges, degree keywords and a skills vocabulary. LLM mode is markedly more accurate on
unusual layouts and for judging the *relevance* of achievements.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dates import find_date, format_partial, parse_date
from .extraction import EMAIL_RE, find_phones
from .matching import SKILL_VOCAB, norm_skill, text_mentions
from .schemas import (
    CandidateProfile,
    CVAnalysis,
    DegreeLevel,
    Education,
    JDAnalysis,
    ModuleAssessment,
    QualitativeAssessment,
    SuggestedSubWeights,
    SuggestedWeights,
    WorkExperience,
    coerce_level,
)
from .weights import heuristic_module_weights

# --------------------------------------------------------------------------------------------
# Section splitting
# --------------------------------------------------------------------------------------------
SECTION_ALIASES: Dict[str, List[str]] = {
    "summary": ["summary", "profile", "professional summary", "professional profile", "about me", "about",
                "objective", "career objective", "career summary", "personal statement"],
    "experience": ["experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "career history", "professional background",
                   "relevant experience", "industry experience"],
    "education": ["education", "academic qualifications", "educational qualifications", "academic background",
                  "qualifications", "education and qualifications", "education and training", "academic profile"],
    "skills": ["skills", "technical skills", "core skills", "key skills", "core competencies", "competencies",
               "technical competencies", "tools and technologies", "technologies", "skills and expertise",
               "it skills", "computer skills"],
    "certifications": ["certifications", "certificates", "professional certifications", "licenses and certifications",
                       "licences and certifications", "professional qualifications", "certifications and training",
                       "courses and certifications", "training"],
    "achievements": ["achievements", "awards", "honours", "honors", "awards and achievements", "awards and honours",
                     "awards and honors", "accomplishments", "publications", "research", "research publications",
                     "scholarships", "achievements and awards"],
    "leadership": ["leadership", "leadership and teamwork", "positions of responsibility", "leadership experience",
                   "leadership roles"],
    "extracurricular": ["extra curricular activities", "extracurricular activities", "extra curricular",
                        "extracurricular", "activities", "sports", "sports and activities", "interests", "hobbies",
                        "hobbies and interests", "volunteering", "volunteer work", "volunteer experience",
                        "community service", "societies", "memberships", "professional memberships",
                        "clubs and societies"],
    "personal": ["personal details", "personal information", "personal data", "contact", "contact details",
                 "contact information", "personal profile"],
    "projects": ["projects", "key projects", "academic projects", "selected projects"],
    "languages": ["languages", "language skills"],
    "references": ["references", "referees", "non related referees"],
}


def _key(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_HEADING = {_key(a): sec for sec, aliases in SECTION_ALIASES.items() for a in aliases}
_BULLETS = " \t•·-*–—▪●◦►✓"


def split_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for raw in text.split("\n"):
        line = raw.strip(_BULLETS)
        if not line:
            continue
        k = _key(line) if len(line) <= 50 else ""
        if k in _HEADING:
            current = _HEADING[k]
            sections.setdefault(current, [])
            continue
        head, sep, rest = line.partition(":")
        if sep and len(head) <= 40 and _key(head) in _HEADING:
            current = _HEADING[_key(head)]
            sections.setdefault(current, [])
            if rest.strip():
                sections[current].append(rest.strip())
            continue
        sections.setdefault(current, []).append(line)
    return sections


# --------------------------------------------------------------------------------------------
# Regexes
# --------------------------------------------------------------------------------------------
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MON}\s*,?\s*\d{{4}}|\d{{1,2}}[/.-]\d{{4}}|\d{{4}}[/.-]\d{{1,2}}(?!\d)|\d{{4}})"
_END = rf"(?:{_DATE}|present|current(?:ly)?|now|to\s*date|till\s*date|ongoing)"
RANGE_RE = re.compile(rf"({_DATE})\s*(?:-|–|—|~|to|until|till)\s*({_END})", re.I)

DOB_RE = re.compile(
    r"(?:date\s+of\s+birth|\bd\.?\s?o\.?\s?b\b\.?|birth\s*date|\bborn(?:\s+on)?)\s*[:\-–]?\s*([^\n|;]{4,40})", re.I
)
AGE_RE = re.compile(r"\bage\s*[:\-–]\s*(\d{2})\b", re.I)
LOC_RE = re.compile(
    r"^(?:current\s+)?(?:location|address|city|based\s+in|residence|residing\s+(?:in|at)|lives?\s+in|home\s*town)"
    r"\s*[:\-–]?\s*(.{2,120})$",
    re.I | re.M,
)
YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\.?\s+(?:of\s+)?(?:[a-z/&-]+\s+){0,4}?experience", re.I
)
_COUNTRIES = ("sri lanka", "india", "united kingdom", "uk", "usa", "united states", "australia", "canada",
              "singapore", "uae", "united arab emirates", "maldives", "qatar", "new zealand", "germany",
              "malaysia", "japan", "ireland", "bangladesh", "pakistan", "nepal")

ACAD_RE = re.compile(
    r"dean'?s list|scholarship|gold medal|silver medal|bronze medal|valedictorian|best (?:student|graduate|thesis|"
    r"project|undergraduate)|journal|conference|proceedings|published (?:a |in |paper)|publication|patent|olympiad|"
    r"academic award|research grant|merit award|vice[- ]chancellor'?s award|top of (?:the )?(?:class|batch)",
    re.I,
)
PROF_RE = re.compile(
    r"\b(award(?:ed)?|employee of the (?:month|year|quarter)|top performer|recogni[sz]ed for|promoted to)\b"
    r"|\b(increased|reduced|improved|saved|grew|cut|boosted)\b[^.]*\d+\s*%",
    re.I,
)
LEAD_RE = re.compile(
    r"\b(led|lead|leading|captain(?:ed)?|vice[- ]captain|president|vice[- ]president|secretary|treasurer|"
    r"head of|managed (?:a |the )?team|team of \d+|supervis(?:ed|ing)|mentor(?:ed|ing)?|coordinat(?:ed|or)|"
    r"organi[sz](?:ed|er)|chair(?:ed|person|man)?|founder|co-founder|team lead|scrum master|prefect|"
    r"cross-functional|collaborat\w*)\b",
    re.I,
)
ACCOMP_RE = re.compile(
    r"\b(winner|won|champion(?:s|ship)?|first place|1st place|second place|2nd place|runner[- ]up|finalist|"
    r"hackathon|best paper|top \d+|national level|international level|gold award|record)\b",
    re.I,
)
SPORT_RE = re.compile(
    r"\b(cricket|rugby|football|soccer|athletics|basketball|volleyball|badminton|tennis|table tennis|swimming|"
    r"chess|hockey|netball|karate|taekwondo|judo|marathon|elle|baseball|cycling|hiking|rowing|boxing|"
    r"wrestling|squash|golf|sports?|carrom)\b",
    re.I,
)
SOCIETY_RE = re.compile(
    r"\b(society|club|association|ieee|acm|rotaract|leo club|toastmasters|aiesec|students'? union|committee|"
    r"member of|membership|chapter|scouts?|cadet)\b",
    re.I,
)
VOLUNTEER_RE = re.compile(
    r"\b(volunteer\w*|charity|ngo|community service|blood donation|fundrais\w*|outreach|relief|shramadana|"
    r"beach clean\w*|donation drive|mentoring students)\b",
    re.I,
)
_NOT_NAME = re.compile(r"curriculum|vitae|resume|résumé|\bcv\b|profile|page \d|confidential", re.I)


# --------------------------------------------------------------------------------------------
# Field extractors
# --------------------------------------------------------------------------------------------
def guess_name(header: List[str], text: str, file_name: str = "") -> str:
    candidates = header[:6] or text.split("\n")[:6]
    for line in candidates:
        line = line.strip()
        if not line or _NOT_NAME.search(line) or "@" in line or re.search(r"\d", line):
            continue
        words = line.replace(",", " ").split()
        if (
            2 <= len(words) <= 5
            and len(line) <= 60
            and all(w[0].isalpha() for w in words)
            and sum(w[0].isupper() for w in words) >= len(words) - 1
        ):
            return line.title() if line.isupper() else line
    stem = Path(file_name).stem
    stem = re.sub(r"(?i)\b(cv|resume|curriculum vitae)\b", " ", stem)
    stem = re.sub(r"[_\-.]+", " ", stem).strip()
    return stem.title() or "Unnamed candidate"


def _dob_age(text: str) -> Tuple[str, Optional[int]]:
    dob = ""
    m = DOB_RE.search(text)
    if m:
        d, prec = find_date(m.group(1))
        if d and prec != "ongoing":
            dob = format_partial(d, prec)
    age = None
    m = AGE_RE.search(text)
    if m:
        age = int(m.group(1))
    return dob, age


def _location(text: str, header: List[str]) -> str:
    m = LOC_RE.search(text)
    if m:
        return m.group(1).strip(" .|")
    for line in header[:6]:
        for part in re.split(r"[|•·]", line):
            p = part.strip()
            if "," in p and "@" not in p and not re.search(r"\d{3,}", p) and len(p) <= 50:
                if any(p.lower().rstrip(". ").endswith(c) for c in _COUNTRIES):
                    return p
    return ""


def _relocation(text: str) -> Optional[bool]:
    if re.search(r"\b(not|un)\s*(willing|open|able|prepared)\s+to\s+relocat|relocation\s*:\s*no\b", text, re.I):
        return False
    if re.search(
        r"\b(willing|open|happy|ready|able|prepared)\s+(?:to\s+)?relocat|open to relocation|relocation\s*:\s*yes",
        text,
        re.I,
    ):
        return True
    return None


def _norm_date(s: str) -> str:
    d, prec = parse_date(s)
    return format_partial(d, prec)


def _work(sections: Dict[str, List[str]]) -> List[WorkExperience]:
    lines = sections.get("experience")
    if not lines:
        skip = {"education", "achievements", "extracurricular", "references", "certifications", "personal",
                "projects", "leadership", "languages"}
        lines = [ln for sec, ls in sections.items() if sec not in skip for ln in ls]
    jobs: List[WorkExperience] = []
    prev = ""
    for line in lines:
        m = RANGE_RE.search(line)
        if not m:
            prev = line
            continue
        title = (line[: m.start()] + " " + line[m.end():]).strip(" |,-–—()[]:\t")
        title = re.sub(r"\s{2,}", " ", title)
        if len(title) < 3:
            title = prev
        org = ""
        for sep in (" at ", " @ ", " | ", ", ", " - ", " – ", " — "):
            if sep in title:
                title, org = [x.strip(" |,-–—") for x in title.split(sep, 1)]
                break
        jobs.append(
            WorkExperience(title=title[:120], organization=org[:120],
                           start=_norm_date(m.group(1)), end=_norm_date(m.group(2)))
        )
    return jobs


def _stated_years(text: str) -> Optional[float]:
    vals = [float(x) for x in YEARS_RE.findall(text)]
    vals = [v for v in vals if 0 < v <= 50]
    return max(vals) if vals else None


def _degree_title(line: str) -> str:
    # keep the first comma/pipe segment that names the qualification ("MSc in X, Univ, 2020")
    for seg in re.split(r",|\|| – | — ", line):
        if coerce_level(seg) != DegreeLevel.NONE:
            line = seg
            break
    t = RANGE_RE.sub("", line)
    t = re.sub(r"\b(19|20)\d{2}\b", "", t)
    t = t.split(" | ")[0]
    return re.sub(r"\s{2,}", " ", t).strip(" ,|-–()")[:120]


def _field(line: str) -> str:
    l = re.sub(r"\(.*?\)", " ", line)
    low = l.lower()
    idx = low.rfind(" in ")
    kw = 4
    if idx == -1:
        idx = low.find(" of ")
    if idx == -1:
        return ""
    rest = l[idx + kw:]
    for cut in (",", "|", " - ", " – ", " — ", " from ", " at ", ";", "  "):
        pos = rest.find(cut)
        if pos != -1:
            rest = rest[:pos]
    rest = re.sub(r"\b(19|20)\d{2}\b.*$", "", rest)
    return rest.strip(" .:-")[:80]


def _institution(lines: List[str]) -> str:
    for line in lines:
        for part in re.split(r",|\||–|—| - ", line):
            if re.search(r"universit|institute|college|school|academy|polytechnic|campus", part, re.I):
                return part.strip()[:120]
    return ""


_GRADES = [
    (r"first\s+class", "First Class"),
    (r"second\s+class\s*\(?\s*upper|upper\s+second", "Second Class (Upper)"),
    (r"second\s+class\s*\(?\s*lower|lower\s+second", "Second Class (Lower)"),
    (r"summa\s+cum\s+laude", "Summa cum laude"),
    (r"magna\s+cum\s+laude", "Magna cum laude"),
    (r"\bcum\s+laude", "Cum laude"),
    (r"\bdistinction\b", "Distinction"),
    (r"\bmerit\b", "Merit"),
]


def _grade(ctx: str) -> str:
    for pat, label in _GRADES:
        if re.search(pat, ctx, re.I):
            return label
    m = re.search(r"\b(?:c?gpa|grade point average)\s*[:\-]?\s*(\d\.\d{1,2})(?:\s*/\s*(\d(?:\.\d+)?))?", ctx, re.I)
    if m:
        return f"GPA {m.group(1)}" + (f"/{m.group(2)}" if m.group(2) else "")
    return ""


def _education(sections: Dict[str, List[str]], text: str) -> List[Education]:
    lines = sections.get("education") or text.split("\n")
    out: List[Education] = []
    seen = set()
    for i, line in enumerate(lines):
        if len(line) > 200:
            continue
        lvl = coerce_level(line)
        if lvl == DegreeLevel.NONE:
            continue
        near = " ".join(lines[i:i + 2])
        years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", near)]
        edu = Education(
            degree=_degree_title(line), level=lvl, field=_field(line),
            institution=_institution(lines[i:i + 3]), end_year=max(years) if years else None,
            grade=_grade(line) or next(
                (_grade(nxt) for nxt in lines[i + 1:i + 2] if coerce_level(nxt) == DegreeLevel.NONE), ""),
        )
        key = (edu.level, edu.field.lower(), edu.degree.lower()[:30])
        if key not in seen:
            seen.add(key)
            out.append(edu)
    return out


def _certifications(sections: Dict[str, List[str]]) -> List[str]:
    out = []
    for line in sections.get("certifications", []):
        for part in re.split(r";", line):
            p = part.strip(_BULLETS)
            if 3 <= len(p) <= 150:
                out.append(p)
    return out


def _dedupe(items: List[str], limit: int = 40) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = norm_skill(it)
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out[:limit]


def _skills(sections: Dict[str, List[str]], text: str) -> List[str]:
    items: List[str] = []
    for line in sections.get("skills", []):
        if ":" in line:
            line = line.split(":", 1)[1]
        items += [x.strip(" .") for x in re.split(r"[,;|•·]", line) if 1 <= len(x.strip(" .")) <= 40]
    tl = text.lower()
    for term in SKILL_VOCAB:
        if len(term) > 2 and text_mentions(term, text, tl):
            items.append(term)
    return _dedupe(items, 60)


def _activities(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in (
        "academic", "professional", "lead", "accomp", "sports", "societies", "volunteer")}

    def pool(names):
        for sec in names:
            for line in sections.get(sec, []):
                if 6 <= len(line) <= 220 and coerce_level(line) == DegreeLevel.NONE:
                    yield sec, line

    for sec, line in pool(["achievements", "education", "extracurricular", "leadership", "summary", "experience"]):
        if ACAD_RE.search(line) and sec in ("achievements", "education", "extracurricular", "leadership"):
            out["academic"].append(line)
        elif PROF_RE.search(line) or (sec == "achievements" and not (
                SPORT_RE.search(line) or SOCIETY_RE.search(line) or VOLUNTEER_RE.search(line))):
            out["professional"].append(line)
    for _, line in pool(["experience", "leadership", "extracurricular", "achievements", "summary", "projects"]):
        if LEAD_RE.search(line):
            out["lead"].append(line)
    for _, line in pool(["achievements", "extracurricular", "projects", "leadership", "education", "experience"]):
        if ACCOMP_RE.search(line) and line not in out["academic"]:
            out["accomp"].append(line)
    for _, line in pool(["extracurricular", "leadership", "achievements", "education"]):
        if line in out["academic"]:
            continue
        if VOLUNTEER_RE.search(line):
            out["volunteer"].append(line)
        elif SPORT_RE.search(line):
            out["sports"].append(line)
        elif SOCIETY_RE.search(line):
            out["societies"].append(line)
    return {k: list(dict.fromkeys(v))[:8] for k, v in out.items()}


def _assess(p: CandidateProfile) -> QualitativeAssessment:
    note = "Rule-based estimate (offline mode) — counts recognised evidence; does not judge relevance to the JD."
    # Academic & professional
    rank_pts = {DegreeLevel.DOCTORATE: 4, DegreeLevel.MASTER: 3, DegreeLevel.BACHELOR: 2,
                DegreeLevel.ASSOCIATE: 1.5, DegreeLevel.DIPLOMA: 1, DegreeLevel.CERTIFICATE: 0.5}
    best = max(p.education, key=lambda e: e.level.rank, default=None)
    deg = rank_pts.get(best.level, 0) if best else 0
    grade_bonus = 0.0
    for e in p.education:
        g = e.grade.lower()
        gpa = re.search(r"(\d\.\d+)(?:/(\d(?:\.\d+)?))?", g)
        ratio = (float(gpa.group(1)) / float(gpa.group(2) or 4.0)) if gpa else 0
        if any(x in g for x in ("first class", "distinction", "summa")) or ratio >= 0.925:
            grade_bonus = max(grade_bonus, 2.5)
        elif any(x in g for x in ("upper", "merit", "magna")) or ratio >= 0.825:
            grade_bonus = max(grade_bonus, 1.5)
        elif any(x in g for x in ("lower", "cum laude")) or ratio >= 0.75:
            grade_bonus = max(grade_bonus, 0.75)
    n_ach = len(p.academic_achievements) + len(p.professional_achievements)
    acad = min(10.0, deg + grade_bonus + 1.5 * n_ach)
    acad_ev = [e.grade + " — " + e.degree for e in p.education if e.grade][:2] + \
        (p.academic_achievements + p.professional_achievements)[:4]

    hi_items = p.teamwork_leadership + p.special_accomplishments
    high = min(10.0, 1.75 * len(p.teamwork_leadership) + 2.0 * len(p.special_accomplishments))

    cats = [c for c in (p.sports, p.societies, p.volunteer_work) if c]
    n_ex = len(p.sports) + len(p.societies) + len(p.volunteer_work)
    extra = min(10.0, 2.0 * n_ex + max(0, len(cats) - 1))

    return QualitativeAssessment(
        academic_professional=ModuleAssessment(
            score=round(acad, 1),
            justification=f"Highest qualification: {best.level.label if best else 'not found'}; "
                          f"grade bonus {grade_bonus:g}; {n_ach} achievement item(s) recognised. {note}",
            evidence=acad_ev,
        ),
        highlights=ModuleAssessment(
            score=round(high, 1),
            justification=f"{len(p.teamwork_leadership)} leadership/teamwork and "
                          f"{len(p.special_accomplishments)} accomplishment item(s) recognised. {note}",
            evidence=hi_items[:6],
        ),
        extracurricular=ModuleAssessment(
            score=round(extra, 1),
            justification=f"{len(p.sports)} sports, {len(p.societies)} society and {len(p.volunteer_work)} "
                          f"volunteering item(s) recognised. {note}",
            evidence=(p.sports + p.societies + p.volunteer_work)[:6],
        ),
    )


def analyze_cv(text: str, jd_text: str = "", file_name: str = "") -> CVAnalysis:
    """Rule-based CV analysis returning the same schema as the LLM path."""
    sections = split_sections(text)
    header = sections.get("header", [])
    p = CandidateProfile()
    p.full_name = guess_name(header, text, file_name)
    emails = EMAIL_RE.findall(text)
    p.email = emails[0] if emails else ""
    phones = find_phones(text)
    p.phone = phones[0] if phones else ""
    p.date_of_birth, p.stated_age = _dob_age(text)
    p.current_location = _location(text, header + sections.get("personal", []))
    p.willing_to_relocate = _relocation(text)
    p.work_experience = _work(sections)
    p.total_years_experience = _stated_years(text)
    p.education = _education(sections, text)
    p.certifications = _certifications(sections)
    p.skills = _skills(sections, text)
    acts = _activities(sections)
    p.academic_achievements = acts["academic"]
    p.professional_achievements = acts["professional"]
    p.teamwork_leadership = acts["lead"]
    p.special_accomplishments = acts["accomp"]
    p.sports, p.societies, p.volunteer_work = acts["sports"], acts["societies"], acts["volunteer"]
    p.languages = [x.strip() for ln in sections.get("languages", []) for x in re.split(r"[,;|]", ln) if x.strip()][:8]
    p.summary = " ".join(sections.get("summary", []))[:600]
    return CVAnalysis(profile=p, assessment=_assess(p))


# --------------------------------------------------------------------------------------------
# Job description
# --------------------------------------------------------------------------------------------
_NICE_RE = re.compile(
    r"\b(prefer\w*|desirable|advantage\w*|nice to have|nice-to-have|bonus|a plus|is a plus|would be|asset|ideally)\b",
    re.I,
)
_SENIOR_RE = re.compile(r"\b(senior|sr\.?|lead|principal|staff|architect|manager|head|director|chief|vp)\b", re.I)
_JUNIOR_RE = re.compile(r"\b(intern|internship|trainee|graduate|junior|jr\.?|entry[- ]level|fresher|apprentice)\b", re.I)
_CERT_NAME_RE = re.compile(
    r"(AWS Certified [A-Z][\w\-]*(?: [A-Z][\w\-]*){0,4}|Microsoft Certified:?(?: [A-Z][\w\-]*){1,6}|"
    r"Azure [A-Z][\w\-]*(?: [A-Z][\w\-]*){0,4} (?:Associate|Expert|Fundamentals)|Google (?:Cloud )?Professional"
    r"(?: [A-Z][\w\-]*){1,4}|TensorFlow Developer Certificate|PMP|PRINCE2|CISSP|CISA|CISM|CCNA|CCNP|CEH|"
    r"CompTIA [A-Z][A-Za-z+]*|ITIL(?: v?4)?|Six Sigma(?: [A-Z][a-z]+)? Belt|CIMA|ACCA|CFA|CPA|SHRM-CP|"
    r"Certified ScrumMaster|CSM|Oracle Certified(?: [A-Z][\w\-]*){1,4})"
)


def analyze_jd(jd: str) -> JDAnalysis:
    sentences = [s.strip(_BULLETS) for s in re.split(r"(?<=[.;!?])\s+|\n", jd or "") if s.strip(_BULLETS)]
    must = [s for s in sentences if not _NICE_RE.search(s)]
    nice = [s for s in sentences if _NICE_RE.search(s)]
    a = JDAnalysis()

    m = re.search(r"(?:job\s+title|position|role|vacancy)\s*[:\-–]\s*(.+)", jd or "", re.I)
    a.job_title = (m.group(1) if m else next((s for s in sentences if len(s) <= 80), "")).strip()[:100]

    for s in must:
        if re.search(r"experience", s, re.I):
            m = re.search(r"(\d{1,2}(?:\.\d)?)\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d{1,2}\s*)?(?:years?|yrs?)", s, re.I)
            if m:
                a.min_years_experience = float(m.group(1))
                break

    for s in must:
        lvl = coerce_level(s)
        if lvl not in (DegreeLevel.NONE, DegreeLevel.CERTIFICATE):
            a.min_degree_level = lvl
            a.related_fields_ok = bool(re.search(r"related|equivalent|similar", s, re.I))
            m = re.search(r"\b(?:in|of)\s+([A-Za-z][A-Za-z ,/&]+?)(?:\.|;|$|\bor\s+(?:a\s+)?related|\bor\s+equivalent)", s)
            if m:
                parts = re.split(r",|/|\bor\b|\band\b", m.group(1))
                a.degree_fields = [p.strip() for p in parts
                                   if 2 < len(p.strip()) <= 40 and not re.search(r"related|equivalent|field", p, re.I)]
            break

    certs = []
    for s in must:
        if re.search(r"certif|licen", s, re.I):
            certs += _CERT_NAME_RE.findall(s)
    a.required_certifications = list(dict.fromkeys(c.strip() for c in certs))

    def vocab_in(lines):
        found = []
        for s in lines:
            sl = s.lower()
            for term in SKILL_VOCAB:
                if len(term) > 2 and text_mentions(term, s, sl):
                    found.append(term)
        return list(dict.fromkeys(found))

    core = vocab_in(must)
    a.core_skills = core
    a.nice_to_have_skills = [t for t in vocab_in(nice) if t not in core]

    m = re.search(r"(?:location|based in|work location|job location|duty station)\s*[:\-–]\s*([^\n.]+)", jd or "", re.I)
    if m:
        locs = re.split(r"/|;|\||\bor\b", m.group(1))
        a.locations = [l.strip() for l in locs if l.strip() and not re.search(r"remote|hybrid|on-?site", l, re.I)]
    if re.search(r"relocat", jd or "", re.I):
        a.relocation_acceptable = True

    m = re.search(r"\bage(?:d)?\s*(?:limit)?\s*[:\-–]?\s*(?:between\s*)?(\d{2})\s*(?:-|–|to|and)\s*(\d{2})", jd or "", re.I)
    if m:
        a.min_age, a.max_age = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(?:below|under|not (?:be )?(?:more|older) than|maximum age(?: of)?|age limit\s*:?)\s*(\d{2})\s*(?:years)?",
                      jd or "", re.I)
        if m:
            a.max_age = int(m.group(1))

    cx = 3
    if _SENIOR_RE.search(a.job_title or jd[:300] if jd else ""):
        cx += 1
    if a.min_years_experience is not None and a.min_years_experience >= 5:
        cx += 1
    if len(a.core_skills) >= 8:
        cx += 1
    if _JUNIOR_RE.search(a.job_title or (jd or "")[:300]):
        cx -= 1
    if a.min_years_experience is not None and a.min_years_experience <= 1:
        cx -= 1
    a.complexity = max(1, min(5, cx))
    a.seniority = {1: "entry", 2: "entry", 3: "mid", 4: "senior", 5: "lead"}[a.complexity]
    a.weights = SuggestedWeights(**heuristic_module_weights(a.complexity))
    if a.complexity >= 4:
        a.compliance_weights = SuggestedSubWeights(skills=30, experience=30, qualification=20, certification=10, location=10)
    elif a.complexity <= 2:
        a.compliance_weights = SuggestedSubWeights(skills=35, experience=15, qualification=30, certification=10, location=10)
    a.rationale = (
        f"Estimated complexity {a.complexity}/5 ({a.seniority}) from seniority keywords, required experience "
        f"and {len(a.core_skills)} core skill(s). Higher complexity shifts points towards mandatory compliance; "
        f"entry-level roles give more weight to academic and extra-curricular evidence."
    )
    return a
