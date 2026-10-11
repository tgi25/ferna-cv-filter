"""Two-stage scoring.

Stage 1 — Mandatory gatekeeper. Each active criterion yields PASS / FAIL / UNKNOWN.
  * HARD criterion FAIL          -> Disqualified
  * SOFT criterion FAIL          -> Flagged
  * any criterion UNKNOWN        -> Flagged (missing information is never auto-disqualified)

Stage 2 — Weighted score out of 100 (computed for every candidate, so reviewers can see how
close a disqualified candidate came):
  compliance   = W_c * sum_k share_k * fraction_k   (skills coverage, years/min, degree fit ...)
  achievements = W_a * score_a / 10                   (LLM or rule-based 0-10 assessment)
  highlights   = W_h * score_h / 10
  extra        = W_e * score_e / 10
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dates import candidate_age, total_experience_years
from .matching import SemanticEngine, cert_match, match_many, phrase_match
from .schemas import (
    COMPLIANCE_KEYS,
    COMPLIANCE_LABELS,
    MODULE_LABELS,
    CandidateProfile,
    CandidateResult,
    CandidateStatus,
    CheckStatus,
    ComplianceDetail,
    CVRecord,
    DegreeLevel,
    Education,
    Enforcement,
    GateCheck,
    LocationRule,
    MandatoryCriteria,
    ModuleScore,
    QualificationRule,
    WeightPlan,
)

STATUS_ORDER = {
    CandidateStatus.QUALIFIED: 0,
    CandidateStatus.FLAGGED: 1,
    CandidateStatus.DISQUALIFIED: 2,
    CandidateStatus.ERROR: 3,
}


# --------------------------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------------------------
def experience_years(p: CandidateProfile, today: Optional[date] = None) -> Tuple[Optional[float], str]:
    """Prefer years computed from dated roles (overlaps merged); fall back to the stated total."""
    computed = total_experience_years(p.work_experience, today)
    if computed is not None and computed > 0:
        if p.total_years_experience and p.total_years_experience > computed + 1.5:
            return computed, f"from dated roles; CV states {p.total_years_experience:g}"
        return computed, "from dated roles"
    if p.total_years_experience is not None:
        return float(p.total_years_experience), "as stated in CV"
    return None, ""


def highest_education(p: CandidateProfile) -> Optional[Education]:
    return max(p.education, key=lambda e: e.level.rank, default=None)


def _primary(loc: str) -> str:
    return loc.split(",")[0].strip().lower()


def evaluate_location(p: CandidateProfile, rule: LocationRule) -> Tuple[CheckStatus, float, str]:
    accepted = [a.strip() for a in rule.accepted_locations if a.strip()]
    if any(a.lower() in ("remote", "anywhere", "any") for a in accepted):
        return CheckStatus.PASS, 1.0, "Role accepts any location"
    loc = (p.current_location or "").strip()
    low = loc.lower()
    if loc:
        for a in accepted:
            prim = _primary(a)
            if prim and (prim in low or (len(_primary(loc)) >= 3 and _primary(loc) in a.lower())):
                return CheckStatus.PASS, 1.0, f"Based in {loc}"
    if rule.accept_relocation and p.willing_to_relocate:
        return CheckStatus.PASS, 0.75, f"Based in {loc or 'unknown location'}; willing to relocate"
    if not loc:
        return CheckStatus.UNKNOWN, 0.4, "Location not stated in CV"
    extra = " (not willing to relocate)" if p.willing_to_relocate is False else ""
    return CheckStatus.FAIL, 0.0, f"Based in {loc}{extra}; accepted: {', '.join(accepted)}"


def evaluate_qualification(p: CandidateProfile, rule: QualificationRule) -> Tuple[CheckStatus, float, str]:
    if not p.education:
        return CheckStatus.UNKNOWN, 0.0, "No education details found in CV"
    req = rule.min_level
    fields = [f for f in rule.fields if f.strip()]

    def field_ok(e: Education) -> bool:
        return not fields or any(phrase_match(f, e.field) or phrase_match(f, e.degree) for f in fields)

    meets_level = [e for e in p.education if e.level.rank >= req.rank]
    full = [e for e in meets_level if field_ok(e)]
    best = highest_education(p)
    req_txt = (req.label if req != DegreeLevel.NONE else "any level") + (f" in {', '.join(fields)}" if fields else "")
    if full:
        e = max(full, key=lambda x: x.level.rank)
        return CheckStatus.PASS, 1.0, f"{e.degree or e.level.label} meets {req_txt}"
    if meets_level:
        e = max(meets_level, key=lambda x: x.level.rank)
        if rule.related_fields_ok:
            return CheckStatus.UNKNOWN, 0.7, (f"{e.degree or e.level.label}: level met; field may be related to "
                                              f"{', '.join(fields)} — review")
        return CheckStatus.FAIL, 0.5, f"{e.degree or e.level.label}: level met but field not in {', '.join(fields)}"
    gap = req.rank - best.level.rank if best else 99
    frac = 0.3 if gap <= 2 else 0.0
    return CheckStatus.FAIL, frac, f"Highest: {best.degree or best.level.label if best else 'none'}; required {req_txt}"


# --------------------------------------------------------------------------------------------
# Main scoring
# --------------------------------------------------------------------------------------------
def score_candidate(
    record: CVRecord,
    criteria: MandatoryCriteria,
    plan: WeightPlan,
    jd_text: str = "",
    engine: Optional[SemanticEngine] = None,
    today: Optional[date] = None,
    jd_hash: str = "",
) -> CandidateResult:
    today = today or date.today()
    base = dict(
        file_name=record.file_name, file_hash=record.file_hash, source=record.source,
        warnings=list(record.warnings), cv_text=record.text,
        stale=bool(jd_hash and record.jd_hash and record.jd_hash != jd_hash),
    )
    if record.error or record.analysis is None:
        return CandidateResult(
            candidate_name=Path(record.file_name).stem, status=CandidateStatus.ERROR,
            error=record.error or "No analysis available.", **base,
        )

    p = record.analysis.profile
    qa = record.analysis.assessment
    text = record.text
    engine = engine or SemanticEngine()
    gate: List[GateCheck] = []
    comp: Dict[str, Tuple[float, str]] = {}

    def add(criterion: str, enf: Enforcement, st: CheckStatus, detail: str):
        gate.append(GateCheck(criterion=criterion, enforcement=enf, status=st, detail=detail))

    # Age (gate only — never scored)
    age = candidate_age(p, today)
    r_age = criteria.age
    if r_age.is_active:
        lo, hi = r_age.min_age, r_age.max_age
        rng = f"{lo or '—'}–{hi or '—'}"
        if age is None:
            add("Age", r_age.enforcement, CheckStatus.UNKNOWN, f"Age not stated in CV (limit {rng})")
        elif (lo and age < lo) or (hi and age > hi):
            add("Age", r_age.enforcement, CheckStatus.FAIL, f"Age {age} outside {rng}")
        else:
            add("Age", r_age.enforcement, CheckStatus.PASS, f"Age {age} within {rng}")

    # Location
    if criteria.location.is_active:
        st, fr, d = evaluate_location(p, criteria.location)
        add("Location", criteria.location.enforcement, st, d)
        comp["location"] = (fr, d)

    # Experience
    years, ysrc = experience_years(p, today)
    r_exp = criteria.experience
    if r_exp.is_active:
        if years is None:
            st, fr, d = CheckStatus.UNKNOWN, 0.0, "Experience could not be determined"
        elif years + 1e-9 >= r_exp.min_years:
            st, fr, d = CheckStatus.PASS, 1.0, f"{years:.1f} yrs ({ysrc}) ≥ {r_exp.min_years:g} required"
        else:
            st, fr, d = CheckStatus.FAIL, years / r_exp.min_years, f"{years:.1f} yrs ({ysrc}) < {r_exp.min_years:g} required"
        add("Experience", r_exp.enforcement, st, d)
        comp["experience"] = (fr, d)

    # Qualification
    if criteria.qualification.is_active:
        st, fr, d = evaluate_qualification(p, criteria.qualification)
        add("Qualification", criteria.qualification.enforcement, st, d)
        comp["qualification"] = (fr, d)

    # Certifications
    r_cert = criteria.certification
    certs_matched: List[str] = []
    certs_missing: List[str] = []
    if r_cert.is_active:
        for c in r_cert.required:
            (certs_matched if cert_match(c, p.certifications, text).matched else certs_missing).append(c)
        need = r_cert.needed
        got = len(certs_matched)
        fr = min(1.0, got / need) if need else 1.0
        d = f"{got}/{len(r_cert.required)} held (need {need})" + (f"; missing {', '.join(certs_missing)}" if certs_missing else "")
        add("Certifications", r_cert.enforcement, CheckStatus.PASS if got >= need else CheckStatus.FAIL, d)
        comp["certification"] = (fr, d)

    # Core skills
    r_sk = criteria.skills
    skill_results = match_many(r_sk.required, p.skills, text, engine) if r_sk.required else []
    skills_matched = [m.required for m in skill_results if m.matched]
    skills_missing = [m.required for m in skill_results if not m.matched]
    if r_sk.is_active:
        cov = len(skills_matched) / len(skill_results) if skill_results else 1.0
        d = f"{len(skills_matched)}/{len(skill_results)} core skills ({cov:.0%}; need {r_sk.min_coverage:.0%})"
        if skills_missing:
            d += f"; missing {', '.join(skills_missing[:8])}"
        add("Core skills", r_sk.enforcement, CheckStatus.PASS if cov + 1e-9 >= r_sk.min_coverage else CheckStatus.FAIL, d)
        comp["skills"] = (cov, d)

    nice_matched = [m.required for m in match_many(criteria.nice_to_have_skills, p.skills, text, engine) if m.matched]

    # ---- Stage 1 verdict
    hard_fail = any(g.status == CheckStatus.FAIL and g.enforcement == Enforcement.HARD for g in gate)
    flagged = any(g.status == CheckStatus.UNKNOWN for g in gate) or any(
        g.status == CheckStatus.FAIL and g.enforcement == Enforcement.SOFT for g in gate
    )
    status = CandidateStatus.DISQUALIFIED if hard_fail else CandidateStatus.FLAGGED if flagged else CandidateStatus.QUALIFIED
    reasons = [f"{g.criterion}: {g.detail}" for g in gate if g.status != CheckStatus.PASS]

    # ---- Stage 2 score
    jd_sim = round(engine.doc_similarity(jd_text, text), 3) if jd_text else None
    sub = plan.compliance.as_dict()
    active = {k: v for k, v in comp.items() if sub.get(k, 0) > 0}
    compliance_details: List[ComplianceDetail] = []
    if active:
        tot_w = sum(sub[k] for k in active)
        comp_frac = 0.0
        for k in COMPLIANCE_KEYS:
            if k in active:
                share = sub[k] / tot_w
                fr, note = active[k]
                fr = max(0.0, min(1.0, fr))
                comp_frac += share * fr
                compliance_details.append(ComplianceDetail(key=k, label=COMPLIANCE_LABELS[k], share=round(share, 3),
                                                           fraction=round(fr, 3), note=note))
        comp_just = "; ".join(f"{c.label} {c.fraction:.0%} (×{c.share:.0%})" for c in compliance_details)
    else:
        comp_frac = min(1.0, (jd_sim or 0.0) / 0.6)
        comp_just = "No scored mandatory criteria configured — based on overall JD similarity."

    w = plan.modules.as_dict()
    modules = [
        ModuleScore(key="mandatory_compliance", label=MODULE_LABELS["mandatory_compliance"],
                    weight=w["mandatory_compliance"], fraction=round(comp_frac, 4),
                    points=round(w["mandatory_compliance"] * comp_frac, 1), justification=comp_just,
                    evidence=[f"Matched skills: {', '.join(skills_matched)}"] if skills_matched else []),
    ]
    for key, assessment in (("academic_professional", qa.academic_professional),
                            ("highlights", qa.highlights), ("extracurricular", qa.extracurricular)):
        frac = assessment.score / 10.0
        modules.append(ModuleScore(key=key, label=MODULE_LABELS[key], weight=w[key], fraction=round(frac, 4),
                                   points=round(w[key] * frac, 1), justification=assessment.justification,
                                   evidence=assessment.evidence))
    total = round(sum(m.points for m in modules), 1)

    best = highest_education(p)
    hq = ""
    if best and best.level != DegreeLevel.NONE:
        hq = best.degree or best.level.label
        if best.grade:
            hq += f" ({best.grade})"

    return CandidateResult(
        candidate_name=p.full_name or Path(record.file_name).stem,
        status=status, total=total, modules=modules, compliance=compliance_details, gate=gate,
        reasons=reasons, years_experience=years, years_source=ysrc, age=age, location=p.current_location,
        highest_qualification=hq, skills_matched=skills_matched, skills_missing=skills_missing,
        certs_matched=certs_matched, certs_missing=certs_missing, nice_matched=nice_matched,
        jd_similarity=jd_sim, profile=p, **base,
    )


def rank_results(results: List[CandidateResult]) -> List[CandidateResult]:
    ordered = sorted(results, key=lambda r: (STATUS_ORDER[r.status], -r.total, r.candidate_name.lower()))
    for i, r in enumerate(ordered, 1):
        r.rank = i
    return ordered
