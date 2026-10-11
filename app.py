"""Ferna CV Filter — Streamlit dashboard.

    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from ferna_cv import APP_NAME, __version__
from ferna_cv.extraction import ExtractionError, approx_tokens, extract_text, file_hash, text_hash
from ferna_cv.heuristic import analyze_jd as heuristic_analyze_jd
from ferna_cv.llm import DEFAULT_MODELS, LLMClient, LLMConfig, LLMError, analyze_jd_llm
from ferna_cv.matching import SemanticEngine
from ferna_cv.pipeline import ScreeningConfig, analyze_batch, cache_key
from ferna_cv.report import results_to_dataframe, results_to_export_df, results_to_json
from ferna_cv.schemas import (
    COMPLIANCE_KEYS,
    COMPLIANCE_LABELS,
    MODULE_KEYS,
    MODULE_LABELS,
    MODULE_SHORT,
    AgeRule,
    CandidateResult,
    CandidateStatus,
    CertificationRule,
    CheckStatus,
    ComplianceSubWeights,
    CVRecord,
    DegreeLevel,
    Enforcement,
    ExperienceRule,
    JDAnalysis,
    LocationRule,
    MandatoryCriteria,
    ModuleWeights,
    QualificationRule,
    SkillsRule,
    WeightPlan,
)
from ferna_cv.scoring import rank_results, score_candidate
from ferna_cv.weights import DEFAULT_RANGES, plan_from_analysis

# ============================================================================================
# Page setup & theme
# ============================================================================================
st.set_page_config(page_title=APP_NAME, page_icon="🗂️", layout="wide", initial_sidebar_state="expanded")

STATUS_COLORS = {
    "Qualified": "#1E7B4F",
    "Flagged": "#B7791F",
    "Disqualified": "#B42318",
    "Error": "#667085",
}
STATUS_ICONS = {"Qualified": "✅", "Flagged": "⚠️", "Disqualified": "⛔", "Error": "🛑"}
CHECK_ICONS = {CheckStatus.PASS: "✅ Pass", CheckStatus.FAIL: "❌ Fail", CheckStatus.UNKNOWN: "❔ Unknown"}

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
.ferna-hero {background: linear-gradient(100deg, #14263F 0%, #1F3A5F 55%, #2E5A88 100%);
  color: #fff; padding: 18px 26px; border-radius: 14px; margin-bottom: 14px;}
.ferna-hero h1 {color: #fff; font-size: 1.65rem; margin: 0; letter-spacing: .2px;}
.ferna-hero p {color: #D6E2F0; margin: 4px 0 0 0; font-size: .95rem;}
div[data-testid="stMetric"] {background: #F4F6FA; border: 1px solid #E3E8EF; border-radius: 12px; padding: 10px 16px;}
.ferna-badge {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .82rem;
  font-weight: 600; color: #fff;}
.ferna-muted {color: #5B6B7F; font-size: .88rem;}
.ferna-card {border: 1px solid #E3E8EF; border-radius: 12px; padding: 12px 16px; background: #FFFFFF;}
</style>
""",
    unsafe_allow_html=True,
)


def badge(status: str) -> str:
    return f'<span class="ferna-badge" style="background:{STATUS_COLORS.get(status, "#667085")}">{status}</span>'


# ============================================================================================
# Session state
# ============================================================================================
ENF_OPTIONS = ["Hard", "Soft", "Off"]
ENF_MAP = {"Hard": Enforcement.HARD, "Soft": Enforcement.SOFT, "Off": Enforcement.OFF}
ENF_HELP = ("Hard = disqualify on failure · Soft = flag for human review · Off = ignore. "
            "Missing information is always flagged, never auto-disqualified.")
WEIGHT_MODES = ["Autonomous (AI-allocated)", "Custom weight matrix"]


def _secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml
        pass
    return os.environ.get(name, "")


DEFAULTS: Dict[str, object] = {
    "jd_text": "", "jd_analysis": None, "records": {}, "uploader_nonce": 0, "_jd_file_id": None, "_flash": None,
    # checklist
    "age_enf": "Off", "age_min": 0, "age_max": 0,
    "loc_enf": "Off", "loc_list": "", "loc_reloc": True,
    "exp_enf": "Hard", "exp_min": 0.0,
    "qual_enf": "Hard", "qual_level": DegreeLevel.NONE.value, "qual_fields": "", "qual_related": True,
    "cert_enf": "Soft", "cert_list": "", "cert_min": 0,
    "skill_enf": "Hard", "skill_list": "", "skill_cov": 70, "nice_list": "",
    # weights
    "weight_mode": WEIGHT_MODES[0], "enforce_bands": True,
    "w_mand": 50, "w_acad": 20, "w_high": 15, "w_extra": 15,
    "sw_skills": 35, "sw_exp": 25, "sw_qual": 20, "sw_cert": 10, "sw_loc": 10,
    # sidebar
    "api_key_openai": _secret("OPENAI_API_KEY"), "api_key_anthropic": _secret("ANTHROPIC_API_KEY"),
    "model_openai": DEFAULT_MODELS["openai"], "model_anthropic": DEFAULT_MODELS["anthropic"],
    "base_url": _secret("OPENAI_BASE_URL"), "temperature": 0.0, "max_tokens": 6000, "workers": 4,
    "semantic_backend": "Lexical (built-in, fast)", "redact_pii": True,
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, dict) else _v
S = st.session_state


def split_list(text: str, newline_only: bool = False) -> List[str]:
    pattern = r"[\n;]+" if newline_only else r"[\n,;]+"
    out, seen = [], set()
    for item in re.split(pattern, text or ""):
        item = item.strip(" •-\t")
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


# ============================================================================================
# Sidebar — provider configuration
# ============================================================================================
PROVIDERS = {
    "Offline heuristic (no API key)": "heuristic",
    "OpenAI / OpenAI-compatible": "openai",
    "Anthropic Claude": "anthropic",
}

with st.sidebar:
    st.markdown(f"### 🗂️ {APP_NAME}")
    st.caption(f"v{__version__} · Ferna Software Solutions")
    provider_label = st.selectbox("AI provider", list(PROVIDERS), key="provider_label",
                                  help="LLM mode gives far better extraction and relevance judgements. "
                                       "Offline mode is rule-based and keeps all data on this machine.")
    provider = PROVIDERS[provider_label]
    if provider == "heuristic":
        st.info("Rule-based extraction — nothing leaves this machine. Ideal for demos; use an LLM for real screening.",
                icon="🔒")
        llm_cfg = LLMConfig(provider="heuristic")
    else:
        st.text_input("API key", type="password", key=f"api_key_{provider}",
                      help="Stored only in this browser session. You can also set OPENAI_API_KEY / "
                           "ANTHROPIC_API_KEY as environment variables or in .streamlit/secrets.toml.")
        st.text_input("Model", key=f"model_{provider}", help="Any chat model your key can access.")
        if provider == "openai":
            st.text_input("Base URL (optional)", key="base_url",
                          help="For Azure OpenAI, OpenRouter, Ollama (http://localhost:11434/v1), vLLM, LM Studio…")
        with st.expander("Model parameters"):
            st.slider("Temperature", 0.0, 1.0, step=0.05, key="temperature",
                      help="0 gives the most repeatable scores.")
            st.number_input("Max output tokens", 1000, 32000, step=500, key="max_tokens")
            st.slider("Parallel requests", 1, 8, key="workers", help="Lower this if you hit rate limits.")
        llm_cfg = LLMConfig(
            provider=provider, api_key=S[f"api_key_{provider}"].strip(), model=S[f"model_{provider}"].strip(),
            base_url=S["base_url"].strip() if provider == "openai" else "", temperature=S["temperature"],
            max_tokens=int(S["max_tokens"]),
        )
        if not llm_cfg.api_key and not llm_cfg.base_url:
            st.warning("Enter an API key to enable processing.", icon="🔑")

    st.divider()
    st.selectbox("Semantic similarity", ["Lexical (built-in, fast)", "Transformer embeddings (sentence-transformers)"],
                 key="semantic_backend",
                 help="Transformer mode uses all-MiniLM-L6-v2 for skill matching and JD↔CV similarity. "
                      "Requires `pip install sentence-transformers` (downloads ~90 MB once).")
    st.toggle("Redact e-mail / phone / URLs before sending to the LLM", key="redact_pii")
    st.caption("CV content is sent to the selected provider for analysis. Inform applicants and follow your data "
               "protection obligations (e.g. Sri Lanka PDPA, GDPR). Treat scores as decision support — a human "
               "should review every shortlist.")
    if st.button("Reset session", width="stretch"):
        for k in list(S.keys()):
            del S[k]
        st.rerun()


@st.cache_resource(show_spinner="Loading embedding model…")
def get_engine(backend_label: str) -> SemanticEngine:
    return SemanticEngine("sbert" if backend_label.startswith("Transformer") else "lexical")


engine = get_engine(S["semantic_backend"])
if S["semantic_backend"].startswith("Transformer") and not engine.has_embeddings:
    st.sidebar.warning(f"Transformer model unavailable — using lexical similarity. ({engine.load_error[:120]})")

# ============================================================================================
# Header
# ============================================================================================
st.markdown(
    f"""<div class="ferna-hero"><h1>{APP_NAME}</h1>
    <p>AI-assisted CV screening · mandatory gatekeeper + weighted scoring out of 100 · {llm_cfg.label}</p></div>""",
    unsafe_allow_html=True,
)

tab_setup, tab_upload, tab_board = st.tabs(
    ["① Job Setup & Weight Configurator", "② CV Batch Upload & Processing", "③ Candidate Leaderboard & Analytics"]
)


# ============================================================================================
# Helpers bound to session state
# ============================================================================================
def apply_jd_analysis(a: JDAnalysis) -> List[str]:
    """Copy a JD analysis into the checklist widgets. Must run before those widgets render."""
    notes = []
    S["jd_analysis"] = a
    if a.min_years_experience:
        S["exp_min"] = float(a.min_years_experience)
    if a.min_degree_level != DegreeLevel.NONE:
        S["qual_level"] = a.min_degree_level.value
    if a.degree_fields:
        S["qual_fields"] = ", ".join(a.degree_fields)
    S["qual_related"] = bool(a.related_fields_ok) or not a.degree_fields
    S["cert_list"] = "\n".join(a.required_certifications)
    S["skill_list"] = "\n".join(a.core_skills)
    S["nice_list"] = "\n".join(a.nice_to_have_skills)
    if a.locations:
        S["loc_list"] = ", ".join(a.locations)
        if S["loc_enf"] == "Off":
            S["loc_enf"] = "Soft"
            notes.append("Location rule switched on as **Soft** — change to Hard if relocation is not possible.")
        S["loc_reloc"] = a.relocation_acceptable is not False
    if a.min_age or a.max_age:
        S["age_min"], S["age_max"] = int(a.min_age or 0), int(a.max_age or 0)
        notes.append("The JD states an age limit. The age rule is filled in but left **Off** — enable it only "
                     "where age criteria are lawful for this role.")
    return notes


def build_criteria() -> MandatoryCriteria:
    return MandatoryCriteria(
        age=AgeRule(enforcement=ENF_MAP[S["age_enf"]], min_age=int(S["age_min"]) or None,
                    max_age=int(S["age_max"]) or None),
        location=LocationRule(enforcement=ENF_MAP[S["loc_enf"]], accepted_locations=split_list(S["loc_list"]),
                              accept_relocation=bool(S["loc_reloc"])),
        experience=ExperienceRule(enforcement=ENF_MAP[S["exp_enf"]], min_years=float(S["exp_min"])),
        qualification=QualificationRule(enforcement=ENF_MAP[S["qual_enf"]], min_level=DegreeLevel(S["qual_level"]),
                                        fields=split_list(S["qual_fields"]), related_fields_ok=bool(S["qual_related"])),
        certification=CertificationRule(enforcement=ENF_MAP[S["cert_enf"]],
                                        required=split_list(S["cert_list"], newline_only=True),
                                        min_required=int(S["cert_min"])),
        skills=SkillsRule(enforcement=ENF_MAP[S["skill_enf"]], required=split_list(S["skill_list"]),
                          min_coverage=S["skill_cov"] / 100.0),
        nice_to_have_skills=split_list(S["nice_list"]),
    )


def donut(weights: Dict[str, int]):
    df = pd.DataFrame({"Module": [MODULE_SHORT[k] for k in MODULE_KEYS], "Points": [weights[k] for k in MODULE_KEYS]})
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=58, cornerRadius=3)
        .encode(
            theta="Points:Q",
            color=alt.Color("Module:N", scale=alt.Scale(range=["#1F3A5F", "#3E7CB1", "#7FB3D5", "#C9DCEB"]),
                            sort=[MODULE_SHORT[k] for k in MODULE_KEYS], legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=["Module", "Points"],
        )
        .properties(height=260)
    )


# ============================================================================================
# TAB 1 — Job setup & weights
# ============================================================================================
with tab_setup:
    col_jd, col_check = st.columns([1.05, 1], gap="large")

    with col_jd:
        st.subheader("Job description")
        jd_file = st.file_uploader("Upload JD (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"], key="jd_file")
        if jd_file is not None:
            fid = (jd_file.name, jd_file.size)
            if S["_jd_file_id"] != fid:
                try:
                    S["jd_text"], _ = extract_text(jd_file.name, jd_file.getvalue())
                    S["_jd_file_id"] = fid
                except ExtractionError as exc:
                    st.error(f"Could not read the JD: {exc}")
        st.text_area("…or paste / edit the JD", key="jd_text", height=330,
                     placeholder="Paste the full job description here.")
        jd_ready = bool(S["jd_text"].strip())
        analyse = st.button("✨ Analyse JD & auto-fill checklist", type="primary", disabled=not jd_ready,
                            help="Extracts mandatory requirements and proposes an autonomous weight breakdown.")
        if analyse:
            try:
                with st.spinner("Analysing the job description…"):
                    if llm_cfg.provider == "heuristic":
                        analysis = heuristic_analyze_jd(S["jd_text"])
                    else:
                        analysis = analyze_jd_llm(LLMClient(llm_cfg), S["jd_text"])
                notes = apply_jd_analysis(analysis)
                S["_flash"] = ("success", f"Checklist filled from JD: **{analysis.job_title or 'role'}** "
                                          f"({analysis.seniority}, complexity {analysis.complexity}/5). Review below.",
                               notes)
            except LLMError as exc:
                S["_flash"] = ("error", f"JD analysis failed: {exc}", [])
        if S["_flash"]:
            kind, msg, notes = S["_flash"]
            (st.success if kind == "success" else st.error)(msg)
            for n in notes:
                st.info(n, icon="ℹ️")

    with col_check:
        st.subheader("Mandatory requirements checklist")
        st.caption(ENF_HELP)

        with st.expander("🎓 Experience & qualifications", expanded=True):
            st.radio("Experience rule", ENF_OPTIONS, key="exp_enf", horizontal=True)
            st.number_input("Minimum years of experience", 0.0, 50.0, step=0.5, key="exp_min")
            st.radio("Qualification rule", ENF_OPTIONS, key="qual_enf", horizontal=True)
            st.selectbox("Minimum qualification", [lvl.value for lvl in DegreeLevel], key="qual_level",
                         format_func=lambda v: DegreeLevel(v).label)
            st.text_input("Accepted fields of study (comma-separated, any)", key="qual_fields",
                          placeholder="Computer Science, Statistics, Mathematics")
            st.checkbox("Other fields → flag for review instead of failing (JD says 'or a related field')",
                        key="qual_related")

        with st.expander("🛠️ Core technical skills", expanded=True):
            st.radio("Skills rule", ENF_OPTIONS, key="skill_enf", horizontal=True)
            st.text_area("Core skills (one per line or comma-separated)", key="skill_list", height=110)
            st.slider("Minimum coverage of core skills (%)", 0, 100, step=5, key="skill_cov")
            st.text_area("Nice-to-have skills (reported, not gated)", key="nice_list", height=70)

        with st.expander("📜 Certifications"):
            st.radio("Certification rule", ENF_OPTIONS, key="cert_enf", horizontal=True)
            st.text_area("Required certifications (one per line)", key="cert_list", height=80)
            st.number_input("How many are required? (0 = all)", 0, 20, step=1, key="cert_min")

        with st.expander("📍 Location / relocation"):
            st.radio("Location rule", ENF_OPTIONS, key="loc_enf", horizontal=True)
            st.text_input("Accepted locations (comma-separated; 'Remote' accepts anyone)", key="loc_list")
            st.checkbox("Accept candidates willing to relocate", key="loc_reloc")

        with st.expander("🎂 Age limit"):
            st.radio("Age rule", ENF_OPTIONS, key="age_enf", horizontal=True)
            a1, a2 = st.columns(2)
            a1.number_input("Minimum age (0 = none)", 0, 90, step=1, key="age_min")
            a2.number_input("Maximum age (0 = none)", 0, 90, step=1, key="age_max")
            st.caption("⚖️ Age criteria are unlawful for many roles in some jurisdictions (e.g. UK Equality Act "
                       "2010; US ADEA for applicants 40+). Use only where permitted. Age is taken solely from an "
                       "explicitly stated age or date of birth — never inferred — and a missing age is flagged.")

    st.divider()
    st.subheader("Stage 2 scoring weights (out of 100)")
    st.radio("Weighting mode", WEIGHT_MODES, key="weight_mode", horizontal=True)

    plan: Optional[WeightPlan] = None
    wcol1, wcol2 = st.columns([1.3, 1], gap="large")
    if S["weight_mode"] == WEIGHT_MODES[0]:
        with wcol1:
            st.toggle("Keep weights inside the default bands", key="enforce_bands",
                      help="Compliance 40–60 · Achievements 15–20 · Highlights 10–15 · Extra-curricular 10–15")
            analysis = S["jd_analysis"] or (heuristic_analyze_jd(S["jd_text"]) if jd_ready else JDAnalysis())
            plan = plan_from_analysis(analysis, enforce_ranges=S["enforce_bands"])
            source = "AI analysis of the JD" if S["jd_analysis"] is not None and llm_cfg.provider != "heuristic" \
                else "rule-based JD complexity estimate"
            if not S["jd_analysis"]:
                st.caption("Run **Analyse JD** for an allocation from the selected provider; showing a rule-based "
                           "estimate meanwhile.")
            mw = plan.modules.as_dict()
            mcols = st.columns(4)
            for c, k in zip(mcols, MODULE_KEYS):
                lo, hi = DEFAULT_RANGES[k]
                c.metric(MODULE_SHORT[k], f"{mw[k]} pts", help=f"{MODULE_LABELS[k]} · default band {lo}–{hi}")
            st.markdown(f"**Rationale** ({source}): {plan.rationale}")
            sub = plan.compliance.as_dict()
            tot = sum(sub.values()) or 1
            st.caption("Inside mandatory compliance: " + " · ".join(
                f"{COMPLIANCE_LABELS[k]} {sub[k] / tot:.0%}" for k in COMPLIANCE_KEYS))
        with wcol2:
            st.altair_chart(donut(plan.modules.as_dict()), width="stretch")
    else:
        with wcol1:
            st.caption("Allocate marks to each module; they must total exactly 100.")
            c = st.columns(4)
            c[0].number_input("Mandatory compliance", 0, 100, step=1, key="w_mand")
            c[1].number_input("Academic & professional", 0, 100, step=1, key="w_acad")
            c[2].number_input("Highlights", 0, 100, step=1, key="w_high")
            c[3].number_input("Extra-curricular", 0, 100, step=1, key="w_extra")
            total_w = S["w_mand"] + S["w_acad"] + S["w_high"] + S["w_extra"]
            st.markdown("**Relative weight of each criterion inside mandatory compliance**")
            s = st.columns(5)
            s[0].slider("Skills", 0, 100, key="sw_skills")
            s[1].slider("Experience", 0, 100, key="sw_exp")
            s[2].slider("Qualification", 0, 100, key="sw_qual")
            s[3].slider("Certification", 0, 100, key="sw_cert")
            s[4].slider("Location", 0, 100, key="sw_loc")
            if total_w != 100:
                st.error(f"Module weights total **{total_w}** — adjust to exactly 100.")
            else:
                st.success("Weights total 100.")
                plan = WeightPlan(
                    mode="custom",
                    modules=ModuleWeights(mandatory_compliance=S["w_mand"], academic_professional=S["w_acad"],
                                          highlights=S["w_high"], extracurricular=S["w_extra"]),
                    compliance=ComplianceSubWeights(skills=S["sw_skills"], experience=S["sw_exp"],
                                                    qualification=S["sw_qual"], certification=S["sw_cert"],
                                                    location=S["sw_loc"]),
                    rationale="Custom weight matrix set by the employer.",
                )
        with wcol2:
            if plan:
                st.altair_chart(donut(plan.modules.as_dict()), width="stretch")

criteria = build_criteria()
jd_text: str = S["jd_text"]
jd_hash = text_hash(jd_text)


# ============================================================================================
# TAB 2 — CV batch upload & processing
# ============================================================================================
def record_row(rec: CVRecord) -> Dict[str, object]:
    return {
        "File": rec.file_name,
        "Status": "🛑 Error" if rec.error else "✅ Analysed",
        "Candidate": (rec.analysis.profile.full_name if rec.analysis else ""),
        "Characters": len(rec.text),
        "≈ Tokens": approx_tokens(rec.text) if rec.text else 0,
        "Seconds": rec.seconds,
        "Analysed by": rec.source,
        "Message": rec.error or " | ".join(rec.warnings),
    }


with tab_upload:
    st.subheader("Upload CVs")
    files = st.file_uploader(
        "Drop CVs here — bulk upload supported (PDF, DOCX, TXT)", type=["pdf", "docx", "txt", "doc"],
        accept_multiple_files=True, key=f"cv_uploader_{S['uploader_nonce']}",
    )
    files = files or []
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files selected", len(files))
    c2.metric("Total size", f"{sum(f.size for f in files) / 1024:,.0f} KB")
    c3.metric("Analysed CVs in session", len(S["records"]))
    c4.metric("Engine", "Offline" if llm_cfg.provider == "heuristic" else llm_cfg.provider.title())

    if not jd_ready:
        st.warning("No job description yet — CVs can still be parsed, but relevance scoring needs a JD (Tab ①).")
    key_missing = llm_cfg.provider != "heuristic" and not (llm_cfg.api_key or llm_cfg.base_url)
    reprocess = st.checkbox("Re-analyse files that were already processed with this JD and model")
    go = st.button("🚀 Process CVs", type="primary", disabled=not files or key_missing)

    if go:
        cfg = ScreeningConfig(llm=llm_cfg, redact_pii=S["redact_pii"])
        payload, cached, seen = [], 0, set()
        for f in files:
            data = f.getvalue()
            fh = file_hash(data) if data else f"empty-{f.name}"
            if fh in seen:
                continue  # duplicate upload in the same batch
            seen.add(fh)
            existing: Optional[CVRecord] = S["records"].get(fh)
            if existing and not reprocess and not existing.error and existing.cache_key == cache_key(fh, jd_text, llm_cfg):
                cached += 1
                continue
            payload.append((f.name, data))

        if not payload:
            st.info(f"All {cached} file(s) were already analysed with this JD and model — see Tab ③.")
        else:
            progress = st.progress(0.0, text=f"Processing 0/{len(payload)}…")
            live = st.empty()
            rows: List[Dict[str, object]] = []

            def on_done(rec: CVRecord) -> None:
                S["records"][rec.file_hash] = rec
                rows.append(record_row(rec))
                progress.progress(len(rows) / len(payload), text=f"Processed {len(rows)}/{len(payload)} — {rec.file_name}")
                live.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            try:
                started = datetime.now()
                done = analyze_batch(payload, jd_text, cfg, max_workers=int(S["workers"]), on_done=on_done)
                ok = sum(1 for r in done if not r.error)
                secs = (datetime.now() - started).total_seconds()
                progress.progress(1.0, text="Done")
                msg = f"Analysed {ok}/{len(done)} CV(s) in {secs:.1f}s" + (f"; {cached} reused from cache" if cached else "")
                (st.success if ok == len(done) else st.warning)(msg + ". Open Tab ③ for the leaderboard.")
            except LLMError as exc:
                st.error(f"Could not start processing: {exc}")

    if S["records"]:
        st.markdown("#### Processing status")
        status_df = pd.DataFrame([record_row(r) for r in S["records"].values()])
        st.dataframe(status_df, hide_index=True, width="stretch",
                     column_config={"Seconds": st.column_config.NumberColumn(format="%.1f")})
        if st.button("🗑️ Clear all results"):
            S["records"] = {}
            S["uploader_nonce"] += 1
            st.rerun()


# ============================================================================================
# TAB 3 — Leaderboard & analytics
# ============================================================================================
@st.dialog("Candidate deep-dive", width="large")
def show_candidate(r: CandidateResult) -> None:
    st.markdown(f"## {r.candidate_name} &nbsp; {badge(r.status.value)}", unsafe_allow_html=True)
    p = r.profile
    contact = " · ".join(x for x in [p.email if p else "", p.phone if p else "", r.location, r.file_name] if x)
    st.markdown(f'<span class="ferna-muted">{contact}</span>', unsafe_allow_html=True)
    if r.stale:
        st.warning("Analysed against an earlier version of the JD — re-process for up-to-date relevance scores.")
    if r.error:
        st.error(r.error)
        return

    top = st.columns(5)
    top[0].metric("Total score", f"{r.total:.1f} / 100")
    for col, m in zip(top[1:], r.modules):
        col.metric(MODULE_SHORT[m.key], f"{m.points:.1f} / {m.weight}")

    st.markdown("#### Stage 1 — Mandatory gatekeeper")
    if r.gate:
        st.dataframe(pd.DataFrame([{
            "Criterion": g.criterion, "Rule": g.enforcement.value.title(), "Result": CHECK_ICONS[g.status],
            "Detail": g.detail} for g in r.gate]), hide_index=True, width="stretch")
    else:
        st.caption("No mandatory criteria are switched on.")

    st.markdown("#### Stage 2 — Weighted score breakdown")
    for m in r.modules:
        st.progress(min(1.0, m.fraction), text=f"**{m.label}** — {m.points:.1f} / {m.weight} pts ({m.fraction:.0%})")
        if m.justification:
            st.caption(m.justification)
        if m.evidence:
            with st.expander("Evidence", expanded=False):
                for e in m.evidence:
                    st.markdown(f"- {e}")
    if r.compliance:
        st.dataframe(pd.DataFrame([{"Criterion": c.label, "Share of compliance": f"{c.share:.0%}",
                                    "Achieved": f"{c.fraction:.0%}", "Note": c.note} for c in r.compliance]),
                     hide_index=True, width="stretch")

    st.markdown("#### Extracted profile")
    t_sum, t_edu, t_exp, t_skill, t_act, t_raw = st.tabs(
        ["Summary", "Education", "Experience", "Skills & certs", "Activities", "CV text"])
    with t_sum:
        st.write(p.summary or "—")
        info = {"Experience": f"{r.years_experience:.1f} yrs ({r.years_source})" if r.years_experience is not None else "—",
                "Highest qualification": r.highest_qualification or "—", "Location": r.location or "—",
                "Willing to relocate": {True: "Yes", False: "No", None: "Not stated"}[p.willing_to_relocate],
                "Age (stated only)": r.age if r.age is not None else "Not stated",
                "Languages": ", ".join(p.languages) or "—",
                "JD similarity": f"{r.jd_similarity:.2f}" if r.jd_similarity is not None else "—",
                "Analysed by": r.source}
        st.table(pd.DataFrame(info.items(), columns=["Field", "Value"]).astype(str).set_index("Field"))
    with t_edu:
        if p.education:
            st.dataframe(pd.DataFrame([e.model_dump(mode="json") for e in p.education]), hide_index=True, width="stretch")
        else:
            st.caption("No education entries extracted.")
    with t_exp:
        if p.work_experience:
            st.dataframe(pd.DataFrame([w.model_dump() for w in p.work_experience]), hide_index=True, width="stretch")
        else:
            st.caption("No dated roles extracted.")
    with t_skill:
        s1, s2 = st.columns(2)
        s1.markdown("**Core skills matched**  \n" + (", ".join(r.skills_matched) or "—"))
        s1.markdown("**Core skills missing**  \n" + (", ".join(r.skills_missing) or "—"))
        s1.markdown("**Nice-to-have matched**  \n" + (", ".join(r.nice_matched) or "—"))
        s2.markdown("**Certifications held**  \n" + ("\n".join(f"- {c}" for c in p.certifications) or "—"))
        if r.certs_missing:
            s2.markdown("**Required certifications missing**  \n" + ", ".join(r.certs_missing))
        st.markdown("**All extracted skills**  \n" + (", ".join(p.skills) or "—"))
    with t_act:
        for title, items in [("Academic achievements", p.academic_achievements),
                             ("Professional achievements", p.professional_achievements),
                             ("Teamwork & leadership", p.teamwork_leadership),
                             ("Special accomplishments", p.special_accomplishments),
                             ("Sports", p.sports), ("Societies", p.societies), ("Volunteer work", p.volunteer_work)]:
            if items:
                st.markdown(f"**{title}**")
                for it in items:
                    st.markdown(f"- {it}")
    with t_raw:
        st.text_area("Extracted text", r.cv_text, height=320, disabled=True, label_visibility="collapsed")
    if r.warnings:
        st.caption("Warnings: " + " | ".join(r.warnings))


with tab_board:
    records: List[CVRecord] = list(S["records"].values())
    if not records:
        st.info("No CVs analysed yet. Upload and process CVs in Tab ②.", icon="📥")
    elif plan is None:
        st.error("The custom weights in Tab ① do not total 100 — fix them to see scores.")
    else:
        results = rank_results([score_candidate(r, criteria, plan, jd_text, engine, jd_hash=jd_hash) for r in records])
        n_stale = sum(r.stale for r in results)
        if n_stale:
            st.warning(f"{n_stale} CV(s) were analysed against an earlier JD. Criteria and weights are re-applied "
                       "live, but tick 'Re-analyse' in Tab ② to refresh their qualitative assessments.")

        counts = {s.value: sum(r.status == s for r in results) for s in CandidateStatus}
        scored = [r.total for r in results if r.status != CandidateStatus.ERROR]
        k = st.columns(5)
        k[0].metric("Candidates", len(results))
        k[1].metric("✅ Qualified", counts["Qualified"])
        k[2].metric("⚠️ Flagged", counts["Flagged"])
        k[3].metric("⛔ Disqualified", counts["Disqualified"])
        k[4].metric("Average score", f"{(sum(scored) / len(scored)) if scored else 0:.1f}")

        f1, f2, f3 = st.columns([2, 1.2, 1.4])
        present = [s.value for s in CandidateStatus if counts[s.value]]
        status_filter = f1.multiselect("Status", [s.value for s in CandidateStatus], default=present)
        min_score = f2.slider("Minimum score", 0, 100, 0)
        query = f3.text_input("Search name / file", placeholder="e.g. Perera")
        view = [r for r in results if r.status.value in status_filter and r.total >= min_score
                and (not query or query.lower() in (r.candidate_name + " " + r.file_name).lower())]

        if not view:
            st.info("No candidates match the filters.")
        else:
            df = results_to_dataframe(view)
            df.insert(2, "", df["Status"].map(STATUS_ICONS))
            event = st.dataframe(
                df, hide_index=True, width="stretch", key="leaderboard", on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "": st.column_config.TextColumn(width="small"),
                    "Score": st.column_config.ProgressColumn("Score /100", min_value=0, max_value=100, format="%.1f"),
                    **{MODULE_SHORT[k]: st.column_config.NumberColumn(
                        f"{MODULE_SHORT[k]} /{plan.modules.as_dict()[k]}", format="%.1f") for k in MODULE_KEYS},
                    "Experience (yrs)": st.column_config.NumberColumn(format="%.1f"),
                    "JD similarity": st.column_config.NumberColumn(format="%.2f"),
                    "Gate notes": st.column_config.TextColumn(width="large"),
                },
            )
            selected = event.selection.rows if event is not None and hasattr(event, "selection") else []
            labels = [f"#{r.rank} · {r.candidate_name} · {r.total:.1f} · {r.status.value}" for r in view]
            d1, d2 = st.columns([3, 1])
            pick = d1.selectbox("Candidate for deep-dive (or select a row above)", range(len(view)),
                                index=selected[0] if selected else 0, format_func=lambda i: labels[i])
            d2.write("")
            if d2.button("🔍 Open deep-dive", type="primary", width="stretch"):
                show_candidate(view[pick])

        # ---- Downloads
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        e1, e2, _ = st.columns([1, 1, 2])
        e1.download_button("⬇️ Download results (CSV)", results_to_export_df(results).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"ferna-cv-filter-results-{stamp}.csv", mime="text/csv", width="stretch")
        e2.download_button("⬇️ Full detail (JSON)", results_to_json(results).encode("utf-8"),
                           file_name=f"ferna-cv-filter-results-{stamp}.json", mime="application/json", width="stretch")

        # ---- Analytics
        st.divider()
        st.subheader("Analytics")
        ok = [r for r in results if r.status != CandidateStatus.ERROR]
        if ok:
            g1, g2 = st.columns(2)
            dist = pd.DataFrame({"Score": [r.total for r in ok], "Status": [r.status.value for r in ok],
                                 "Candidate": [r.candidate_name for r in ok]})
            color = alt.Color("Status:N", scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())))
            g1.markdown("**Score distribution**")
            g1.altair_chart(
                alt.Chart(dist).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("Score:Q", bin=alt.Bin(extent=[0, 100], step=10), title="Score (/100)"),
                    y=alt.Y("count():Q", title="Candidates"), color=color, tooltip=["Status", "count()"],
                ).properties(height=260), width="stretch")

            mod_rows = []
            for kk in MODULE_KEYS:
                pts = [m.points for r in ok for m in r.modules if m.key == kk]
                mod_rows.append({"Module": MODULE_SHORT[kk], "Average points": sum(pts) / len(pts),
                                 "Available": plan.modules.as_dict()[kk]})
            mdf = pd.DataFrame(mod_rows)
            g2.markdown("**Average points per module vs. available**")
            base = alt.Chart(mdf).encode(y=alt.Y("Module:N", sort=None, title=None))
            g2.altair_chart(
                (base.mark_bar(color="#E3E8EF").encode(x=alt.X("Available:Q", title="Points"))
                 + base.mark_bar(color="#1F3A5F", height=12).encode(x="Average points:Q",
                                                                     tooltip=["Module", alt.Tooltip("Average points:Q", format=".1f"), "Available"]))
                .properties(height=260), width="stretch")

            fails = {}
            for r in ok:
                for g in r.gate:
                    if g.status != CheckStatus.PASS:
                        key = (g.criterion, "Fail" if g.status == CheckStatus.FAIL else "Unknown")
                        fails[key] = fails.get(key, 0) + 1
            if fails:
                st.markdown("**Where candidates fall short (Stage 1)**")
                fdf = pd.DataFrame([{"Criterion": c, "Outcome": o, "Candidates": n} for (c, o), n in fails.items()])
                st.altair_chart(
                    alt.Chart(fdf).mark_bar().encode(
                        x=alt.X("Candidates:Q"), y=alt.Y("Criterion:N", sort="-x", title=None),
                        color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Fail", "Unknown"], range=["#B42318", "#B7791F"])),
                        tooltip=["Criterion", "Outcome", "Candidates"],
                    ).properties(height=40 + 34 * len({c for c, _ in fails})), width="stretch")

            missing = {}
            for r in ok:
                for s_ in r.skills_missing:
                    missing[s_] = missing.get(s_, 0) + 1
            if missing:
                st.markdown("**Most commonly missing core skills**")
                st.bar_chart(pd.Series(missing, name="Candidates").sort_values(ascending=False).head(12),
                             color="#3E7CB1", horizontal=True)

st.markdown(
    f'<div class="ferna-muted" style="text-align:center;margin-top:28px">{APP_NAME} v{__version__} · '
    f'Ferna Software Solutions · Decision support only — final hiring decisions rest with people.</div>',
    unsafe_allow_html=True,
)
