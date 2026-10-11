"""Tabular views and exports of scored candidates."""

from __future__ import annotations

import json
from typing import List

import pandas as pd

from .schemas import MODULE_KEYS, MODULE_SHORT, CandidateResult


def _module_map(r: CandidateResult):
    return {m.key: m for m in r.modules}


def results_to_dataframe(results: List[CandidateResult]) -> pd.DataFrame:
    """Leaderboard view (one row per candidate)."""
    rows = []
    for r in results:
        mm = _module_map(r)
        row = {"Rank": r.rank, "Candidate": r.candidate_name, "Status": r.status.value, "Score": r.total}
        for k in MODULE_KEYS:
            row[MODULE_SHORT[k]] = mm[k].points if k in mm else None
        n_sk = len(r.skills_matched) + len(r.skills_missing)
        row.update({
            "Experience (yrs)": r.years_experience,
            "Highest qualification": r.highest_qualification,
            "Location": r.location,
            "Skills": f"{len(r.skills_matched)}/{n_sk}" if n_sk else "",
            "JD similarity": r.jd_similarity,
            "Gate notes": r.error or " | ".join(r.reasons),
            "File": r.file_name,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def results_to_export_df(results: List[CandidateResult]) -> pd.DataFrame:
    """Full CSV export: marks, weights and justification notes per module."""
    rows = []
    for r in results:
        mm = _module_map(r)
        p = r.profile
        row = {
            "Rank": r.rank, "Candidate": r.candidate_name, "Status": r.status.value, "Total (/100)": r.total,
            "Email": p.email if p else "", "Phone": p.phone if p else "",
        }
        for k in MODULE_KEYS:
            m = mm.get(k)
            short = MODULE_SHORT[k]
            row[f"{short} points"] = m.points if m else None
            row[f"{short} weight"] = m.weight if m else None
            row[f"{short} justification"] = m.justification if m else ""
        row.update({
            "Gate checks": " | ".join(f"{g.criterion} [{g.enforcement.value}]: {g.status.value} — {g.detail}" for g in r.gate),
            "Experience (yrs)": r.years_experience,
            "Experience basis": r.years_source,
            "Age": r.age,
            "Location": r.location,
            "Highest qualification": r.highest_qualification,
            "Skills matched": ", ".join(r.skills_matched),
            "Skills missing": ", ".join(r.skills_missing),
            "Certifications missing": ", ".join(r.certs_missing),
            "Nice-to-have matched": ", ".join(r.nice_matched),
            "JD similarity": r.jd_similarity,
            "File": r.file_name,
            "Analysed by": r.source,
            "Warnings": " | ".join(r.warnings),
            "Error": r.error,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def results_to_json(results: List[CandidateResult]) -> str:
    return json.dumps([r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False)
