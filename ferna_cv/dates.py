"""Tolerant date parsing and experience/age arithmetic."""

from __future__ import annotations

import re
from datetime import date
from typing import Iterable, List, Optional, Tuple

from .schemas import CandidateProfile, WorkExperience

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_ONGOING = re.compile(r"^(present|current(ly)?|now|to ?date|till ?date|ongoing|today|continuing)$", re.I)


def _safe(y: int, m: int = 1, d: int = 1) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(value: object, today: Optional[date] = None, day_first: bool = True,
               _search: bool = True) -> Tuple[Optional[date], str]:
    """Parse a loose date. Returns (date, precision) with precision in
    {"day", "month", "year", "ongoing", ""}. Year-only dates are anchored at 1 July so that
    "2018 - 2021" counts as three years."""
    today = today or date.today()
    if value is None:
        return None, ""
    s = str(value).strip().lower().strip(".,;:()")
    if not s:
        return None, ""
    if _ONGOING.match(s):
        return today, "ongoing"

    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", s)  # ISO
    if m:
        return _safe(int(m[1]), int(m[2]), int(m[3])), "day"
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", s)  # dd/mm/yyyy (or mm/dd/yyyy)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        d, mo = (a, b) if day_first else (b, a)
        if mo > 12 and d <= 12:
            d, mo = mo, d
        return _safe(y, mo, d), "day"
    m = re.match(r"^(\d{4})[-/.](\d{1,2})$", s)
    if m and 1 <= int(m[2]) <= 12:
        return _safe(int(m[1]), int(m[2])), "month"
    m = re.match(r"^(\d{1,2})[-/.](\d{4})$", s)
    if m and 1 <= int(m[1]) <= 12:
        return _safe(int(m[2]), int(m[1])), "month"
    m = re.match(
        r"^(?:(\d{1,2})(?:st|nd|rd|th)?\s+)?([a-z]{3,9})\.?,?\s+(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(\d{4})$", s
    )
    if m and m[2][:3] in MONTHS:
        mo = MONTHS[m[2][:3]]
        day = m[1] or m[3]
        if day:
            return _safe(int(m[4]), mo, int(day)), "day"
        return _safe(int(m[4]), mo), "month"
    m = re.match(r"^(\d{4})$", s)
    if m:
        return _safe(int(m[1]), 7, 1), "year"
    return find_date(s, today, day_first) if _search else (None, "")


def find_date(text: str, today: Optional[date] = None, day_first: bool = True) -> Tuple[Optional[date], str]:
    """Search for the first date-like substring inside longer text."""
    patterns = [
        r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}",
        r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}",
        r"(?:\d{1,2}(?:st|nd|rd|th)?\s+)?[a-z]{3,9}\.?,?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?\d{4}",
        r"\d{1,2}[-/.]\d{4}",
        r"\b(?:19|20)\d{2}\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text.lower()):
            d, prec = parse_date(m.group(), today, day_first, _search=False)
            if d:
                return d, prec
    return None, ""


def format_partial(d: Optional[date], precision: str) -> str:
    if d is None:
        return ""
    if precision == "ongoing":
        return "present"
    if precision == "day":
        return d.isoformat()
    if precision == "month":
        return f"{d.year}-{d.month:02d}"
    return str(d.year)


def _month_index(d: date) -> int:
    return d.year * 12 + (d.month - 1)


def experience_intervals(roles: Iterable[WorkExperience], today: Optional[date] = None) -> List[Tuple[int, int]]:
    """Convert roles to half-open month intervals [start, end)."""
    today = today or date.today()
    now = _month_index(today) + 1
    out: List[Tuple[int, int]] = []
    for r in roles:
        s, sp = parse_date(r.start, today)
        if s is None:
            continue
        e, ep = parse_date(r.end, today) if r.end else (None, "")
        if e is None:
            continue  # no end date -> cannot measure safely
        start = _month_index(s)
        end = _month_index(e) + (1 if ep in ("month", "day", "ongoing") else 0)
        if sp == "year" and ep == "year" and end <= start:
            end = start + 6  # "2021 - 2021": count half a year
        end = min(end, now)
        if start >= now or end <= start:
            continue
        out.append((start, end))
    return out


def total_experience_years(roles: Iterable[WorkExperience], today: Optional[date] = None) -> Optional[float]:
    """Total experience with overlapping roles merged (no double counting)."""
    intervals = sorted(experience_intervals(roles, today))
    if not intervals:
        return None
    months = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            months += cur_e - cur_s
            cur_s, cur_e = s, e
    months += cur_e - cur_s
    return round(months / 12.0, 1)


def candidate_age(profile: CandidateProfile, today: Optional[date] = None) -> Optional[int]:
    """Age from an explicitly stated age or date of birth. Never inferred from other data."""
    today = today or date.today()
    if profile.stated_age:
        return profile.stated_age
    if not profile.date_of_birth:
        return None
    dob, prec = parse_date(profile.date_of_birth, today)
    if dob is None or prec == "ongoing" or dob >= today:
        return None
    if prec == "year":
        age = today.year - dob.year
    else:
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age if 14 <= age <= 90 else None
