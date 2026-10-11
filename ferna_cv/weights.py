"""Weight allocation for the Stage 2 score.

``fit_to_ranges`` turns any raw suggestion (from the LLM or the heuristic) into integer
module weights that (a) sit inside the default bands and (b) total exactly 100.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

from .schemas import (
    MODULE_KEYS,
    ComplianceSubWeights,
    JDAnalysis,
    ModuleWeights,
    WeightPlan,
)

DEFAULT_RANGES: Dict[str, Tuple[int, int]] = {
    "mandatory_compliance": (40, 60),
    "academic_professional": (15, 20),
    "highlights": (10, 15),
    "extracurricular": (10, 15),
}
OPEN_RANGES: Dict[str, Tuple[int, int]] = {k: (0, 100) for k in MODULE_KEYS}


def fit_to_ranges(
    raw: Mapping[str, float],
    ranges: Optional[Mapping[str, Tuple[int, int]]] = None,
    total: int = 100,
) -> Dict[str, int]:
    """Clamp to bands, redistribute the residual in proportion to available room/slack, then
    round with the largest-remainder method without leaving the bands."""
    ranges = ranges or DEFAULT_RANGES
    lo = {k: ranges[k][0] for k in MODULE_KEYS}
    hi = {k: ranges[k][1] for k in MODULE_KEYS}
    if sum(lo.values()) > total or sum(hi.values()) < total:
        raise ValueError("Infeasible weight bands for the requested total.")

    x = {k: min(hi[k], max(lo[k], float(raw.get(k, 0) or 0))) for k in MODULE_KEYS}
    diff = total - sum(x.values())
    if diff > 1e-9:
        room = {k: hi[k] - x[k] for k in MODULE_KEYS}
        tot = sum(room.values())
        for k in MODULE_KEYS:
            x[k] += diff * room[k] / tot
    elif diff < -1e-9:
        slack = {k: x[k] - lo[k] for k in MODULE_KEYS}
        tot = sum(slack.values())
        for k in MODULE_KEYS:
            x[k] += diff * slack[k] / tot  # diff is negative

    ints = {k: int(math.floor(x[k] + 1e-9)) for k in MODULE_KEYS}
    remaining = total - sum(ints.values())
    order = sorted(MODULE_KEYS, key=lambda k: x[k] - ints[k], reverse=True)
    i = 0
    while remaining > 0 and i < len(order) * 4:
        k = order[i % len(order)]
        if ints[k] + 1 <= hi[k]:
            ints[k] += 1
            remaining -= 1
        i += 1
    return ints


def heuristic_module_weights(complexity: int) -> Dict[str, float]:
    """Raw allocation from JD complexity (1..5): senior/specialised roles lean on mandatory
    compliance; entry-level roles give more room to academics and extra-curricular activity."""
    c = (max(1, min(5, complexity)) - 1) / 4.0
    return {
        "mandatory_compliance": 40 + 20 * c,
        "academic_professional": 20 - 5 * c,
        "highlights": 10 + 5 * c,
        "extracurricular": 15 - 5 * c,
    }


def plan_from_analysis(analysis: JDAnalysis, enforce_ranges: bool = True) -> WeightPlan:
    raw = analysis.weights.model_dump()
    fitted = fit_to_ranges(raw, DEFAULT_RANGES if enforce_ranges else OPEN_RANGES)
    sub = analysis.compliance_weights.model_dump()
    if sum(sub.values()) <= 0:
        sub = ComplianceSubWeights().as_dict()
    rationale = analysis.rationale.strip() or (
        f"Complexity {analysis.complexity}/5 ({analysis.seniority}) — weights allocated automatically."
    )
    if enforce_ranges:
        rationale += " Weights were fitted to the default bands (compliance 40–60, achievements 15–20, " \
                     "highlights 10–15, extra-curricular 10–15) and normalised to 100."
    return WeightPlan(
        mode="autonomous",
        modules=ModuleWeights(**fitted),
        compliance=ComplianceSubWeights(**sub),
        rationale=rationale,
    )
