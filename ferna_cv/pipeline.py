"""Batch processing: file bytes -> CVRecord (text + structured analysis).

Analysis is the expensive step (one LLM call per CV) and is cached per
(file content, job description, provider/model). Scoring is cheap and is recomputed whenever
the employer changes criteria or weights.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from . import heuristic
from .extraction import (
    ExtractionError,
    extract_text,
    file_hash,
    find_contacts,
    redact_pii,
    text_hash,
    truncate_for_llm,
)
from .llm import LLMClient, LLMConfig, LLMError, analyze_cv_llm
from .schemas import CVRecord

logger = logging.getLogger(__name__)


class ScreeningConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    redact_pii: bool = True
    max_chars: int = 24000


def cache_key(fhash: str, jd_text: str, cfg: LLMConfig) -> str:
    raw = f"{fhash}|{text_hash(jd_text)}|{cfg.provider}|{cfg.model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def analyze_file(
    name: str, data: bytes, jd_text: str, cfg: ScreeningConfig, client: Optional[LLMClient] = None
) -> CVRecord:
    """Never raises: every failure is captured in ``record.error`` so one bad file cannot stop a batch."""
    t0 = time.perf_counter()
    fh = file_hash(data) if data else "empty"
    rec = CVRecord(file_name=name, file_hash=fh, cache_key=cache_key(fh, jd_text, cfg.llm),
                   jd_hash=text_hash(jd_text), source=cfg.llm.label)
    try:
        text, warnings = extract_text(name, data)
        rec.text = text
        rec.warnings.extend(warnings)
        local = heuristic.analyze_cv(text, jd_text, file_name=name)  # cheap; used as fallback values

        if cfg.llm.provider == "heuristic":
            analysis = local
        else:
            contacts = find_contacts(text)
            llm_text = redact_pii(text) if cfg.redact_pii else text
            llm_text, truncated = truncate_for_llm(llm_text, cfg.max_chars)
            if truncated:
                rec.warnings.append(f"CV text truncated to {cfg.max_chars:,} characters for the model.")
            if client is None:
                client = LLMClient(cfg.llm)
            analysis = analyze_cv_llm(client, llm_text, jd_text)
            prof = analysis.profile
            # Contacts were redacted before sending, so restore them from local extraction.
            if not prof.email or "[EMAIL]" in prof.email:
                prof.email = contacts["email"]
            if not prof.phone or "[PHONE]" in prof.phone:
                prof.phone = contacts["phone"]
            if not prof.full_name:
                prof.full_name = local.profile.full_name
        rec.analysis = analysis
    except ExtractionError as exc:
        rec.error = str(exc)
    except LLMError as exc:
        rec.error = f"AI analysis failed: {exc}"
    except Exception as exc:  # defensive: never let one CV crash the batch
        logger.exception("Unexpected failure on %s", name)
        rec.error = f"Unexpected error: {type(exc).__name__}: {exc}"
    rec.seconds = round(time.perf_counter() - t0, 2)
    return rec


def analyze_batch(
    files: Sequence[Tuple[str, bytes]],
    jd_text: str,
    cfg: ScreeningConfig,
    max_workers: int = 4,
    on_done: Optional[Callable[[CVRecord], None]] = None,
) -> List[CVRecord]:
    """Analyse files in parallel. ``on_done`` is called in the caller's thread (safe for UI updates)."""
    client = None if cfg.llm.provider == "heuristic" else LLMClient(cfg.llm)  # validates key/model up front
    workers = 1 if cfg.llm.provider == "heuristic" else max(1, max_workers)
    results: List[CVRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(analyze_file, n, d, jd_text, cfg, client) for n, d in files]
        for fut in as_completed(futures):
            rec = fut.result()
            results.append(rec)
            if on_done:
                on_done(rec)
    return results
