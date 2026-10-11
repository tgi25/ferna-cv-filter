"""File -> clean text, plus light tokenisation helpers and PII redaction.

Every failure mode surfaces as ``ExtractionError`` with a message fit for the end user:
empty files, oversized files, legacy .doc, password-protected or corrupted PDFs, broken DOCX
archives, and image-only (scanned) documents that need OCR.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 15 * 1024 * 1024
MIN_TEXT_CHARS = 80
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")


class ExtractionError(Exception):
    """A file could not be turned into usable text. The message is user-facing."""


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------------------------
def detect_kind(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if data[:5] == b"%PDF-":
        return ".pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return ".docx"
        except zipfile.BadZipFile:
            pass
        raise ExtractionError("This ZIP-based file is not a Word document (.docx). Upload PDF or DOCX.")
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ExtractionError("Legacy Word .doc files are not supported — save as .docx or PDF and upload again.")
    if ext in SUPPORTED_EXTENSIONS:
        return ext
    raise ExtractionError(f"Unsupported file type '{ext or 'unknown'}'. Upload PDF, DOCX or TXT.")


# --------------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------------
def _looks_encrypted(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__} {exc}".lower()
    return "password" in msg or "encrypt" in msg


def _pdf_text(data: bytes) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    try:
        import pdfplumber  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        pdfplumber = None

    if pdfplumber is not None:
        try:
            pages: List[str] = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        pages.append(page.extract_text(x_tolerance=1.5, y_tolerance=3) or "")
                    except Exception as exc:  # one bad page should not sink the CV
                        warnings.append(f"Page {i + 1} could not be read ({type(exc).__name__}).")
            text = "\n\n".join(pages)
            if text.strip():
                return text, warnings
        except Exception as exc:
            if _looks_encrypted(exc):
                raise ExtractionError("PDF is password-protected — remove the password and upload again.") from exc
            warnings.append(f"Primary PDF parser failed ({type(exc).__name__}); used fallback parser.")

    # Fallback: pypdf (successor of PyPDF2), then PyPDF2 itself.
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise ExtractionError("No PDF parser installed (pdfplumber / pypdf).") from exc
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise ExtractionError("PDF is password-protected — remove the password and upload again.")
        parts = []
        for i, page in enumerate(reader.pages):
            try:
                parts.append(page.extract_text() or "")
            except Exception as exc:
                warnings.append(f"Page {i + 1} could not be read ({type(exc).__name__}).")
        return "\n\n".join(parts), warnings
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"PDF appears to be corrupted or malformed ({type(exc).__name__}).") from exc


def _docx_text(data: bytes) -> Tuple[str, List[str]]:
    try:
        import docx  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
        from docx.table import Table  # type: ignore
        from docx.text.paragraph import Paragraph  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("python-docx is not installed.") from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # BadZipFile, KeyError (missing part), XML errors ...
        raise ExtractionError(f"DOCX file is corrupted or not a valid Word document ({type(exc).__name__}).") from exc

    parts: List[str] = []
    try:
        # Headers often hold the candidate's name and contact line.
        for section in document.sections:
            for p in section.header.paragraphs:
                if p.text.strip():
                    parts.append(p.text)
        # Body in document order: paragraphs and tables interleaved.
        body = document.element.body
        for child in body.iterchildren():
            if child.tag == qn("w:p"):
                parts.append(Paragraph(child, document).text)
            elif child.tag == qn("w:tbl"):
                for row in Table(child, document).rows:
                    cells: List[str] = []
                    for cell in row.cells:
                        t = cell.text.strip()
                        if t and (not cells or cells[-1] != t):  # merged cells repeat their text
                            cells.append(t)
                    if cells:
                        parts.append(" | ".join(cells))
        # Text boxes (common in designer CV templates) are not reachable via paragraphs.
        for tb in body.iter(qn("w:txbxContent")):
            for p in tb.iter(qn("w:p")):
                t = "".join(x.text or "" for x in p.iter(qn("w:t")))
                if t.strip():
                    parts.append(t)
    except Exception as exc:
        raise ExtractionError(f"Could not read the DOCX contents ({type(exc).__name__}).") from exc
    return "\n".join(parts), []


def _txt(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("­", "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"[    ]{2,}", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(name: str, data: bytes) -> Tuple[str, List[str]]:
    """Return (clean_text, warnings) or raise ExtractionError."""
    if not data:
        raise ExtractionError("File is empty (0 bytes).")
    if len(data) > MAX_FILE_BYTES:
        raise ExtractionError(f"File is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    kind = detect_kind(name, data)
    if kind == ".pdf":
        raw, warnings = _pdf_text(data)
    elif kind == ".docx":
        raw, warnings = _docx_text(data)
    else:
        raw, warnings = _txt(data), []
    text = clean_text(raw)
    if len(text) < MIN_TEXT_CHARS:
        raise ExtractionError(
            "Little or no selectable text found — the file is probably a scanned image. "
            "Run OCR (e.g. ocrmypdf) or export a text-based PDF and upload again."
        )
    return text, warnings


# --------------------------------------------------------------------------------------------
# Tokenisation / budgeting
# --------------------------------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English)."""
    return max(1, len(text or "") // 4)


def truncate_for_llm(text: str, max_chars: int = 24000) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text.rfind("\n", 0, max_chars)
    cut = cut if cut > max_chars * 0.8 else max_chars
    return text[:cut] + "\n[... truncated ...]", True


# --------------------------------------------------------------------------------------------
# Contacts and PII redaction
# --------------------------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.I)
_PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")


def find_phones(text: str) -> List[str]:
    out = []
    for m in _PHONE_CANDIDATE.finditer(text or ""):
        s = m.group().strip()
        digits = re.sub(r"\D", "", s)
        if not 9 <= len(digits) <= 15:
            continue
        if not (s[0] in "+0(" or len(digits) >= 10):
            continue
        if re.fullmatch(r"(?:(?:19|20)\d{2}[\s.-]*){2,}", s):  # sequences of years
            continue
        out.append(s)
    return out


def find_contacts(text: str) -> Dict[str, str]:
    emails = EMAIL_RE.findall(text or "")
    phones = find_phones(text)
    return {"email": emails[0] if emails else "", "phone": phones[0] if phones else ""}


def redact_pii(text: str) -> str:
    """Remove e-mails, phone numbers and profile URLs before text leaves the machine."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    for phone in sorted(set(find_phones(text)), key=len, reverse=True):
        text = text.replace(phone, "[PHONE]")
    return text
