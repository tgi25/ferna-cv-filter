# Ferna CV Filter

AI-assisted CV screening for employers: paste a job description, set the mandatory
requirements, bulk-upload CVs (PDF/DOCX), and get a ranked leaderboard with a
**Stage 1 pass/flag/disqualify gate** and a **Stage 2 score out of 100** — every mark
backed by a justification and evidence from the CV.

Ferna Software Solutions · MIT licence · Python 3.10+ · Streamlit

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, choose a provider in the sidebar and paste an API key
(or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, or copy `.streamlit/secrets.toml.example`
to `.streamlit/secrets.toml`). **Offline heuristic** mode needs no key and keeps every CV on
the machine — handy for demos; use an LLM for real screening.

Try it with the bundled fictional data:

```bash
pip install -r requirements-dev.txt
python samples/make_samples.py      # writes samples/job_description.txt and samples/cvs/*
```

Docker: `docker build -t ferna-cv-filter . && docker run -p 8501:8501 -e OPENAI_API_KEY=... ferna-cv-filter`

## Workflow

| Tab | What it does |
|---|---|
| **① Job Setup & Weight Configurator** | Upload/paste the JD → **Analyse JD** auto-fills the checklist (years, degree level + fields, certifications, core & nice-to-have skills, location, age only if the JD states one). Each rule is **Hard** (disqualify), **Soft** (flag) or **Off**. Choose **Autonomous** weights (allocated from JD complexity, fitted to the default bands) or a **Custom weight matrix** that must total 100, plus the relative weight of each criterion inside compliance. |
| **② CV Batch Upload & Processing** | Bulk upload, parallel analysis with a live progress table, per-file errors (corrupted, password-protected, scanned/no text, legacy .doc). Results are cached per file + JD + model, so re-runs are free. |
| **③ Candidate Leaderboard & Analytics** | KPI cards, filters (status, minimum score, search), a sortable table, a **deep-dive dialog** (gate checks, module bars with justifications and evidence, extracted profile, raw text), CSV/JSON export, score distribution, module averages, gate failure reasons and most-missed skills. Changing criteria or weights re-scores instantly without new API calls. |

## Scoring model

**Stage 1 — mandatory gatekeeper** (age, location/relocation, experience, qualification,
certifications, core skills):

* Hard rule fails → **Disqualified**
* Soft rule fails → **Flagged**
* Information missing from the CV → **Flagged** (never auto-disqualified)
* Degree in a different field when the JD says "or a related field" → **Flagged** for review

**Stage 2 — weighted score out of 100** (computed for everyone, so reviewers can see near misses):

| Module | Default band | How it is measured |
|---|---|---|
| Mandatory compliance | 40–60 | Σ shareₖ × fractionₖ over active criteria: skills coverage, years ÷ required (capped at 1), degree fit (1 / 0.7 related / 0.5 wrong field / 0.3 one–two levels short), certifications held ÷ needed, location (1 / 0.75 relocating / 0.4 unknown) |
| Academic & professional achievements | 15–20 | LLM 0–10 rubric score × weight ÷ 10 |
| Other highlights (teamwork, leadership, accomplishments) | 10–15 | LLM 0–10 rubric score × weight ÷ 10 |
| Extra-curricular (sports, societies, volunteering) | 10–15 | LLM 0–10 rubric score × weight ÷ 10 |

Experience is computed from dated roles with overlaps merged (no double counting); the
stated total is the fallback. Age is used only from an explicitly stated age or date of birth.

Autonomous weights: the LLM (or the rule-based estimator) proposes raw weights from JD
complexity/seniority; `fit_to_ranges` clamps them to the bands, redistributes the residual in
proportion to the remaining room, and rounds with the largest-remainder method so the total is
exactly 100.

## Architecture

```
app.py                     Streamlit dashboard (sidebar + 3 tabs + deep-dive dialog)
ferna_cv/
  schemas.py               Pydantic models: LLM outputs (lenient), criteria, weights, results
  extraction.py            PDF (pdfplumber → pypdf fallback) / DOCX (body, tables, headers, text boxes) / TXT,
                           cleaning, token budgeting, PII redaction, error classification
  llm.py                   OpenAI(-compatible) JSON mode + Anthropic forced tool use; schema-validated with one
                           automatic repair round-trip; friendly error messages
  heuristic.py             Offline rule-based CV + JD analyser (section detection, date ranges, degree keywords)
  matching.py              Skill normalisation + aliases, fuzzy/contains/semantic/text matching,
                           SemanticEngine (sentence-transformers or lexical cosine)
  dates.py                 Loose date parsing, merged experience intervals, age
  weights.py               Band fitting and autonomous allocation
  scoring.py               Stage 1 gate + Stage 2 weighted score, ranking
  pipeline.py              Parallel batch analysis with caching keys; never raises per file
  report.py                Leaderboard / CSV / JSON exports
samples/make_samples.py    Fictional JD + CVs (incl. corrupted and scanned PDFs)
tests/                     25 tests: engine, mocked OpenAI/Anthropic, Streamlit AppTest UI
```

One LLM call per CV returns both the structured profile and the three qualitative module
assessments (`CVAnalysis`). Scoring is deterministic Python on top of that, so the gate and the
compliance marks are fully auditable.

### Providers

* **OpenAI** and any OpenAI-compatible endpoint (Azure, OpenRouter, Ollama, vLLM, LM Studio via
  *Base URL*). Default model `gpt-4.1-mini` — change it in the sidebar to any model your key
  can use.
* **Anthropic Claude** via tool use. Default model `claude-sonnet-5-5`.
* Parameters a given model rejects (e.g. `temperature` on some reasoning models) are dropped
  automatically and the request retried.

### Transformer embeddings (optional)

`pip install sentence-transformers`, then pick *Transformer embeddings* in the sidebar. MiniLM
embeddings are used for semantic skill matching and JD↔CV similarity; if the package or the
model download is unavailable the app falls back to the built-in lexical similarity.

## Responsible use

* Treat output as **decision support**. A person should review every shortlist, and especially
  every Flagged candidate.
* The extraction prompt forbids using gender, ethnicity, religion, caste, marital status,
  nationality, disability, photos or age in the assessment, and tells the model to ignore any
  instructions embedded in a CV (prompt-injection defence).
* Age limits are **off by default**. They are unlawful for many roles in some jurisdictions
  (e.g. UK Equality Act 2010; US ADEA for applicants 40+); enable them only where permitted.
* CV text is sent to the chosen provider. Contact details are redacted before sending by
  default. Inform applicants and comply with applicable data-protection law (e.g. Sri Lanka's
  Personal Data Protection Act, GDPR).

## Testing

```bash
pip install -r requirements-dev.txt
python samples/make_samples.py
pytest -q
```

Covered: extraction of DOCX/PDF and every error path, date/experience/age logic, degree-level
detection, band fitting (500 random cases), skill matching, the full offline pipeline and
expected statuses, mocked OpenAI (JSON coercion, parameter fallback, schema repair) and
Anthropic (tool use, inlined schema), PII redaction round-trip, and Streamlit UI flows
(auto-fill, leaderboard metrics, weight validation, deep-dive dialog).

## Known limitations

* Scanned/image-only CVs need OCR first (e.g. `ocrmypdf`).
* Offline mode counts recognisable evidence and cannot judge relevance; unusual CV layouts
  extract less reliably than with an LLM.
* LLM scores vary slightly between runs and models; keep temperature at 0 and use one model
  per hiring round for comparability.
