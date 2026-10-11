"""Skill / phrase matching and semantic similarity.

Matching cascade for a required skill:
  exact (after normalisation + alias table) -> token containment -> fuzzy ratio
  -> transformer embedding similarity (if sentence-transformers is installed)
  -> whole-word mention anywhere in the CV text.

``SemanticEngine`` gives a JD<->CV similarity: sentence-transformer embeddings when available,
otherwise a dependency-free log-TF cosine over content words.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

logger = logging.getLogger(__name__)

ALIASES: Dict[str, str] = {
    "js": "javascript", "javascript es6": "javascript", "ts": "typescript", "py": "python",
    "python3": "python", "python 3": "python", "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "cv": "computer vision", "ai": "artificial intelligence",
    "genai": "generative ai", "gen ai": "generative ai", "llms": "llm",
    "large language models": "llm", "large language model": "llm",
    "k8s": "kubernetes", "postgres": "postgresql", "psql": "postgresql", "mongo": "mongodb",
    "react.js": "react", "reactjs": "react", "vue.js": "vue", "vuejs": "vue", "angularjs": "angular",
    "node": "node.js", "nodejs": "node.js", "express.js": "express", "expressjs": "express",
    "aws": "amazon web services", "gcp": "google cloud", "google cloud platform": "google cloud",
    "ms azure": "azure", "microsoft azure": "azure", "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "tf": "tensorflow", "hf": "hugging face", "huggingface": "hugging face", "ci cd": "ci/cd", "cicd": "ci/cd",
    "ms excel": "excel", "microsoft excel": "excel", "powerbi": "power bi", "microsoft power bi": "power bi",
    "c sharp": "c#", "csharp": "c#", "cpp": "c++", "golang": "go", "it": "information technology",
    "dsa": "data structures and algorithms", "oop": "object-oriented programming",
    "object oriented programming": "object-oriented programming", "rest": "rest api", "restful api": "rest api",
    "restful apis": "rest api", "rest apis": "rest api", "ux": "ui/ux", "ui ux": "ui/ux",
}
_REVERSE_ALIASES: Dict[str, List[str]] = {}
for _k, _v in ALIASES.items():
    _REVERSE_ALIASES.setdefault(_v, []).append(_k)

# Vocabulary used by the offline analyser to spot skills in free text.
SKILL_VOCAB: List[str] = [
    "python", "java", "javascript", "typescript", "c++", "c#", "scala", "kotlin", "swift", "php", "ruby",
    "rust", "matlab", "sql", "nosql", "mysql", "postgresql", "mongodb", "oracle", "redis", "sqlite",
    "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi",
    "spring boot", ".net", "excel", "power bi", "tableau", "looker", "pandas", "numpy", "scipy",
    "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm", "machine learning",
    "deep learning", "natural language processing", "computer vision", "reinforcement learning",
    "data analysis", "data science", "statistics", "data visualization", "big data", "spark", "pyspark",
    "hadoop", "airflow", "kafka", "etl", "data engineering", "data warehousing", "mlops", "llm",
    "generative ai", "transformers", "hugging face", "langchain", "rag", "prompt engineering",
    "amazon web services", "azure", "google cloud", "docker", "kubernetes", "terraform", "ci/cd",
    "git", "linux", "rest api", "graphql", "microservices", "agile", "scrum", "jira",
    "project management", "stakeholder management", "communication", "problem solving",
    "accounting", "financial analysis", "financial modelling", "auditing", "taxation", "sap",
    "salesforce", "autocad", "solidworks", "spss", "sas", "stata", "selenium", "unit testing",
    "cybersecurity", "networking", "figma", "ui/ux", "digital marketing", "seo", "content writing",
    "object-oriented programming", "data structures and algorithms", "time series", "a/b testing",
    "opencv", "bert", "gpt", "mlflow", "dbt", "snowflake", "databricks", "bigquery",
]

_STOP = set(
    """a an the and or of to in for on with by at from as is are be been this that these those will
    you your we our their they it its into using use used over under such can may must should have has
    had not but also etc job role work team candidate candidates experience years year strong good
    ability skills skill knowledge required requirements responsibilities including include well within
    across per who what which while more than least plus preferred minimum etc""".split()
)


def norm_skill(s: object) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^\w+#./ -]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .-/")
    return ALIASES.get(s, s)


def _tokens(s: str) -> set:
    return {t.strip(".") for t in re.findall(r"[a-z0-9+#.]+", s.lower()) if t.strip(".")}


def text_mentions(term: str, text: str, text_lower: Optional[str] = None) -> bool:
    """Whole-word mention of ``term`` in ``text``. Very short terms ('R', 'Go') are case-sensitive."""
    t = (term or "").strip()
    if not t:
        return False
    if len(t) <= 2:
        return re.search(rf"(?<![\w+#]){re.escape(t)}(?![\w+#])", text) is not None
    tl = text_lower if text_lower is not None else text.lower()
    return re.search(rf"(?<![\w]){re.escape(t.lower())}(?![\w+#])", tl) is not None


def phrase_match(a: str, b: str, fuzzy: float = 0.86) -> bool:
    """Does phrase ``a`` (requirement) appear in phrase ``b`` (candidate)?"""
    na, nb = norm_skill(a), norm_skill(b)
    if not na or not nb:
        return False
    if na == nb or re.search(rf"(?<![\w]){re.escape(na)}(?![\w])", nb):
        return True
    ta, tb = _tokens(na), _tokens(nb)
    if ta and ta <= tb:
        return True
    return len(na) > 4 and difflib.SequenceMatcher(None, na, nb).ratio() >= fuzzy


class SkillMatch(NamedTuple):
    required: str
    matched: bool
    matched_as: str
    method: str  # exact | contains | fuzzy | semantic | text | ""


def match_skill(
    required: str,
    candidate_skills: Sequence[str],
    cv_text: str = "",
    engine: Optional["SemanticEngine"] = None,
    text_lower: Optional[str] = None,
) -> SkillMatch:
    r = norm_skill(required)
    if not r:
        return SkillMatch(required, False, "", "")
    cands = [(c, norm_skill(c)) for c in candidate_skills if str(c).strip()]
    for orig, c in cands:
        if c == r:
            return SkillMatch(required, True, orig, "exact")
    rt = _tokens(r)
    for orig, c in cands:
        ct = _tokens(c)
        if rt and rt <= ct and len(ct) <= len(rt) + 3:
            return SkillMatch(required, True, orig, "contains")
    if len(r) > 4:
        for orig, c in cands:
            if len(c) > 4 and difflib.SequenceMatcher(None, r, c).ratio() >= 0.88:
                return SkillMatch(required, True, orig, "fuzzy")
    if engine is not None and engine.has_embeddings and cands:
        best, score = engine.best_match(required, [o for o, _ in cands])
        if best is not None and score >= 0.82:
            return SkillMatch(required, True, best, "semantic")
    if cv_text:
        tl = text_lower if text_lower is not None else cv_text.lower()
        for form in {required.strip(), r, *(_REVERSE_ALIASES.get(r, []))}:
            if len(form) > 1 and text_mentions(form, cv_text, tl):
                return SkillMatch(required, True, form, "text")
    return SkillMatch(required, False, "", "")


def match_many(
    required: Iterable[str], candidate_skills: Sequence[str], cv_text: str = "",
    engine: Optional["SemanticEngine"] = None,
) -> List[SkillMatch]:
    tl = cv_text.lower()
    return [match_skill(r, candidate_skills, cv_text, engine, tl) for r in required if str(r).strip()]


def cert_match(required: str, candidate_certs: Sequence[str], cv_text: str = "") -> SkillMatch:
    """Certifications: >=80% of the required title's tokens inside one listed certification,
    or (for short names like 'PMP', 'CCNA') a whole-word mention in the CV."""
    rt = {t for t in _tokens(norm_skill(required)) if len(t) > 1 and t not in {"-", "and", "of", "the"}}
    if not rt:
        return SkillMatch(required, False, "", "")
    for c in candidate_certs:
        ct = _tokens(norm_skill(c))
        if len(rt & ct) / len(rt) >= 0.8:
            return SkillMatch(required, True, c, "exact")
    if len(rt) <= 2 and cv_text and text_mentions(required, cv_text):
        return SkillMatch(required, True, required, "text")
    if cv_text:
        tl = cv_text.lower()
        if len(required) > 6 and required.lower() in tl:
            return SkillMatch(required, True, required, "text")
    return SkillMatch(required, False, "", "")


# --------------------------------------------------------------------------------------------
# Semantic similarity
# --------------------------------------------------------------------------------------------
def _bow(text: str) -> Counter:
    toks = re.findall(r"[a-z][a-z0-9+#]{1,}", (text or "").lower())
    return Counter(t for t in toks if t not in _STOP)


def _cosine_counts(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    wa = {k: 1 + math.log(v) for k, v in a.items()}
    wb = {k: 1 + math.log(v) for k, v in b.items()}
    dot = sum(wa[k] * wb[k] for k in wa.keys() & wb.keys())
    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    return dot / (na * nb) if na and nb else 0.0


class SemanticEngine:
    """JD<->CV similarity and skill-embedding matcher.

    backend="sbert" tries sentence-transformers (all-MiniLM-L6-v2 by default) and silently falls
    back to the lexical cosine if the package or model is unavailable."""

    def __init__(self, backend: str = "lexical", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = "lexical"
        self.load_error = ""
        self._model = None
        self._cache: Dict[str, object] = {}
        if backend == "sbert":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(model_name)
                self.backend = "sbert"
            except Exception as exc:  # missing package, no network for model download, ...
                self.load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("sentence-transformers unavailable, using lexical similarity: %s", exc)

    @property
    def has_embeddings(self) -> bool:
        return self._model is not None

    @property
    def label(self) -> str:
        return "Transformer embeddings (MiniLM)" if self.has_embeddings else "Lexical cosine (built-in)"

    def _embed(self, texts: List[str]):
        import numpy as np  # sentence-transformers depends on numpy

        out = []
        missing = [t for t in texts if hashlib.md5(t.encode()).hexdigest() not in self._cache]
        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[hashlib.md5(t.encode()).hexdigest()] = v
        for t in texts:
            out.append(self._cache[hashlib.md5(t.encode()).hexdigest()])
        return np.asarray(out)

    def _doc_vector(self, text: str):
        import numpy as np

        words = text.split()
        chunks = [" ".join(words[i:i + 180]) for i in range(0, min(len(words), 180 * 12), 180)] or [""]
        v = self._embed(chunks).mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def doc_similarity(self, a: str, b: str) -> float:
        """Similarity in [0, 1]."""
        if not a or not b:
            return 0.0
        if self.has_embeddings:
            try:
                return float(max(0.0, min(1.0, self._doc_vector(a) @ self._doc_vector(b))))
            except Exception as exc:  # pragma: no cover
                logger.warning("Embedding similarity failed, falling back: %s", exc)
        return round(_cosine_counts(_bow(a), _bow(b)), 4)

    def best_match(self, query: str, options: Sequence[str]):
        if not self.has_embeddings or not options:
            return None, 0.0
        q = self._embed([query])[0]
        m = self._embed(list(options))
        sims = m @ q
        i = int(sims.argmax())
        return options[i], float(sims[i])
