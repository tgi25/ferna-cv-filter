"""Ferna CV Filter — CV screening engine.

Pipeline: file -> text (extraction) -> structured profile + qualitative assessment
(LLM or offline heuristic) -> Stage 1 mandatory gate -> Stage 2 weighted score (/100).
"""

__version__ = "1.0.0"
APP_NAME = "Ferna CV Filter"
