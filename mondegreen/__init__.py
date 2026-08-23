"""Mondegreen -- a private glossary, compiled into a hard phonetic constraint.

Whisper does not know your colleagues' names, your product names or your team's
jargon, and no amount of prompting will make it learn 10,000 of them.  Mondegreen
does the correction *afterwards*, locally, as a constrained edit: a span of the
transcript may only be replaced by a glossary term that it actually sounds like.

    >>> from mondegreen import ConstrainedCorrector, load_glossary
    >>> corrector = ConstrainedCorrector(load_glossary("terms.csv"))
    >>> corrector.correct("進藤さんが両氏誤り訂正の話をしました。").text
    '新藤さんが量子誤り訂正の話をしました。'

Four claims, and every public docstring in the package says which one it serves:

``TERM-RECALL``      glossary terms come back
``LOW-DAMAGE``       text that was already right stays right
``UNBOUNDED-VOCAB``  performance keeps improving past any prompt-token ceiling
``LOCAL-SPEED``      it runs on a laptop, and the audio never leaves it
"""

from __future__ import annotations

__version__ = "0.1.0"

from .types import (
    CLAIMS,
    Candidate,
    Correction,
    CorrectionResult,
    ErrorPair,
    GlossaryEntry,
    SpanDecision,
    Token,
)
from .phonetics import (
    DEFAULT_CONFIG,
    PhoneticConfig,
    align,
    kana_to_phonemes,
    normalized_distance,
    phonetic_distance,
    phoneme_string,
)
from .glossary import Glossary, load_glossary, loads_glossary, save_glossary
from .index import PhoneticIndex
from .gate import ConservativeGate, HeuristicGate, sweep_thresholds, pick_threshold
from .hallucination import DEFAULT_PATTERNS, HallucinationFilter
from .corrector import ConstrainedCorrector, CorrectorConfig, SoftPromptCorrector

__all__ = [
    "__version__",
    "CLAIMS",
    "Candidate",
    "Correction",
    "CorrectionResult",
    "ErrorPair",
    "GlossaryEntry",
    "SpanDecision",
    "Token",
    "DEFAULT_CONFIG",
    "PhoneticConfig",
    "align",
    "kana_to_phonemes",
    "normalized_distance",
    "phonetic_distance",
    "phoneme_string",
    "Glossary",
    "load_glossary",
    "loads_glossary",
    "save_glossary",
    "PhoneticIndex",
    "ConservativeGate",
    "HeuristicGate",
    "sweep_thresholds",
    "pick_threshold",
    "DEFAULT_PATTERNS",
    "HallucinationFilter",
    "ConstrainedCorrector",
    "CorrectorConfig",
    "SoftPromptCorrector",
]
