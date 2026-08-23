"""Shared dataclasses for Mondegreen.

Every public function in this package carries a ``Claim:`` line in its docstring
naming which experimental claim it substantiates:

* ``TERM-RECALL``     -- glossary terms are restored in the transcript.
* ``LOW-DAMAGE``      -- spans that were already correct are left alone.
* ``UNBOUNDED-VOCAB`` -- performance keeps improving as the glossary grows past
  any prompt-token ceiling.
* ``LOCAL-SPEED``     -- the whole thing runs on a laptop, no round trip.
* ``SUPPORT``         -- plumbing that makes one of the above measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A phoneme sequence is a tuple of mora-phoneme symbols, e.g. ("t","o","o","ky","o","o").
Phonemes = Tuple[str, ...]

CLAIMS = ("TERM-RECALL", "LOW-DAMAGE", "UNBOUNDED-VOCAB", "LOCAL-SPEED", "SUPPORT")


@dataclass(frozen=True)
class GlossaryEntry:
    """One private-vocabulary item: a surface form plus its reading(s).

    Claim: TERM-RECALL -- the reading is what makes a term recoverable from a
    phonetically mangled ASR span; the surface alone is not enough.
    """

    surface: str
    reading: str                       # katakana or hiragana reading of ``surface``
    aliases: Tuple[str, ...] = ()      # extra readings that should also match
    category: str = ""                 # person / product / jargon / unit ...
    weight: float = 1.0                # prior; higher = more expected in this domain
    notes: str = ""

    def all_readings(self) -> Tuple[str, ...]:
        """Reading plus aliases, de-duplicated, order preserved.

        Claim: TERM-RECALL.
        """
        seen: List[str] = []
        for r in (self.reading, *self.aliases):
            if r and r not in seen:
                seen.append(r)
        return tuple(seen)


@dataclass
class Token:
    """A unit of the ASR transcript that a correction span can start or end on.

    Claim: SUPPORT -- span enumeration granularity directly bounds both recall
    (TERM-RECALL) and the number of chances to break something (LOW-DAMAGE).
    """

    surface: str
    start: int                      # character offset into the source transcript
    end: int
    readings: Tuple[str, ...] = ()  # candidate kana readings, best first
    kind: str = "other"             # kanji / hiragana / katakana / latin / digit / punct / space
    pos: str = ""                   # coarse part of speech, when the reader knows one
    pos2: str = ""                  # sub-category: 固有名詞 / 普通名詞 / 格助詞 ...


@dataclass
class Candidate:
    """A glossary term proposed as the replacement for one span."""

    entry: GlossaryEntry
    reading: str
    term_phonemes: Phonemes
    span_phonemes: Phonemes
    distance: float             # raw weighted phonetic edit distance
    norm_distance: float        # distance / max(len(a), len(b))  in [0, 1]
    alignment: Tuple[Tuple[str, str, str, float], ...] = ()  # (op, a, b, cost)


@dataclass
class Correction:
    """One accepted (or rejected) span replacement, with its full justification.

    The Gradio Space renders exactly these fields as the evidence panel:
    original phoneme string / candidate / phonetic distance.

    Claim: TERM-RECALL + LOW-DAMAGE -- a correction is only ever emitted with a
    receipt showing it cleared the hard phonetic bound and the gate.
    """

    start: int
    end: int
    original: str
    replacement: str
    original_phonemes: Phonemes
    candidate_phonemes: Phonemes
    distance: float
    norm_distance: float
    threshold: float
    gate_prob: float
    margin: float                 # norm_distance(runner-up) - norm_distance(best)
    accepted: bool
    reason: str = ""
    category: str = ""
    alignment: Tuple[Tuple[str, str, str, float], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view used by the CLI, the Space and the benchmark dumps.

        Claim: SUPPORT.
        """
        d = asdict(self)
        d["original_phonemes"] = " ".join(self.original_phonemes)
        d["candidate_phonemes"] = " ".join(self.candidate_phonemes)
        d["alignment"] = [list(x) for x in self.alignment]
        return d


@dataclass
class CorrectionResult:
    """The output of a corrector run over one transcript."""

    text: str                                    # corrected transcript
    source: str                                  # the transcript we were given
    corrections: List[Correction] = field(default_factory=list)
    rejected: List[Correction] = field(default_factory=list)
    removed_hallucinations: List[Tuple[int, int, str]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """True when the corrector touched anything at all.

        Claim: LOW-DAMAGE -- the empty-glossary harmlessness test asserts this is
        False (see tests/test_harmlessness.py).
        """
        return self.text != self.source

    def to_dict(self) -> Dict[str, Any]:
        """Claim: SUPPORT."""
        return {
            "text": self.text,
            "source": self.source,
            "corrections": [c.to_dict() for c in self.corrections],
            "rejected": [c.to_dict() for c in self.rejected],
            "removed_hallucinations": [list(x) for x in self.removed_hallucinations],
            "stats": self.stats,
        }


@dataclass
class ErrorPair:
    """One (ASR hypothesis, gold text) pair produced by the ErrorHarvester.

    ``gold`` is the source text that was fed to TTS, so it is exact by
    construction -- no LLM ever judges correctness anywhere in this project.

    Claim: SUPPORT -- this is the ground truth all four metrics are computed
    against.
    """

    id: str
    gold: str
    hypothesis: str
    glossary_terms: Tuple[str, ...] = ()   # surfaces that occur in ``gold``
    error_types: Tuple[str, ...] = ()      # PathologySet labels
    speaker: str = ""
    speed: float = 1.0
    snr_db: Optional[float] = None
    room: str = "close"                    # close / far / reverb
    asr_model: str = ""
    source_corpus: str = ""
    source_license: str = ""
    split: str = "train"
    provenance: str = "measured"           # measured (real TTS+ASR) or simulated

    def to_dict(self) -> Dict[str, Any]:
        """Claim: SUPPORT."""
        return asdict(self)


@dataclass
class SpanDecision:
    """Feature vector + label for one gate training example.

    Claim: LOW-DAMAGE -- the gate is the component that decides *not* to act, and
    it can only be calibrated from labelled span decisions like these.
    """

    features: Dict[str, float]
    label: int          # 1 = replacing this span moves us toward the gold text
    span_text: str = ""
    candidate: str = ""
    gold_span: str = ""
    pair_id: str = ""
