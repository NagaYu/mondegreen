"""Removal of Whisper's canned hallucinations on silence and noise.

Whisper does not fail quietly.  Fed a stretch of silence, room tone or applause,
Japanese Whisper emits fluent, well-formed, entirely fabricated sentences -- and
it emits the *same* handful of them, because they are the boilerplate that
saturates its YouTube-caption training data.  「ご視聴ありがとうございました」 is
the canonical one.

This is a different failure from a mis-heard proper noun, so it gets a different
mechanism: a pattern list plus positional evidence, never a free-form rewrite.
Deleting text is the highest-variance thing a post-processor can do, so the
filter only fires when the pattern is *structurally* suspicious -- it occupies a
whole segment, it repeats, or it sits at the very end of a transcript where
Whisper's decoder runs off the end of the audio.  Everything it does is scored by
:func:`~mondegreen.metrics.hallucination_removal_rate`, which reports the
false-removal rate right next to the win.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .phonetics import DEFAULT_CONFIG, PhoneticConfig, kana_to_phonemes, normalized_distance

#: Canned phrases Japanese Whisper emits on non-speech audio.  Collected as a
#: labelled class by the PathologySet builder (see :mod:`mondegreen.harvest`);
#: this list is the seed set that ships with the package.
DEFAULT_PATTERNS: Tuple[str, ...] = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご清聴ありがとうございました",
    "最後までご視聴いただきありがとうございました",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "高評価とチャンネル登録をお願いします",
    "次回もお楽しみに",
    "また次の動画でお会いしましょう",
    "本日はご覧いただきありがとうございます",
    "字幕視聴ありがとうございました",
    "この動画は自動生成された字幕です",
    "エンディング",
    "おわり",
    "終わり",
    "スタッフのみなさん、ありがとうございました",
    "皆さんこんにちは",
    "はい",
    "うん",
    "ありがとうございました",
)

#: Patterns that are far too ordinary to delete on sight.  They only count as
#: hallucinations under the strictest evidence (whole segment AND repeated).
_WEAK_PATTERNS = frozenset({"はい", "うん", "おわり", "終わり", "エンディング", "ありがとうございました", "皆さんこんにちは"})

_SEGMENT_SPLIT = re.compile(r"[。．\.\n\r！!？\?]+")


@dataclass
class HallucinationHit:
    """One suspected hallucination, with the evidence that condemned it."""

    start: int
    end: int
    text: str
    pattern: str
    evidence: Tuple[str, ...]
    score: float

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "pattern": self.pattern,
            "evidence": list(self.evidence),
            "score": self.score,
        }


@dataclass
class HallucinationFilter:
    """Positional, evidence-gated removal of canned ASR hallucinations.

    Claim: LOW-DAMAGE -- metric (5) is only a win if the false-removal rate stays
    near zero, which is why every hit needs structural evidence and not just a
    string match.
    """

    patterns: Tuple[str, ...] = DEFAULT_PATTERNS
    config: PhoneticConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    #: Fire on a phonetic near-match, not just an exact one (「ご視聴ありがとうございました」
    #: vs 「ご清聴ありがとうございました」).  0.0 disables fuzzy matching.
    fuzzy_tau: float = 0.12
    #: Minimum evidence score required to delete.
    min_score: float = 1.0
    #: Require the phrase to be the whole segment before considering deletion.
    require_segment: bool = True

    def __post_init__(self) -> None:
        self._pattern_phonemes = {}
        for p in self.patterns:
            try:
                self._pattern_phonemes[p] = kana_to_phonemes(_to_reading(p), self.config)
            except Exception:  # pragma: no cover - defensive
                self._pattern_phonemes[p] = ()

    # ------------------------------------------------------------------ find
    def find(self, text: str, tail_window: int = 40) -> List[HallucinationHit]:
        """Locate suspected hallucinations without modifying anything.

        Evidence sources, each worth points:

        * the phrase is an entire punctuation-delimited segment (+1.0)
        * the phrase repeats in the transcript (+1.0)
        * the phrase sits in the final ``tail_window`` characters (+0.5)
        * the phrase is the entire transcript (+1.5)

        Claim: LOW-DAMAGE -- the score is what stops the filter from deleting a
        genuinely spoken 「ありがとうございました」.
        """
        hits: List[HallucinationHit] = []
        if not text.strip():
            return hits
        segments = _segments(text)
        n = len(text)
        for pattern in self.patterns:
            for start, end, matched in _occurrences(text, pattern, self._pattern_phonemes.get(pattern, ()), self.fuzzy_tau, self.config):
                evidence: List[str] = []
                score = 0.0
                seg = _containing_segment(segments, start, end)
                whole_segment = seg is not None and seg[2].strip(" 　、,・") == matched.strip(" 　、,・")
                if whole_segment:
                    evidence.append("whole-segment")
                    score += 1.0
                if text.count(pattern) > 1:
                    evidence.append("repeated")
                    score += 1.0
                if n - end <= tail_window:
                    evidence.append("transcript-tail")
                    score += 0.5
                if matched.strip() == text.strip():
                    evidence.append("entire-transcript")
                    score += 1.5
                if self.require_segment and not whole_segment:
                    continue
                if pattern in _WEAK_PATTERNS:
                    # Ordinary Japanese; needs two independent signals.
                    if len([e for e in evidence if e != "transcript-tail"]) < 2:
                        continue
                    score -= 0.5
                if score >= self.min_score:
                    hits.append(
                        HallucinationHit(start, end, matched, pattern, tuple(evidence), score)
                    )
        return _dedupe_overlaps(hits)

    # ----------------------------------------------------------------- apply
    def apply(self, text: str, tail_window: int = 40) -> Tuple[str, List[HallucinationHit]]:
        """Remove the hits :meth:`find` reports, returning ``(text, hits)``.

        Removal also swallows one adjacent sentence terminator so the result does
        not end up with a floating 「。」.

        Claim: LOW-DAMAGE + SUPPORT (metric 5).
        """
        hits = self.find(text, tail_window=tail_window)
        if not hits:
            return text, []
        out = text
        for hit in sorted(hits, key=lambda h: -h.start):
            end = hit.end
            # Swallow exactly one trailing terminator, so removing 「…ました」 from
            # 「…です。…ました。」 does not leave a stranded 「。」.
            if end < len(out) and out[end] in "。．.、,！!？? 　":
                end += 1
            out = out[: hit.start] + out[end:]
        return out.strip(), hits


def _to_reading(text: str) -> str:
    """Best-effort kana reading, tolerant of a missing reader backend.

        Claim: SUPPORT.
        """
    from .reading import get_reader, text_to_reading

    try:
        return text_to_reading(text, get_reader())
    except Exception:  # pragma: no cover
        return text


def _segments(text: str) -> List[Tuple[int, int, str]]:
    """Punctuation-delimited segments as ``(start, end, text)``.

    Claim: SUPPORT -- "is this phrase the whole utterance?" is the strongest
    single piece of evidence that it was hallucinated.
    """
    out: List[Tuple[int, int, str]] = []
    pos = 0
    for m in _SEGMENT_SPLIT.finditer(text):
        if m.start() > pos:
            out.append((pos, m.start(), text[pos : m.start()]))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text), text[pos:]))
    return out


def _containing_segment(
    segments: Sequence[Tuple[int, int, str]], start: int, end: int
) -> Optional[Tuple[int, int, str]]:
    """The punctuation-delimited segment containing ``[start, end)``, if any.

        Claim: LOW-DAMAGE -- whole-segment occupancy is the strongest evidence a
        phrase was hallucinated rather than spoken.
        """
    for seg in segments:
        if seg[0] <= start and end <= seg[1]:
            return seg
    return None


def _occurrences(
    text: str,
    pattern: str,
    pattern_ph: Sequence[str],
    fuzzy_tau: float,
    config: PhoneticConfig,
) -> List[Tuple[int, int, str]]:
    """Exact occurrences, plus phonetic near-misses of the same length band.

    Claim: SUPPORT -- Whisper varies its boilerplate slightly between runs
    (ご視聴 / ご清聴), and an exact-match-only filter would miss half of it.
    """
    out: List[Tuple[int, int, str]] = []
    start = 0
    while True:
        i = text.find(pattern, start)
        if i < 0:
            break
        out.append((i, i + len(pattern), pattern))
        start = i + len(pattern)
    if fuzzy_tau <= 0 or not pattern_ph:
        return out
    # Fuzzy pass over whole segments only -- scanning every substring would be
    # both slow and a great way to delete real text.
    for s, e, seg in _segments(text):
        cand = seg.strip()
        if not cand or any(s <= o[0] and o[1] <= e for o in out):
            continue
        if not (0.6 * len(pattern) <= len(cand) <= 1.6 * len(pattern)):
            continue
        try:
            cand_ph = kana_to_phonemes(_to_reading(cand), config)
        except Exception:  # pragma: no cover
            continue
        if not cand_ph:
            continue
        if normalized_distance(cand_ph, tuple(pattern_ph), config) <= fuzzy_tau:
            offset = seg.find(cand)
            out.append((s + offset, s + offset + len(cand), cand))
    return out


def _dedupe_overlaps(hits: List[HallucinationHit]) -> List[HallucinationHit]:
    """Keep the highest-scoring, longest hit among overlapping candidates.

    Claim: SUPPORT.
    """
    ordered = sorted(hits, key=lambda h: (-h.score, -(h.end - h.start), h.start))
    kept: List[HallucinationHit] = []
    for h in ordered:
        if any(not (h.end <= k.start or h.start >= k.end) for k in kept):
            continue
        kept.append(h)
    return sorted(kept, key=lambda h: h.start)
