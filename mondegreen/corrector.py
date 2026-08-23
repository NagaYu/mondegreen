"""ConstrainedCorrector: the hard-constraint ASR corrector, and its soft foil.

The central formalism
=====================

A correction is *only* ever the operation

    replace transcript[s:e] with g.surface,  for some g in Glossary

subject to the hard constraint

    normalized_phonetic_distance(read(transcript[s:e]), read(g)) <= tau(span)

Nothing else is expressible.  There is no free-form rewrite, no insertion of
novel text, no "improve the grammar while you're in there".  The set of legal
outputs for a span is computed *before* any model is consulted, and it is a
finite list.  That is the difference between this and prompting an LLM with a
glossary: the LLM's output space is every string, and its glossary is a
suggestion.

Where the language model fits
-----------------------------

An optional LM re-ranks *within* the already-legal candidate set, and grammar
constrained decoding (outlines / xgrammar) is used to make a generative model
respect that set token-by-token.  Note carefully: the guarantee does not depend
on outlines being installed.  Membership in the candidate set is enforced
structurally, by selecting an element of a Python list.  Constrained decoding is
how we let a *generative* model participate without being able to escape; it is
belt-and-braces, not the belt.

The soft foil
-------------

:class:`SoftPromptCorrector` implements the thing everyone actually does today --
paste the glossary into a prompt and ask a model to fix the transcript.  It is
here so that (D) has something honest to be measured against, and it shares the
metrics code with the constrained path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .gate import ConservativeGate, HeuristicGate
from .glossary import Glossary
from .hallucination import DEFAULT_PATTERNS, HallucinationFilter
from .index import PhoneticIndex
from .phonetics import (
    DEFAULT_CONFIG,
    PhoneticConfig,
    UNKNOWN,
    align,
    kana_to_phonemes,
    mora_count,
    phoneme_string,
)
from .reading import (
    char_kind, get_reader, is_boundary_blocked, is_common_word, is_proper_noun,
    span_reading_variants,
)
from .types import Candidate, Correction, CorrectionResult, GlossaryEntry, Token


@dataclass
class CorrectorConfig:
    """Every decision knob of the correction pass, in one serialisable object.

    Claim: SUPPORT -- ``tau`` and ``gate_threshold`` are the two numbers the
    LOW-DAMAGE curve is swept over, so they must be explicit and recorded.
    """

    #: The hard phonetic bound.  A replacement outside this is not merely
    #: discouraged, it is unrepresentable.
    tau: float = 0.28
    #: Stricter bound for very short spans, where a coincidental match is likely.
    tau_short: float = 0.16
    short_mora_limit: int = 3
    #: Far stricter bound for spans that are ordinary dictionary words.  At the
    #: default, a common word can only be replaced by a near-exact homophone --
    #: 「両氏誤り訂正」 -> 「量子誤り訂正」 (distance 0.000) passes, while
    #: 「稼働率」 -> 「加藤率」 (distance 0.060, glossary contains the surname 加藤)
    #: does not.  Requires a reader with POS information; see
    #: :func:`mondegreen.reading.is_common_word`.
    tau_common_word: float = 0.03
    #: Absolute cap on the raw weighted phonetic distance, independent of length.
    #: Roughly "one voicing slip plus a long-vowel slip, and no more".
    #:
    #: This is the binding constraint in practice, and its value is the most
    #: consequential number in the project.  Tuned on held-out *training* data
    #: (300 sentences, 300-term glossary), term recall is flat from 0.5 all the
    #: way to 3.2 -- loosening it buys no recall at all -- while the damage rate
    #: climbs monotonically.  The knee sits at 0.40, which is the last value with
    #: a measured damage rate of exactly zero on both the character and term
    #: metrics.
    #:
    #: The reason recall is nearly flat: real recoverable ASR errors on proper
    #: nouns are overwhelmingly *near-exact homophones* (進藤/新藤, 両氏/量子).
    #: The candidates that only become reachable well above this are almost all
    #: different words that happen to rhyme.
    max_raw_distance: float = 0.25
    #: Extra absolute budget per sqrt(mora).  A fourteen-phoneme loanword can
    #: absorb more total error than a three-mora surname without that error being
    #: any less plausible, so the absolute cap grows -- sub-linearly -- with span
    #: length.  A flat cap silently made long katakana product names
    #: uncorrectable (ミライドライバー -> ミライドライブ costs 0.70 raw over 8 morae).
    #: Tuned on held-out training data: 0.20 is the smallest slope that recovers
    #: long katakana product names, and raising it further buys no measurable
    #: recall.
    max_raw_slope: float = 0.20
    #: Cap on the relative difference in mora count between span and term.
    max_mora_ratio: float = 0.34
    #: Spans shorter than this are never corrected at all.
    min_span_morae: int = 2
    max_span_tokens: int = 8
    max_span_chars: int = 24
    top_k: int = 5
    gate_threshold: float = 0.5
    max_reading_variants: int = 8
    remove_hallucinations: bool = True
    #: Never touch a span that is already exactly a glossary surface form.
    protect_glossary_surfaces: bool = True
    #: Never touch an all-hiragana span (Japanese function words live there and
    #: they are phonetically short, i.e. maximum damage risk for minimum gain).
    protect_all_hiragana: bool = True
    #: Spans may not begin or end on a particle, auxiliary or suffix.
    protect_affix_boundaries: bool = True
    #: Reject replacements where one of span/term contains the other, since the
    #: only thing such an edit does is add or delete surrounding text.
    protect_containment: bool = True
    reader: str = "auto"
    #: Weight of the LM re-ranking term, when an LM backend is supplied.
    lm_weight: float = 0.35
    #: Emit rejected candidates too, so the Space can explain near misses.
    keep_rejected: bool = True

    def max_raw_for(self, morae: int) -> float:
        """Absolute distance budget for a span of ``morae`` morae.

        ``max_raw_distance + max_raw_slope * sqrt(morae)``.  Sub-linear because
        the number of independent slips ASR makes in a span grows with length but
        not proportionally.

        Claim: LOW-DAMAGE (bounds absolute error) + TERM-RECALL (without the
        length term, long loanwords become unreachable).
        """
        import math as _math

        return self.max_raw_distance + self.max_raw_slope * _math.sqrt(max(0, morae))

    def tau_for(self, morae: int, common_word: bool = False) -> float:
        """Effective hard threshold for one span.

        Three tiers, tightest first:

        * an ordinary dictionary word -> ``tau_common_word`` (near-exact only)
        * a span of at most ``short_mora_limit`` morae -> ``tau_short``
        * everything else -> ``tau``

        Short spans get a tighter bound because normalisation makes a single
        phoneme slip look proportionally large, and because a two-mora span has
        far more glossary neighbours by chance.  Common words get the tightest
        bound of all because they are, by definition, text the ASR probably got
        right.

        Claim: LOW-DAMAGE -- most accidental matches are either short or ordinary
        vocabulary.
        """
        if common_word:
            return min(self.tau_common_word, self.tau)
        return self.tau_short if morae <= self.short_mora_limit else self.tau

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return asdict(self)


class LMReranker(Protocol):
    """Anything that can score candidate replacements in context.

    Claim: TERM-RECALL -- when two glossary terms are equally close phonetically,
    only context can break the tie.
    """

    name: str

    def score_candidates(
        self, prefix: str, candidates: Sequence[str], suffix: str
    ) -> Sequence[float]:
        """Return one log-probability-ish score per candidate.

        Claim: TERM-RECALL -- ranking happens strictly inside the candidate set
        the hard constraint already produced; this method cannot add to it.
        """
        ...


def select_non_overlapping(items: Sequence[Tuple[int, int, float]]) -> List[int]:
    """Maximum-weight set of non-overlapping ``(start, end, score)`` intervals.

    Classic weighted interval scheduling: sort by end, binary-search the last
    interval that finishes at or before this one starts, and take the better of
    "use it" and "skip it".  Returns indices into the *input* order.

    Claim: LOW-DAMAGE -- applying two overlapping edits mangles the text, so the
    choice of which to keep is a safety decision, not a formatting one.
    """
    if not items:
        return []
    order = sorted(range(len(items)), key=lambda i: (items[i][1], items[i][0]))
    ends = [items[i][1] for i in order]
    n = len(order)
    best = [0.0] * (n + 1)
    take = [False] * (n + 1)
    prev = [0] * (n + 1)
    for k in range(1, n + 1):
        start, _, score = items[order[k - 1]]
        # rightmost j < k with ends[j-1] <= start
        lo, hi = 0, k - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if ends[mid] <= start:
                lo = mid + 1
            else:
                hi = mid
        p = lo
        with_it = best[p] + score
        if with_it > best[k - 1] + 1e-12:
            best[k], take[k], prev[k] = with_it, True, p
        else:
            best[k], take[k], prev[k] = best[k - 1], False, k - 1
    chosen: List[int] = []
    k = n
    while k > 0:
        if take[k]:
            chosen.append(order[k - 1])
            k = prev[k]
        else:
            k -= 1
    chosen.reverse()
    return chosen


def _is_containment(span_text: str, surface: str) -> bool:
    """Is the only difference between these two strings some surrounding text?

    ``新藤さん`` -> ``新藤`` deletes an honorific.  ``新藤`` -> ``新藤製作所``
    invents three characters nobody said.  Both are within the phonetic bound --
    a few cheap indels -- and both are pure damage.  Containment is the crisp
    test that catches the whole family.

    Claim: LOW-DAMAGE -- this guard plus the affix-boundary guard remove the two
    largest sources of breakage the corrector is capable of.
    """
    if span_text == surface:
        return True
    return surface in span_text or span_text in surface


@dataclass
class _SpanProposal:
    start: int
    end: int
    text: str
    candidate: Candidate
    runner_up: Optional[Candidate]
    features: Dict[str, float]
    threshold: float
    n_candidates: int
    reading_variants: int


class ConstrainedCorrector:
    """Replace mangled spans with glossary terms, under a hard phonetic bound.

    Claim: TERM-RECALL (condition D restores glossary terms baseline ASR lost),
    LOW-DAMAGE (the bound plus the gate keep breakage near zero),
    UNBOUNDED-VOCAB (cost per span is independent of glossary size),
    LOCAL-SPEED (pure Python, no network, no GPU required).
    """

    def __init__(
        self,
        glossary: Glossary,
        config: Optional[CorrectorConfig] = None,
        phonetic_config: Optional[PhoneticConfig] = None,
        gate: Optional[ConservativeGate] = None,
        index: Optional[PhoneticIndex] = None,
        hallucination_filter: Optional[HallucinationFilter] = None,
        lm: Optional[LMReranker] = None,
    ) -> None:
        """Build the corrector: index, gate, hallucination filter and reader.

                Claim: TERM-RECALL, LOW-DAMAGE, UNBOUNDED-VOCAB, LOCAL-SPEED.
                """
        self.config = config or CorrectorConfig()
        self.phonetic_config = phonetic_config or DEFAULT_CONFIG
        self.glossary = glossary
        self.index = index or PhoneticIndex(glossary, self.phonetic_config)
        self.gate = gate or ConservativeGate(threshold=self.config.gate_threshold)
        self.gate.threshold = self.config.gate_threshold
        self.hallucination_filter = hallucination_filter or HallucinationFilter(
            config=self.phonetic_config
        )
        self.lm = lm
        self.reader = get_reader(self.config.reader)

    # ------------------------------------------------------------------ public
    def correct(self, text: str) -> CorrectionResult:
        """Correct one transcript and return the result plus every receipt.

        Claim: TERM-RECALL + LOW-DAMAGE.
        """
        t0 = time.perf_counter()
        source = text
        removed: List[Tuple[int, int, str]] = []

        working = text
        if self.config.remove_hallucinations:
            working, hits = self.hallucination_filter.apply(working)
            removed = [(h.start, h.end, h.text) for h in hits]

        proposals, scanned = self._propose(working)
        accepted, rejected = self._gate_proposals(proposals)
        chosen = self._resolve_overlaps(accepted)
        out_text = self._apply(working, chosen)

        elapsed = time.perf_counter() - t0
        result = CorrectionResult(
            text=out_text,
            source=source,
            corrections=chosen,
            rejected=rejected if self.config.keep_rejected else [],
            removed_hallucinations=removed,
            stats={
                "chars": len(source),
                "spans_enumerated": scanned["enumerated"],
                "spans_screened_out": scanned["screened_out"],
                "spans_queried": scanned["queried"],
                "proposals": len(proposals),
                "accepted": len(chosen),
                "rejected": len(rejected),
                "seconds": elapsed,
                "chars_per_second": (len(source) / elapsed) if elapsed > 0 else 0.0,
                "glossary_terms": len(self.glossary),
                "indexed_readings": len(self.index),
                "gate": getattr(self.gate, "name", "unknown"),
                "gate_trained": bool(getattr(self.gate, "is_trained", False)),
                "reader": self.reader.name,
                "tau": self.config.tau,
                "gate_threshold": self.config.gate_threshold,
                "lm": getattr(self.lm, "name", None),
            },
        )
        return result

    def correct_many(self, texts: Sequence[str]) -> List[CorrectionResult]:
        """Batch helper; identical semantics, one result per input.

        Claim: LOCAL-SPEED -- amortises index construction across a whole
        transcript set, which is how the throughput number is measured.
        """
        return [self.correct(t) for t in texts]

    # ---------------------------------------------------------------- proposal
    def _propose(self, text: str) -> Tuple[List[_SpanProposal], Dict[str, int]]:
        """Enumerate spans and collect the best legal candidate for each.

        Claim: TERM-RECALL -- recall is bounded above by what span enumeration
        reaches, so the enumeration is deliberately generous and the *gate*, not
        the enumerator, does the refusing.
        """
        cfg = self.config
        tokens = self.reader.tokenize(text)
        # A real morphological analyser already resolved the reading *in context*,
        # so enumerating rendaku and on/kun alternatives would be eight redundant
        # index queries per span.  Only the dependency-free fallback reader needs
        # the beam.  Claim: LOCAL-SPEED.
        needs_variants = bool(getattr(self.reader, "needs_variants", True))
        max_variants = cfg.max_reading_variants if needs_variants else 1
        # Without a morphological analyser there is no way to tell 稼働 (ordinary
        # word, must not be rewritten) from 進藤 (proper noun, the whole point).
        # Rather than warn and do damage, degrade the *constraint*: with no POS,
        # a kanji span may only be replaced by a near-exact homophone. That still
        # recovers 進藤 -> 新藤 (distance 0.000) and 両氏誤り訂正 -> 量子誤り訂正
        # (0.000) while refusing 稼働 -> 加藤 (0.060). Katakana spans keep the
        # normal bound, because katakana is how ASR renders the unknown proper
        # nouns a glossary exists for.
        pos_available = bool(getattr(self.reader, "has_pos", False))
        counters = {"enumerated": 0, "screened_out": 0, "queried": 0}
        proposals: List[_SpanProposal] = []
        n = len(tokens)

        for i in range(n):
            if tokens[i].kind in ("punct", "space"):
                continue
            if cfg.protect_affix_boundaries and is_boundary_blocked(tokens[i]):
                continue
            for j in range(i + 1, min(i + cfg.max_span_tokens, n) + 1):
                tok_j = tokens[j - 1]
                if tok_j.kind in ("punct", "space"):
                    break
                if cfg.protect_affix_boundaries and is_boundary_blocked(tok_j):
                    continue
                start, end = tokens[i].start, tok_j.end
                if end - start > cfg.max_span_chars:
                    break
                span_text = text[start:end]
                if not span_text.strip():
                    continue
                counters["enumerated"] += 1
                span_tokens = tokens[i:j]
                # A span counts as ordinary vocabulary when every content token in
                # it is an ordinary word and none is a proper noun.
                #
                # One exception, and it matters: UniDic tags katakana loanwords as
                # 普通名詞, so 「ミライ + ドライバー」 would be classified as ordinary
                # vocabulary and protected -- when a multi-token katakana compound
                # is in fact the single most likely way Japanese ASR renders a
                # product name it has never seen.  A *single* katakana token
                # (パーセント, テーブル) really is ordinary and stays protected;
                # a katakana compound is not.
                katakana_compound = (
                    len(span_tokens) > 1
                    and all(t.kind == "katakana" for t in span_tokens)
                )
                if pos_available:
                    span_common = (
                        bool(span_tokens)
                        and not katakana_compound
                        and all(is_common_word(t) or is_boundary_blocked(t) for t in span_tokens)
                        and not any(is_proper_noun(t) for t in span_tokens)
                    )
                else:
                    span_common = any(t.kind == "kanji" for t in span_tokens)
                span_proper = any(is_proper_noun(t) for t in span_tokens)

                if cfg.protect_glossary_surfaces and self.index.has_surface(span_text):
                    continue
                if cfg.protect_all_hiragana and all(
                    char_kind(c) == "hiragana" for c in span_text
                ):
                    continue

                variants = span_reading_variants(
                    tokens, i, j, max_variants=max_variants,
                    per_token=3 if needs_variants else 1,
                    apply_rendaku=needs_variants,
                )
                if not variants:
                    continue

                best: Optional[Candidate] = None
                second: Optional[Candidate] = None
                n_cands = 0
                any_query = False
                containment_hit = False
                for variant in variants:
                    if containment_hit:
                        break
                    ph = kana_to_phonemes(variant, self.phonetic_config)
                    if not ph:
                        continue
                    morae = mora_count(ph)
                    if morae < cfg.min_span_morae:
                        continue
                    tau = cfg.tau_for(morae, span_common)
                    if not self.index.screen(ph, tau):
                        continue
                    any_query = True
                    cands = self.index.query(
                        ph, tau, top_k=cfg.top_k,
                        max_raw=cfg.max_raw_for(morae),
                        max_mora_ratio=cfg.max_mora_ratio,
                    )
                    n_cands = max(n_cands, len(cands))
                    for c in cands:
                        if c.entry.surface == span_text:
                            continue
                        if cfg.protect_containment and _is_containment(span_text, c.entry.surface):
                            # The span looks like a truncation or extension of a
                            # glossary term, so it is probably already essentially
                            # right.  Protect the *whole span*, not just this
                            # candidate: dropping only the contained candidate
                            # leaves a worse one to win, which is how
                            # 「非線形符号」 (a truncation of 非線形符号化) ends up
                            # rewritten to the unrelated 「非線形復号」.
                            containment_hit = True
                            break
                        if best is None:
                            best = c
                            continue
                        if c.entry.surface == best.entry.surface:
                            # The same term reached through a different reading
                            # variant.  Keep the tighter distance, but it is not a
                            # rival, so it must not become the runner-up.
                            if c.norm_distance < best.norm_distance:
                                best = c
                            continue
                        if c.norm_distance < best.norm_distance:
                            second = best
                            best = c
                        elif second is None or c.norm_distance < second.norm_distance:
                            second = c
                if any_query:
                    counters["queried"] += 1
                else:
                    counters["screened_out"] += 1
                if containment_hit or best is None:
                    continue
                # ``margin`` is "how much better is the winner than its nearest
                # rival"; a runner-up that is the same term is not a rival.
                if second is not None and second.entry.surface == best.entry.surface:
                    second = None

                morae = mora_count(best.span_phonemes)
                threshold = cfg.tau_for(morae, span_common)
                # Belt-and-braces: the index already guaranteed this, but the
                # invariant is important enough to re-assert at the boundary.
                if best.norm_distance > threshold + 1e-9:
                    continue
                if best.distance > cfg.max_raw_for(morae) + 1e-9:
                    continue

                features = self._features(
                    span_text=span_text,
                    best=best,
                    second=second,
                    n_candidates=n_cands,
                    reading_variants=len(variants),
                    text=text,
                    start=start,
                    end=end,
                    span_common=span_common,
                    span_proper=span_proper,
                )
                proposals.append(
                    _SpanProposal(
                        start=start,
                        end=end,
                        text=span_text,
                        candidate=best,
                        runner_up=second,
                        features=features,
                        threshold=threshold,
                        n_candidates=n_cands,
                        reading_variants=len(variants),
                    )
                )
        if self.lm is not None:
            self._apply_lm(text, proposals)
        return proposals, counters

    # ---------------------------------------------------------------- features
    def _features(
        self,
        span_text: str,
        best: Candidate,
        second: Optional[Candidate],
        n_candidates: int,
        reading_variants: int,
        text: str,
        start: int,
        end: int,
        span_common: bool = False,
        span_proper: bool = False,
    ) -> Dict[str, float]:
        """Build the gate's feature vector for one span.

        Every feature is cheap, bounded and explainable -- the Space shows them
        verbatim when a user asks why an edit was or was not made.

        Claim: LOW-DAMAGE.
        """
        span_ph = best.span_phonemes
        morae = mora_count(span_ph)
        term_morae = mora_count(best.term_phonemes)
        kinds = [char_kind(c) for c in span_text]
        unknown_ratio = (
            sum(1 for p in span_ph if p == UNKNOWN) / len(span_ph) if span_ph else 1.0
        )
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < len(text) else ""
        return {
            "norm_distance": best.norm_distance,
            "margin": (second.norm_distance - best.norm_distance) if second else 1.0,
            "exact_reading_match": 1.0 if best.norm_distance <= 1e-9 else 0.0,
            "span_morae": float(morae),
            "term_morae": float(term_morae),
            "mora_ratio": float(morae) / float(term_morae) if term_morae else 0.0,
            "span_chars": float(end - start),
            "n_candidates": float(n_candidates),
            "unknown_ratio": float(unknown_ratio),
            "reading_ambiguity": float(reading_variants),
            "log_term_weight": math.log1p(max(0.0, best.entry.weight)),
            "span_has_kanji": 1.0 if "kanji" in kinds else 0.0,
            "span_has_katakana": 1.0 if "katakana" in kinds else 0.0,
            "span_all_hiragana": 1.0 if kinds and all(k == "hiragana" for k in kinds) else 0.0,
            "span_is_glossary_surface": 1.0 if self.index.has_surface(span_text) else 0.0,
            "span_is_common_word": 1.0 if span_common else 0.0,
            "span_is_proper_noun": 1.0 if span_proper else 0.0,
            "boundary_left": 1.0 if (not left or char_kind(left) in ("punct", "space")) else 0.0,
            "boundary_right": 1.0 if (not right or char_kind(right) in ("punct", "space")) else 0.0,
            "lm_delta": 0.0,
        }

    def _apply_lm(self, text: str, proposals: List[_SpanProposal]) -> None:
        """Fill in ``lm_delta`` by scoring candidates in context.

        The LM only ever sees the already-legal candidate list; it cannot add to
        it.  ``lm_delta`` is the normalised advantage of the chosen candidate
        over leaving the span alone.

        Claim: TERM-RECALL -- context disambiguation, inside the constraint.
        """
        assert self.lm is not None
        for p in proposals:
            prefix = text[max(0, p.start - 64) : p.start]
            suffix = text[p.end : p.end + 64]
            options = [p.text, p.candidate.entry.surface]
            try:
                scores = list(self.lm.score_candidates(prefix, options, suffix))
            except Exception:  # pragma: no cover - backend specific
                continue
            if len(scores) != 2:
                continue
            delta = scores[1] - scores[0]
            p.features["lm_delta"] = float(max(-5.0, min(5.0, delta)))

    # -------------------------------------------------------------------- gate
    def _gate_proposals(
        self, proposals: Sequence[_SpanProposal]
    ) -> Tuple[List[Correction], List[Correction]]:
        """Turn proposals into accepted/rejected :class:`Correction` receipts.

        Claim: LOW-DAMAGE.
        """
        accepted: List[Correction] = []
        rejected: List[Correction] = []
        for p in proposals:
            prob, ok = self.gate.decide(p.features)
            corr = Correction(
                start=p.start,
                end=p.end,
                original=p.text,
                replacement=p.candidate.entry.surface,
                original_phonemes=p.candidate.span_phonemes,
                candidate_phonemes=p.candidate.term_phonemes,
                distance=p.candidate.distance,
                norm_distance=p.candidate.norm_distance,
                threshold=p.threshold,
                gate_prob=float(prob),
                margin=p.features.get("margin", 0.0),
                accepted=bool(ok),
                reason="gate-accept" if ok else f"gate-reject(p={prob:.3f}<{self.gate.threshold:.2f})",
                category=p.candidate.entry.category,
            )
            (accepted if ok else rejected).append(corr)
        return accepted, rejected

    # -------------------------------------------------------------- resolution
    @staticmethod
    def _score(c: Correction) -> float:
        """Ranking score for overlap resolution.

        Prefers confident, phonetically tight, longer corrections.

        Claim: LOW-DAMAGE -- when two spans compete, the safer one should win.
        """
        return (
            c.gate_prob
            + 0.5 * (1.0 - c.norm_distance)
            + 0.10 * math.log1p(max(0, c.end - c.start))
        )

    def _resolve_overlaps(self, corrections: Sequence[Correction]) -> List[Correction]:
        """Weighted interval scheduling: pick the best non-overlapping subset.

        Overlapping proposals are the normal case (「新藤製作所」 offers a
        two-character span and a five-character span), and applying both would
        corrupt the text.

        Claim: LOW-DAMAGE.
        """
        items = [(c.start, c.end, self._score(c)) for c in corrections]
        keep = select_non_overlapping(items)
        return [corrections[i] for i in keep]

    # ------------------------------------------------------------------- apply
    def _apply(self, text: str, corrections: Sequence[Correction]) -> str:
        """Splice accepted corrections into the transcript, right to left.

        Re-checks the hard constraint one final time immediately before mutating
        the string.  If this ever raises, a bug has smuggled an illegal edit past
        the index; failing loudly is strictly better than shipping the edit.

        Claim: LOW-DAMAGE -- the invariant is enforced at the point of no return,
        not merely at the point of proposal.
        """
        out = text
        for c in sorted(corrections, key=lambda c: -c.start):
            if c.norm_distance > c.threshold + 1e-9:
                raise AssertionError(
                    "hard phonetic constraint violated: "
                    f"{c.original!r} -> {c.replacement!r} "
                    f"d={c.norm_distance:.4f} > tau={c.threshold:.4f}"
                )
            budget = self.config.max_raw_for(mora_count(c.original_phonemes))
            if c.distance > budget + 1e-9:
                raise AssertionError(
                    "absolute phonetic bound violated: "
                    f"{c.original!r} -> {c.replacement!r} "
                    f"raw={c.distance:.4f} > max_raw={budget:.4f}"
                )
            out = out[: c.start] + c.replacement + out[c.end :]
        return out

    # ------------------------------------------------------------------ extras
    def explain(self, text: str) -> List[Dict[str, object]]:
        """Full per-span evidence dump, accepted and rejected alike.

        This is what the Gradio Space renders: original phoneme string, candidate,
        phonetic distance, threshold, gate probability, and the alignment that
        produced the distance.

        Claim: LOW-DAMAGE -- an auditable corrector is a correctable corrector.
        """
        result = self.correct(text)
        rows: List[Dict[str, object]] = []
        for c in [*result.corrections, *result.rejected]:
            _, ops = align(c.original_phonemes, c.candidate_phonemes, self.phonetic_config)
            d = c.to_dict()
            d["alignment"] = [
                {"op": op, "from": a, "to": b, "cost": round(cost, 3)} for op, a, b, cost in ops
            ]
            rows.append(d)
        rows.sort(key=lambda r: (not r["accepted"], r["start"]))
        return rows

    def proposals(self, text: str) -> List[Dict[str, object]]:
        """Every legal span replacement, *before* the gate decides anything.

        Each row carries the span offsets, the candidate, the phonetic evidence
        and the full gate feature vector.  The gate is trained by labelling
        exactly these rows against gold text, so exposing them is what keeps
        training and inference on the same features.

        Claim: LOW-DAMAGE -- the gate can only be calibrated on the same
        proposals it will later see.
        """
        working = text
        if self.config.remove_hallucinations:
            working, _ = self.hallucination_filter.apply(working)
        props, _ = self._propose(working)
        out: List[Dict[str, object]] = []
        for p in props:
            out.append({
                "start": p.start,
                "end": p.end,
                "text": p.text,
                "replacement": p.candidate.entry.surface,
                "category": p.candidate.entry.category,
                "norm_distance": p.candidate.norm_distance,
                "distance": p.candidate.distance,
                "threshold": p.threshold,
                "span_phonemes": p.candidate.span_phonemes,
                "term_phonemes": p.candidate.term_phonemes,
                "features": dict(p.features),
                "working_text": working,
            })
        return out

    def candidate_set(self, span_text: str) -> List[Candidate]:
        """The complete legal replacement set for a literal span of text.

        Exposed because "what *could* this have become?" is the question the hard
        constraint is designed to answer, and tests assert on it directly.

        Claim: LOW-DAMAGE.
        """
        tokens = self.reader.tokenize(span_text)
        variants = span_reading_variants(
            tokens, 0, len(tokens), max_variants=self.config.max_reading_variants
        )
        out: Dict[str, Candidate] = {}
        for v in variants:
            ph = kana_to_phonemes(v, self.phonetic_config)
            if not ph:
                continue
            morae = mora_count(ph)
            tau = self.config.tau_for(morae)
            for c in self.index.query(
                ph, tau, top_k=self.config.top_k,
                max_raw=self.config.max_raw_for(morae),
                max_mora_ratio=self.config.max_mora_ratio,
            ):
                prior = out.get(c.entry.surface)
                if prior is None or c.norm_distance < prior.norm_distance:
                    out[c.entry.surface] = c
        return sorted(out.values(), key=lambda c: c.norm_distance)


# --------------------------------------------------------------------------------------
# The soft foil
# --------------------------------------------------------------------------------------

@dataclass
class SoftPromptCorrector:
    """Glossary-in-the-prompt correction: the thing everybody does today.

    No phonetic constraint, no gate, no candidate set -- the model is handed the
    glossary as *context* and trusted to use it.  This is condition (C)'s
    mechanism and the "soft context injection" foil the brief asks for, and it
    exists so that (D)'s damage rate has a meaningful comparison.

    The failure mode it exhibits is the interesting part: a model given a
    glossary and a transcript will happily also fix grammar, drop hedges, merge
    sentences and normalise numbers -- all of which are damage under metric (3),
    and none of which the user asked for.

    Claim: LOW-DAMAGE -- (D) is only interesting if (C) is measured on the same
    ruler; UNBOUNDED-VOCAB -- this corrector is the one that hits a token ceiling.
    """

    glossary: Glossary
    backend: Optional[Callable[[str], str]] = None
    #: Prompt-token budget.  ``None`` means unlimited (a cloud LLM); 244 is
    #: Whisper's ``initial_prompt`` ceiling.
    token_budget: Optional[int] = None
    tokenizer_name: str = "heuristic"
    instruction: str = (
        "以下は日本語の音声認識結果です。用語集にある語が音的に近い誤りになっている場合のみ、"
        "その語に置き換えてください。それ以外は一切変更しないでください。\n"
    )

    def build_prompt(self, transcript: str) -> Tuple[str, int, int]:
        """Assemble the prompt and report ``(prompt, terms_included, terms_dropped)``.

        Terms are included in glossary order until the budget is exhausted, which
        is exactly how a real prompt-stuffing implementation behaves and exactly
        why it plateaus.

        Claim: UNBOUNDED-VOCAB -- this method is where the ceiling in the
        headline figure physically comes from.
        """
        from .baselines import count_tokens

        header = self.instruction + "用語集: "
        included: List[str] = []
        dropped = 0
        used = count_tokens(header, self.tokenizer_name)
        for e in self.glossary:
            piece = f"{e.surface}({e.reading})、"
            cost = count_tokens(piece, self.tokenizer_name)
            if self.token_budget is not None and used + cost > self.token_budget:
                dropped += 1
                continue
            included.append(piece)
            used += cost
        prompt = header + "".join(included) + "\n\n認識結果:\n" + transcript
        return prompt, len(included), dropped

    def correct(self, transcript: str) -> CorrectionResult:
        """Run the soft corrector.  Without a backend this is a no-op passthrough.

        Refusing to fabricate a result when no backend is configured is
        deliberate: a simulated (C) must be labelled as simulated, and that
        labelling happens in :mod:`mondegreen.baselines`, not here.

        Claim: SUPPORT.
        """
        prompt, included, dropped = self.build_prompt(transcript)
        if self.backend is None:
            return CorrectionResult(
                text=transcript,
                source=transcript,
                stats={
                    "mode": "soft-prompt",
                    "backend": None,
                    "terms_in_prompt": included,
                    "terms_dropped": dropped,
                    "token_budget": self.token_budget,
                    "note": "no backend configured; returned input unchanged",
                },
            )
        out = self.backend(prompt)
        return CorrectionResult(
            text=out,
            source=transcript,
            stats={
                "mode": "soft-prompt",
                "backend": getattr(self.backend, "name", "callable"),
                "terms_in_prompt": included,
                "terms_dropped": dropped,
                "token_budget": self.token_budget,
            },
        )
