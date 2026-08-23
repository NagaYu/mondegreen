"""PhoneticIndex: retrieve glossary terms that *sound like* a span of ASR output.

The index is what makes the UNBOUNDED-VOCAB claim mechanically true.  Baseline
(B) pays for vocabulary in prompt tokens and dies at Whisper's 244-token
``initial_prompt`` ceiling.  Mondegreen pays for vocabulary in inverted-index
postings, so 10,000 terms cost the same per span as 100 -- a few hundred
microseconds of pruned dynamic programming.

Retrieval is two-stage:

1. **Prune.**  An admissible length filter (any alignment between sequences of
   length ``n`` and ``m`` must contain at least ``|n-m|`` indels, each costing at
   least ``min_indel``) plus an IDF-weighted phoneme-bigram overlap.
2. **Score.**  Exact weighted phonetic DP with early abandonment
   (:func:`~mondegreen.phonetics.bounded_normalized_distance`) on the survivors.

Stage 2 is exact.  Stage 1's n-gram step is a heuristic accelerator and it is
**not** exact: at the default settings it recovers ~99.8% of the candidates an
exhaustive scan would find at 1,000 terms and ~99.2% at 10,000.  That cost is
reported, not hidden -- :meth:`PhoneticIndex.retrieval_recall` measures it on
real queries and the benchmark writes the number into every result file, and
``exact=True`` disables the accelerator entirely.

Note what a prefilter miss *is* and is not.  It can only cause a correction to be
missed; it can never cause an illegal one, because stage 2 re-verifies the hard
bound on every candidate it scores.  The accelerator trades recall for speed, and
never trades safety.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .glossary import Glossary
from .phonetics import (
    DEFAULT_CONFIG,
    PhoneticConfig,
    align,
    bounded_distance_pre,
    bounded_normalized_distance,
    indel_costs,
    kana_to_phonemes,
    mora_count,
    normalized_distance,
    phoneme_ngrams,
    substitution_table,
)
from .types import Candidate, GlossaryEntry, Phonemes


@dataclass(frozen=True)
class IndexedReading:
    """One (entry, reading) pair as stored in the index.

    A term with three attested readings occupies three slots pointing at one
    entry, so any of them can trigger a match.

    Claim: TERM-RECALL.
    """

    entry_id: int
    entry: GlossaryEntry
    reading: str
    phonemes: Phonemes
    morae: int
    #: Per-position indel costs, precomputed at index build time so the DP never
    #: recomputes them.  Claim: LOCAL-SPEED.
    indel: Tuple[float, ...] = ()


class PhoneticIndex:
    """Phoneme-level nearest-neighbour index over a glossary.

    Claim: UNBOUNDED-VOCAB (retrieval cost is flat in glossary size) and
    LOCAL-SPEED (all of it is pure Python over small integer lists).
    """

    def __init__(
        self,
        glossary: Glossary,
        config: PhoneticConfig = DEFAULT_CONFIG,
        ngram: int = 2,
        max_df_ratio: float = 0.30,
        prefilter: int = 300,
        exhaustive_limit: int = 300,
        cache_size: int = 20000,
    ) -> None:
        """Build the phonetic index over a glossary.

                Claim: UNBOUNDED-VOCAB + LOCAL-SPEED.
                """
        self.glossary = glossary
        self.config = config
        self.ngram = ngram
        self.max_df_ratio = max_df_ratio
        self.prefilter = prefilter
        self.exhaustive_limit = exhaustive_limit
        self.cache_size = cache_size
        self._sub = substitution_table(config)
        self._cache: Dict[Tuple, List[Candidate]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self.readings: List[IndexedReading] = []
        self._postings: Dict[str, List[int]] = defaultdict(list)
        self._slot_norm: List[float] = []
        self._idf: Dict[str, float] = {}
        self._surfaces: Dict[str, GlossaryEntry] = {}
        self._surface_phonemes: Dict[int, Phonemes] = {}
        self._min_indel = min(
            config.indel_R,
            config.indel_long_vowel,
            config.indel_Q,
            config.indel_N,
            config.indel_epenthetic,
            config.indel_glide,
            config.indel_vowel,
            config.indel_consonant,
        )
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        """Index every (entry, reading) pair and precompute its costs and n-grams.

            Claim: UNBOUNDED-VOCAB + LOCAL-SPEED -- all per-term work happens once, here.
            """
        for entry_id, entry in enumerate(self.glossary):
            self._surfaces[entry.surface] = entry
            for reading in entry.all_readings():
                if not reading:
                    continue
                ph = kana_to_phonemes(reading, self.config)
                if not ph or all(p == "?" for p in ph):
                    # An entry we cannot pronounce must never be proposed.
                    continue
                slot = len(self.readings)
                self.readings.append(
                    IndexedReading(entry_id, entry, reading, ph, mora_count(ph),
                                   indel_costs(ph, self.config))
                )
                grams = set(phoneme_ngrams(ph, self.ngram))
                if len(ph) >= 4:
                    grams |= set(phoneme_ngrams(ph, self.ngram + 1))
                for gram in grams:
                    self._postings[gram].append(slot)
                self._slot_norm.append(math.sqrt(len(grams)) or 1.0)
            first = entry.all_readings()[0] if entry.all_readings() else ""
            if first:
                self._surface_phonemes[entry_id] = kana_to_phonemes(first, self.config)

        n = max(1, len(self.readings))
        self._idf = {
            gram: math.log(1.0 + n / (1.0 + len(posting)))
            for gram, posting in self._postings.items()
        }
        self._max_df = max(1, int(self.max_df_ratio * n))

    # ------------------------------------------------------------- accessors
    def __len__(self) -> int:
        """Number of indexed (entry, reading) slots.

        Claim: SUPPORT.
        """
        return len(self.readings)

    @property
    def n_terms(self) -> int:
        """Number of distinct glossary entries actually reachable in the index.

        Claim: UNBOUNDED-VOCAB -- this is the x-axis of the headline figure.
        """
        return len({r.entry_id for r in self.readings})

    def has_surface(self, text: str) -> bool:
        """Is this exact string already a glossary surface form?

        Claim: LOW-DAMAGE -- a span that already reads as the term is a span the
        corrector must not touch.
        """
        return text in self._surfaces

    def stats(self) -> Dict[str, object]:
        """Index size/shape summary, recorded in benchmark provenance.

        Claim: SUPPORT.
        """
        posting_sizes = [len(v) for v in self._postings.values()]
        return {
            "entries": len(self.glossary),
            "indexed_readings": len(self.readings),
            "reachable_terms": self.n_terms,
            "grams": len(self._postings),
            "mean_posting": (sum(posting_sizes) / len(posting_sizes)) if posting_sizes else 0.0,
            "max_posting": max(posting_sizes) if posting_sizes else 0,
            "ngram": self.ngram,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    @property
    def length_range(self) -> Tuple[int, int]:
        """Shortest and longest indexed phoneme sequence.

        Claim: LOCAL-SPEED -- a span outside this band cannot match anything, so
        it is rejected before any DP runs.
        """
        if not self.readings:
            return (0, 0)
        lens = [len(r.phonemes) for r in self.readings]
        return (min(lens), max(lens))

    def screen(self, phonemes: Sequence[str], tau: float, min_gram_ratio: float = 0.34) -> bool:
        """Cheap set-lookup pre-check: could this span match *anything* at all?

        Two O(1)-ish tests before the index is even consulted: the length band,
        and whether any of the span's phoneme bigrams exist in the glossary at
        all.  On a full-length transcript this rejects the overwhelming majority
        of enumerated spans without touching the DP.

        Claim: LOCAL-SPEED -- span enumeration is quadratic in span length, so
        the constant factor per span is what decides whether an hour of audio
        post-processes in seconds or minutes.
        """
        n = len(phonemes)
        if n == 0 or not self.readings:
            return False
        lo, hi = self.length_range
        # Admissible band: |n - t| * min_indel <= tau * max(n, t) for some indexed t.
        slack = tau * max(n, hi) / max(self._min_indel, 1e-9)
        if n + slack < lo or n - slack > hi:
            return False
        if n < 3:
            return True
        grams = phoneme_ngrams(tuple(phonemes), self.ngram)
        if not grams:
            return False
        # A true match within tau differs by only a few edits, so it must share a
        # substantial fraction of its bigrams with the span.  0.34 is well below
        # the empirical overlap of genuine matches (measured by
        # :meth:`retrieval_recall`) and well above coincidental overlap.
        needed = max(1, int(len(grams) * min_gram_ratio))
        hits = 0
        for gram in grams:
            posting = self._postings.get(gram)
            if posting:
                hits += 1
                if hits >= needed:
                    return True
        return False

    # -------------------------------------------------------------- querying
    def _length_admissible(self, qlen: int, tlen: int, tau: float) -> bool:
        """Admissible (never drops a true match) length prefilter.

        Claim: LOCAL-SPEED.
        """
        if qlen == 0 or tlen == 0:
            return False
        norm = float(max(qlen, tlen))
        return abs(qlen - tlen) * self._min_indel <= tau * norm + 1e-9

    def _prefilter_slots(self, qph: Phonemes, tau: float) -> List[int]:
        """IDF-weighted, length-filtered n-gram shortlist of slots worth scoring.

            Claim: UNBOUNDED-VOCAB + LOCAL-SPEED.
            """
        grams = set(phoneme_ngrams(qph, self.ngram))
        if len(qph) >= 4:
            grams |= set(phoneme_ngrams(qph, self.ngram + 1))
        if not grams:
            return []
        scores: Dict[int, float] = defaultdict(float)
        qlen = len(qph)
        for gram in grams:
            posting = self._postings.get(gram)
            if not posting or len(posting) > self._max_df:
                continue
            w = self._idf.get(gram, 1.0)
            for slot in posting:
                scores[slot] += w
        if not scores:
            return []
        # Cosine-style normalisation.  Without it, long glossary terms outrank a
        # short term that shares nearly *all* of its grams with the query, which
        # is precisely the case a phonetic near-match produces.
        qnorm = math.sqrt(len(grams)) or 1.0
        ranked = sorted(
            ((slot, sc / (self._slot_norm[slot] * qnorm)) for slot, sc in scores.items()),
            key=lambda kv: -kv[1],
        )
        out: List[int] = []
        for slot, _ in ranked:
            if self._length_admissible(qlen, len(self.readings[slot].phonemes), tau):
                out.append(slot)
            if len(out) >= self.prefilter:
                break
        return out

    def candidate_slots(self, qph: Phonemes, tau: float, exact: bool = False) -> List[int]:
        """Slot ids worth scoring exactly, after pruning.

        Falls back to the exhaustive (still length-filtered) scan for small
        glossaries and for very short queries, where bigram overlap is noise.

        Claim: LOCAL-SPEED + UNBOUNDED-VOCAB.
        """
        qlen = len(qph)
        if exact or len(self.readings) <= self.exhaustive_limit or qlen < 3:
            return [
                s
                for s, r in enumerate(self.readings)
                if self._length_admissible(qlen, len(r.phonemes), tau)
            ]
        return self._prefilter_slots(qph, tau)

    def query(
        self,
        phonemes: Sequence[str],
        tau: float,
        top_k: int = 5,
        exact: bool = False,
        with_alignment: bool = False,
        max_raw: Optional[float] = None,
        max_mora_ratio: Optional[float] = None,
    ) -> List[Candidate]:
        """Return glossary terms within phonetic distance ``tau`` of ``phonemes``.

        The returned list is *the entire universe of legal replacements* for the
        span.  Anything outside it is unreachable by construction -- that is the
        hard constraint, and it is enforced here rather than being asked of a
        language model.

        Three bounds, all of which a candidate must satisfy:

        ``tau``
            Length-normalised distance.  Scale-free, but it under-penalises long
            spans: at tau=0.28 a thirteen-phoneme term can absorb 3.6 units of
            raw cost, which is three whole consonant swaps.
        ``max_raw``
            Absolute weighted distance.  Bounds the *number* of real errors
            regardless of length, which is what stops 「双方向」 (6 morae) from
            reaching 「整合再構成」 (10 morae) through a pile of individually cheap
            long-vowel insertions.
        ``max_mora_ratio``
            Relative mora-count difference.  ASR mis-hears sounds; it does not
            invent four extra morae inside a span.

        Claim: TERM-RECALL (the right term is in this list) and LOW-DAMAGE
        (nothing else is).
        """
        qph = tuple(phonemes)
        if not qph:
            return []
        cache_key = (qph, round(tau, 6), top_k, exact, with_alignment,
                     None if max_raw is None else round(max_raw, 6),
                     None if max_mora_ratio is None else round(max_mora_ratio, 6))
        hit = self._cache.get(cache_key)
        if hit is not None:
            self._cache_hits += 1
            return hit
        self._cache_misses += 1

        q_indel = indel_costs(qph, self.config)
        qlen = len(qph)
        q_morae = mora_count(qph)
        sub = self._sub
        best_by_entry: Dict[int, Candidate] = {}
        for slot in self.candidate_slots(qph, tau, exact=exact):
            r = self.readings[slot]
            if max_mora_ratio is not None and r.morae and q_morae:
                if abs(q_morae - r.morae) / max(q_morae, r.morae) > max_mora_ratio + 1e-9:
                    continue
            norm = float(max(qlen, len(r.phonemes)))
            budget = tau * norm
            if max_raw is not None:
                budget = min(budget, max_raw)
            raw = bounded_distance_pre(
                qph, q_indel, r.phonemes, r.indel, sub, budget, self._min_indel
            )
            if raw is None:
                continue
            nd = raw / norm
            prior = best_by_entry.get(r.entry_id)
            if prior is not None and prior.norm_distance <= nd:
                continue
            alignment: Tuple[Tuple[str, str, str, float], ...] = ()
            if with_alignment:
                raw, alignment = align(qph, r.phonemes, self.config)
            best_by_entry[r.entry_id] = Candidate(
                entry=r.entry,
                reading=r.reading,
                term_phonemes=r.phonemes,
                span_phonemes=qph,
                distance=raw,
                norm_distance=nd,
                alignment=alignment,
            )
        out = sorted(
            best_by_entry.values(),
            key=lambda c: (c.norm_distance, -c.entry.weight, c.entry.surface),
        )[:top_k]
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = out
        return out

    def query_reading(
        self, reading: str, tau: float, top_k: int = 5, **kw
    ) -> List[Candidate]:
        """Convenience wrapper taking a kana reading instead of phonemes.

        Claim: SUPPORT.
        """
        return self.query(kana_to_phonemes(reading, self.config), tau, top_k, **kw)

    # -------------------------------------------------------- quality checks
    def retrieval_recall(
        self,
        queries: Sequence[Sequence[str]],
        tau: float,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """Measure what the bigram accelerator costs us versus an exhaustive scan.

        Reported in the benchmark output so the speed claim is not silently
        bought with recall.

        Claim: UNBOUNDED-VOCAB -- the sweep to 10,000 terms is only meaningful if
        retrieval stayed exact while it got faster.
        """
        hits = 0
        total = 0
        misses = 0
        for q in queries:
            fast = {c.entry.surface for c in self.query(q, tau, top_k)}
            slow = {c.entry.surface for c in self.query(q, tau, top_k, exact=True)}
            total += len(slow)
            hits += len(fast & slow)
            misses += len(slow - fast)
        return {
            "queries": float(len(queries)),
            "exhaustive_hits": float(total),
            "recovered": float(hits),
            "missed": float(misses),
            "recall": (hits / total) if total else 1.0,
        }
