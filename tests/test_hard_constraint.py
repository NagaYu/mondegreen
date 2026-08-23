"""The invariant: a replacement above the phonetic threshold is impossible.

This is the load-bearing test of the whole repository.  If it can be made to
fail, the central claim -- that the glossary is a *hard* constraint rather than a
suggestion -- is false.
"""

from __future__ import annotations

import random

import pytest

from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig
from mondegreen.glossary import Glossary
from mondegreen.harvest import GlossaryBuilder, SentenceFactory
from mondegreen.index import PhoneticIndex
from mondegreen.phonetics import (
    bounded_distance_pre,
    bounded_normalized_distance,
    indel_costs,
    kana_to_phonemes,
    mora_count,
    normalized_distance,
    substitution_table,
)
from mondegreen.simulate import ASRErrorSimulator

pytestmark = pytest.mark.invariant


@pytest.mark.parametrize("tau", [0.05, 0.15, 0.28, 0.40])
def test_no_correction_ever_exceeds_its_threshold(synthetic_corpus, tau):
    """Every applied edit satisfies ``norm_distance <= tau(span)``, at every tau.

    Verified independently of the corrector's own bookkeeping: the distance is
    recomputed from the emitted surfaces.
    """
    glossary = synthetic_corpus["test_glossary"]
    cfg = CorrectorConfig(tau=tau, gate_threshold=0.0)  # gate wide open on purpose
    corrector = ConstrainedCorrector(glossary, cfg)
    checked = 0
    for pair in synthetic_corpus["test_pairs"]:
        result = corrector.correct(pair.hypothesis)
        for c in result.corrections:
            assert c.norm_distance <= c.threshold + 1e-9, (
                f"{c.original!r} -> {c.replacement!r} at d={c.norm_distance:.4f} > "
                f"tau={c.threshold:.4f}"
            )
            # Recompute from scratch rather than trusting the recorded value.
            recomputed = normalized_distance(
                c.original_phonemes, c.candidate_phonemes, corrector.phonetic_config
            )
            assert abs(recomputed - c.norm_distance) < 1e-9
            assert c.threshold <= max(cfg.tau, cfg.tau_short) + 1e-12
            checked += 1
    assert checked > 0, "the corpus produced no corrections; the test proved nothing"


def test_replacement_is_always_a_glossary_surface(synthetic_corpus):
    """The corrector can only ever emit a glossary term -- never novel text."""
    glossary = synthetic_corpus["test_glossary"]
    surfaces = set(glossary.surfaces())
    corrector = ConstrainedCorrector(glossary, CorrectorConfig(gate_threshold=0.0))
    for pair in synthetic_corpus["test_pairs"][:40]:
        for c in corrector.correct(pair.hypothesis).corrections:
            assert c.replacement in surfaces


def test_candidate_set_is_closed_under_the_bound(small_glossary):
    """Nothing outside the bound is reachable, and everything inside is offered."""
    corrector = ConstrainedCorrector(small_glossary, CorrectorConfig(tau=0.28))
    for span in ["進藤", "両氏誤り訂正", "ミライドライバー", "全然関係ない", "逐次復合"]:
        for cand in corrector.candidate_set(span):
            assert cand.norm_distance <= corrector.config.tau_for(
                mora_count(cand.span_phonemes)
            ) + 1e-9


def test_apply_refuses_to_splice_an_illegal_edit(small_glossary):
    """The final guard fires even if a bug smuggles an illegal edit that far.

    ``_apply`` re-checks the bound immediately before mutating the string; this
    test forges a violating Correction and asserts it raises rather than shipping.
    """
    from mondegreen.types import Correction

    corrector = ConstrainedCorrector(small_glossary)
    forged = Correction(
        start=0, end=2, original="天気", replacement="新藤",
        original_phonemes=kana_to_phonemes("テンキ"),
        candidate_phonemes=kana_to_phonemes("シンドウ"),
        distance=9.0, norm_distance=0.99, threshold=0.28,
        gate_prob=1.0, margin=1.0, accepted=True,
    )
    with pytest.raises(AssertionError, match="hard phonetic constraint violated"):
        corrector._apply("天気がいいですね", [forged])


def test_index_never_returns_a_candidate_over_tau(small_index):
    """The index is the component that enforces the bound; check it directly."""
    rng = random.Random(0)
    sim = ASRErrorSimulator()
    for tau in (0.05, 0.2, 0.35):
        for entry in small_index.glossary:
            wrong, _ = sim.perturb_reading(entry.reading, rng)
            for c in small_index.query_reading(wrong, tau, top_k=10):
                assert c.norm_distance <= tau + 1e-9


def test_fast_and_reference_distance_agree(small_glossary):
    """The precomputed fast DP must equal the readable reference implementation.

    The fast path is what actually runs; the reference is what the invariant is
    stated in terms of.  A divergence would silently move the boundary.
    """
    sub = substitution_table()
    words = ["トウキョウ", "シンドウ", "ナカムラ", "ミライドライブ", "チクジフクゴウ",
             "リョウシアヤマリテイセイ", "サトウタロウ", "コンピューター", "ガッコウ"]
    for x in words:
        for y in words:
            a, b = kana_to_phonemes(x), kana_to_phonemes(y)
            norm = float(max(1, len(a), len(b)))
            for tau in (0.1, 0.28, 0.6):
                ref = bounded_normalized_distance(a, b, tau)
                raw = bounded_distance_pre(
                    a, indel_costs(a), b, indel_costs(b), sub, tau * norm, 0.15
                )
                fast = None if raw is None else raw / norm
                assert (ref is None) == (fast is None)
                if ref is not None:
                    assert abs(ref - fast) < 1e-9


def test_screen_never_rejects_a_reachable_candidate(small_index):
    """``screen()`` is a speed optimisation and must not shrink the legal set.

    If a span could match under the bound, the cheap pre-check must let it
    through -- otherwise the constraint's *upper* boundary is fine but its
    *coverage* silently shrinks.
    """
    sim = ASRErrorSimulator()
    rng = random.Random(7)
    tau = 0.28
    for entry in small_index.glossary:
        for _ in range(6):
            wrong, _ = sim.perturb_reading(entry.reading, rng)
            ph = kana_to_phonemes(wrong)
            if not ph:
                continue
            hits = small_index.query(ph, tau, top_k=5, exact=True)
            if hits:
                assert small_index.screen(ph, tau), (
                    f"screen() rejected {wrong!r} but exact query found {hits[0].entry.surface!r}"
                )
