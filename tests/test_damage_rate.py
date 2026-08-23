"""The damage rate must stay under budget on held-out data.

This is the test that would fail first if any change to the phonetics, the index,
the span enumerator or the gate made the corrector more aggressive.  The budgets
below are deliberately a little looser than the measured values so the suite is
not flaky, but tight enough that a real regression trips them.
"""

from __future__ import annotations

import pytest

from mondegreen.benchmark import collect_sweep_records, sweep_operating_points
from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig
from mondegreen.gate import ConservativeGate, pick_threshold
from mondegreen.metrics import summarize, term_recall
from tests.conftest import CLEAN_SENTENCES

pytestmark = pytest.mark.invariant

#: Of the characters the raw ASR already got right, at most this fraction may be
#: broken.  The headline 破壊率.
MAX_DAMAGE_RATE_CHARS = 0.01
#: Of the glossary-term occurrences that were already correct, at most this many.
MAX_DAMAGE_RATE_TERMS = 0.02
#: Of the edits we choose to make, at most this fraction may be wrong.
MAX_DAMAGE_RATE_EDITS = 0.15


def _run(corpus, threshold: float = 0.5):
    glossary = corpus["test_glossary"]
    pairs = corpus["test_pairs"]
    corrector = ConstrainedCorrector(
        glossary, CorrectorConfig(gate_threshold=threshold)
    )
    refs = [p.gold for p in pairs]
    raws = [p.hypothesis for p in pairs]
    outs = [corrector.correct(h).text for h in raws]
    return refs, raws, outs, list(glossary.surfaces())


def test_damage_rate_is_under_budget(synthetic_corpus):
    """Metric (3), on a glossary and corpus never seen during development."""
    refs, raws, outs, terms = _run(synthetic_corpus)
    m = summarize(refs, raws, outs, terms)
    dmg = m["damage"]
    assert dmg["damage_rate_chars"] <= MAX_DAMAGE_RATE_CHARS, dmg
    assert dmg["damage_rate_terms"] <= MAX_DAMAGE_RATE_TERMS, dmg


def test_correction_actually_helps(synthetic_corpus):
    """A low damage rate is trivial if nothing is corrected; check the other side."""
    refs, raws, outs, terms = _run(synthetic_corpus)
    before = term_recall(refs, raws, terms).recall
    after = term_recall(refs, outs, terms).recall
    assert after > before + 0.15, f"term recall {before:.3f} -> {after:.3f}"
    m = summarize(refs, raws, outs, terms)
    assert m["cer"] < m["cer_baseline"], (m["cer"], m["cer_baseline"])


def test_damage_never_increases_as_the_gate_tightens(synthetic_corpus):
    """The operating curve must be monotone: a stricter gate cannot do more harm.

    A non-monotone curve would mean the threshold is not a safety knob, which
    would invalidate the second headline figure.
    """
    glossary = synthetic_corpus["test_glossary"]
    records = collect_sweep_records(synthetic_corpus["test_pairs"], glossary, tau=0.28)
    points = sweep_operating_points(records, thresholds=[i / 20 for i in range(21)])
    damages = [p.damage_rate for p in points]
    for lo, hi in zip(damages, damages[1:]):
        assert hi <= lo + 1e-9, f"damage rose as the threshold tightened: {damages}"


def test_a_conservative_threshold_reaches_zero_damage(synthetic_corpus):
    """There must exist an operating point with zero damage and real corrections.

    If no such point exists, the system cannot be deployed safely at any setting.
    """
    glossary = synthetic_corpus["test_glossary"]
    records = collect_sweep_records(synthetic_corpus["test_pairs"], glossary, tau=0.28)
    points = sweep_operating_points(records)
    zero = [p for p in points if p.damage_rate == 0.0 and p.correction_rate > 0.3]
    assert zero, "no threshold achieves zero damage with a useful correction rate"


def test_clean_text_is_never_damaged(synthetic_corpus):
    """Text with no errors in it must survive untouched at the default setting."""
    glossary = synthetic_corpus["test_glossary"]
    corrector = ConstrainedCorrector(glossary, CorrectorConfig(gate_threshold=0.5))
    for text in CLEAN_SENTENCES:
        assert corrector.correct(text).text == text


def test_pick_threshold_respects_the_budget(synthetic_corpus):
    """``pick_threshold`` must not return a point that violates the budget."""
    glossary = synthetic_corpus["test_glossary"]
    records = collect_sweep_records(synthetic_corpus["test_pairs"], glossary, tau=0.28)
    points = sweep_operating_points(records)
    for budget in (0.0, 0.005, 0.02):
        th = pick_threshold(points, budget)
        at = min(points, key=lambda p: abs(p.threshold - th))
        assert at.damage_rate <= budget + 1e-9, (budget, at.to_dict())
