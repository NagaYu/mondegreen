"""Quantisation must not cost meaningful term recall.

Condition (E) claims that the Q4_K_M build performs like the unquantised one.
There are two ways to check that, and both are here:

1. **Structural (always runs).**  The only component quantisation can degrade is
   the optional LM re-ranker.  The hard constraint, the index and the gate are
   plain arithmetic and are bit-identical at any quantisation.  So the *worst
   case* of quantisation is bounded by "the LM is gone entirely" -- and that
   bound is testable today, with no model files.  If term recall barely moves
   when the LM is removed, then no amount of quantisation can move it much
   either.

2. **Empirical (runs when a checkpoint is available).**  Set
   ``MONDEGREEN_QUANTIZED_MODEL`` to a ``.gguf`` file or an MLX directory and the
   real comparison runs.
"""

from __future__ import annotations

import os

import pytest

from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig
from mondegreen.metrics import summarize, term_recall

#: Absolute term-recall points that may be lost to quantisation.
MAX_RECALL_DROP = 0.02
#: Absolute damage-rate points that quantisation may add.
MAX_DAMAGE_INCREASE = 0.005

pytestmark = pytest.mark.invariant


class _FakeReranker:
    """A perfect-information reranker, standing in for an unquantised LM.

    It always prefers the true replacement, i.e. it is strictly better than any
    real model could be.  If recall with this oracle is close to recall with no
    LM at all, the LM contributes almost nothing -- which is the property that
    makes quantisation safe.
    """

    name = "oracle"

    def __init__(self, gold_by_prefix):
        self._gold = gold_by_prefix

    def score_candidates(self, prefix, candidates, suffix):
        return [1.0 if c in self._gold else 0.0 for c in candidates]


def _evaluate(pairs, glossary, lm=None):
    corrector = ConstrainedCorrector(
        glossary, CorrectorConfig(gate_threshold=0.5), lm=lm
    )
    refs = [p.gold for p in pairs]
    raws = [p.hypothesis for p in pairs]
    outs = [corrector.correct(h).text for h in raws]
    return summarize(refs, raws, outs, list(glossary.surfaces()))


def test_recall_barely_depends_on_the_lm(synthetic_corpus):
    """Removing the LM entirely -- quantisation's worst case -- must be near-free.

    This is the structural bound behind the (D) vs (E) claim.
    """
    glossary = synthetic_corpus["test_glossary"]
    pairs = synthetic_corpus["test_pairs"]
    gold_surfaces = set(glossary.surfaces())

    without = _evaluate(pairs, glossary, lm=None)
    with_oracle = _evaluate(pairs, glossary, lm=_FakeReranker(gold_surfaces))

    drop = with_oracle["term_recall"]["recall"] - without["term_recall"]["recall"]
    assert drop <= MAX_RECALL_DROP, (
        f"term recall depends on the LM by {drop:.3f} points, which is more than "
        f"quantisation is allowed to cost ({MAX_RECALL_DROP})"
    )


def test_damage_barely_depends_on_the_lm(synthetic_corpus):
    """The same bound, for the metric that actually matters."""
    glossary = synthetic_corpus["test_glossary"]
    pairs = synthetic_corpus["test_pairs"]
    without = _evaluate(pairs, glossary, lm=None)
    with_oracle = _evaluate(pairs, glossary, lm=_FakeReranker(set(glossary.surfaces())))
    delta = (
        without["damage"]["damage_rate_chars"] - with_oracle["damage"]["damage_rate_chars"]
    )
    assert delta <= MAX_DAMAGE_INCREASE, delta


def test_constraint_is_bit_identical_without_the_lm(synthetic_corpus):
    """The legal candidate set must not depend on the LM at all.

    An LM that could change *which* replacements are legal would make the safety
    argument quantisation-dependent.  It cannot, and this asserts it.
    """
    glossary = synthetic_corpus["test_glossary"]
    plain = ConstrainedCorrector(glossary, CorrectorConfig())
    with_lm = ConstrainedCorrector(
        glossary, CorrectorConfig(), lm=_FakeReranker(set(glossary.surfaces()))
    )
    for span in ["進藤", "両氏誤り訂正", "ミライドライバー", "稼働率"]:
        a = [(c.entry.surface, round(c.norm_distance, 9)) for c in plain.candidate_set(span)]
        b = [(c.entry.surface, round(c.norm_distance, 9)) for c in with_lm.candidate_set(span)]
        assert a == b


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("MONDEGREEN_QUANTIZED_MODEL"),
    reason="set MONDEGREEN_QUANTIZED_MODEL to a .gguf file or MLX dir to run the real check",
)
def test_real_quantized_model_recall_drop(synthetic_corpus):
    """The empirical (D) vs (E) comparison, when a checkpoint is available."""
    from mondegreen.runtime import build_reranker

    glossary = synthetic_corpus["test_glossary"]
    pairs = synthetic_corpus["test_pairs"][:60]
    lm = build_reranker(os.environ["MONDEGREEN_QUANTIZED_MODEL"])
    unquantised = _evaluate(pairs, glossary, lm=None)
    quantised = _evaluate(pairs, glossary, lm=lm)
    drop = unquantised["term_recall"]["recall"] - quantised["term_recall"]["recall"]
    assert drop <= MAX_RECALL_DROP, f"quantised model lost {drop:.3f} recall points"
    increase = (
        quantised["damage"]["damage_rate_chars"] - unquantised["damage"]["damage_rate_chars"]
    )
    assert increase <= MAX_DAMAGE_INCREASE
