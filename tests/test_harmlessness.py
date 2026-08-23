"""Harmlessness: with nothing to say, the corrector says nothing.

An empty glossary must be an exact identity function.  A glossary that shares no
sounds with the text must be too.  This is the property that makes it safe to run
Mondegreen over a transcript you are not sure needs correcting.
"""

from __future__ import annotations

import pytest

from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig
from mondegreen.glossary import Glossary, loads_glossary
from tests.conftest import CLEAN_SENTENCES

pytestmark = pytest.mark.invariant


def test_empty_glossary_is_the_identity(empty_glossary):
    """No glossary, no edits -- byte-for-byte identical output."""
    corrector = ConstrainedCorrector(
        empty_glossary, CorrectorConfig(remove_hallucinations=False)
    )
    for text in CLEAN_SENTENCES:
        result = corrector.correct(text)
        assert result.text == text
        assert result.corrections == []
        assert not result.changed


def test_empty_glossary_identity_across_thresholds(empty_glossary):
    """Even with the bound wide open and the gate fully off, there is nothing to pick."""
    for tau in (0.05, 0.3, 0.9):
        corrector = ConstrainedCorrector(
            empty_glossary,
            CorrectorConfig(tau=tau, gate_threshold=0.0, remove_hallucinations=False),
        )
        for text in CLEAN_SENTENCES:
            assert corrector.correct(text).text == text


def test_empty_glossary_still_removes_hallucinations(empty_glossary):
    """Hallucination removal is glossary-independent, and must stay that way.

    It is the one edit Mondegreen makes without a glossary, so it gets its own
    explicit assertion rather than being an accident of the identity test.
    """
    corrector = ConstrainedCorrector(
        empty_glossary, CorrectorConfig(remove_hallucinations=True)
    )
    out = corrector.correct("本日の議題は予算です。ご視聴ありがとうございました。")
    assert out.text == "本日の議題は予算です。"
    assert len(out.removed_hallucinations) == 1


def test_unrelated_glossary_does_not_touch_clean_text(small_glossary):
    """A real glossary must leave text that shares none of its sounds alone."""
    corrector = ConstrainedCorrector(small_glossary, CorrectorConfig(gate_threshold=0.5))
    for text in CLEAN_SENTENCES:
        result = corrector.correct(text)
        assert result.text == text, f"unexpectedly edited: {result.corrections}"


def test_text_already_containing_the_terms_is_untouched(small_glossary):
    """Correct text stays correct -- honorifics and particles included.

    Regression test for the failure mode where a corrector "restores" 新藤 by
    deleting the さん after it.
    """
    corrector = ConstrainedCorrector(small_glossary, CorrectorConfig(gate_threshold=0.5))
    for text in [
        "新藤さんが量子誤り訂正の話をしました。",
        "中村さんはミライドライブの担当です。",
        "佐藤太郎さんと逐次復号について議論しました。",
        "新藤さんの量子誤り訂正のミライドライブへの適用について。",
    ]:
        result = corrector.correct(text)
        assert result.text == text, f"damaged {text!r} -> {result.text!r}"


def test_idempotence(small_glossary):
    """Correcting a corrected transcript changes nothing further."""
    corrector = ConstrainedCorrector(small_glossary, CorrectorConfig(gate_threshold=0.5))
    once = corrector.correct("進藤さんと両氏誤り訂正の話をしました。").text
    twice = corrector.correct(once).text
    assert once == twice


def test_output_is_fully_explained_by_its_own_receipts(synthetic_corpus):
    """The corrected text must be reconstructible from the source plus the receipts.

    This is stronger than a length bound and it is the property that actually
    matters: nothing may change that the corrector did not report.  If the output
    ever contains an edit with no matching :class:`Correction`, this fails.

    (Length alone is a bad proxy -- restoring 「ジケイレツジョウリュウ」 to
    「時系列蒸留」 legitimately halves it.)
    """
    glossary = synthetic_corpus["test_glossary"]
    corrector = ConstrainedCorrector(glossary, CorrectorConfig(gate_threshold=0.5))
    checked = 0
    for pair in synthetic_corpus["test_pairs"]:
        out = corrector.correct(pair.hypothesis)
        if out.removed_hallucinations:
            continue  # deletion also trims punctuation; covered by its own test
        rebuilt = out.source
        for c in sorted(out.corrections, key=lambda c: -c.start):
            assert rebuilt[c.start : c.end] == c.original, "receipt offsets do not match source"
            rebuilt = rebuilt[: c.start] + c.replacement + rebuilt[c.end :]
        assert rebuilt == out.text
        checked += 1
    assert checked > 20


def test_no_novel_characters_appear(synthetic_corpus):
    """Every character in the output came from the source or from a glossary term.

    A post-processor that invents text -- new punctuation, a helpful clarifying
    clause, a normalised number -- fails this even if CER improves.
    """
    glossary = synthetic_corpus["test_glossary"]
    corrector = ConstrainedCorrector(glossary, CorrectorConfig(gate_threshold=0.5))
    allowed_from_glossary = set("".join(glossary.surfaces()))
    for pair in synthetic_corpus["test_pairs"][:80]:
        out = corrector.correct(pair.hypothesis)
        novel = set(out.text) - set(out.source) - allowed_from_glossary
        assert not novel, f"invented characters {sorted(novel)} in {out.text!r}"
