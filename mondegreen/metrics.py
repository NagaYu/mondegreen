"""Evaluation metrics.  The damage rate is the one that matters most.

A post-corrector that raises term recall by breaking healthy text is worse than
useless, so this module defines *breakage* with as much care as it defines
success, and reports it three ways:

``damage_rate_chars``
    Of the characters that the raw ASR already got right, what fraction did we
    turn wrong?  This is the literal reading of 「元々正しかった箇所を壊した割合」
    and it is the headline number.

``damage_rate_terms``
    Of the glossary-term occurrences the raw ASR already got right, what fraction
    did we turn wrong?  The domain-relevant version.

``damage_rate_edits``
    Of the edits we chose to make, what fraction made things worse?  The
    discriminative one -- it is the y-axis of the correction-rate vs damage-rate
    curve that the gate threshold sweeps.

All three are computed against gold text that came from the corpus we fed to
TTS.  No model, LLM or human ever judges correctness in this pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Alignment = List[Tuple[str, int, int]]  # (op, index_in_a, index_in_b); -1 = gap


# --------------------------------------------------------------------------------------
# String alignment
# --------------------------------------------------------------------------------------

def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Plain unweighted edit distance, used for CER/WER.

    Claim: SUPPORT -- CER improvement is metric (1).
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j - 1] + (0 if ai == b[j - 1] else 1),
                prev[j] + 1,
                cur[j - 1] + 1,
            )
        prev, cur = cur, prev
    return prev[m]


def align_sequences(a: Sequence[str], b: Sequence[str]) -> Alignment:
    """Character alignment with backtrace: ``[(op, i, j), ...]``.

    ``op`` is one of ``"="`` (match), ``"~"`` (substitution), ``"-"`` (deletion
    from ``a``), ``"+"`` (insertion from ``b``).

    Claim: LOW-DAMAGE -- projecting a corrected span back onto the reference is
    the only honest way to ask "was this bit already right?".
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    out: Alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            out.append(("=" if a[i - 1] == b[j - 1] else "~", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append(("-", i - 1, -1))
            i -= 1
        else:
            out.append(("+", -1, j - 1))
            j -= 1
    out.reverse()
    return out


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate.  ``len(reference) == 0`` yields 0.0 for empty hyps.

    Claim: SUPPORT -- metric (1), overall CER improvement.
    """
    ref = list(reference)
    if not ref:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(ref, list(hypothesis)) / len(ref)


def corpus_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level CER: total edits over total reference characters.

    Micro-averaged, not the mean of per-sentence CERs, because short sentences
    would otherwise dominate.

    Claim: SUPPORT -- metric (1).
    """
    num = 0
    den = 0
    for r, h in zip(references, hypotheses):
        num += levenshtein(list(r), list(h))
        den += len(r)
    return num / den if den else 0.0


def tokenize_for_wer(text: str, reader=None) -> List[str]:
    """Morpheme-ish tokenisation for WER.

    Japanese has no spaces, so "word" error rate needs a segmenter.  With
    fugashi/pyopenjtalk installed we use real morphemes; otherwise we fall back
    to character-class runs, which is a coarser but *consistent* unit -- and
    since every condition is scored with the same tokeniser, comparisons stay
    fair even on the fallback.

    Claim: SUPPORT -- metric (1), the WER half.
    """
    from .reading import get_reader, char_kind

    if reader is None:
        try:
            reader = get_reader()
        except Exception:  # pragma: no cover
            reader = None
    if reader is not None and getattr(reader, "name", "fallback") != "fallback":
        return [t.surface for t in reader.tokenize(text) if t.kind not in ("space",)]
    out: List[str] = []
    buf = ""
    prev = ""
    for ch in text:
        k = char_kind(ch)
        if k == "space":
            if buf:
                out.append(buf)
                buf = ""
            prev = ""
            continue
        if k == "punct":
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
            prev = ""
            continue
        if k != prev and buf:
            out.append(buf)
            buf = ""
        buf += ch
        prev = k
    if buf:
        out.append(buf)
    return out


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str], reader=None) -> float:
    """Corpus-level WER over :func:`tokenize_for_wer` units.

    Claim: SUPPORT -- metric (1).
    """
    num = 0
    den = 0
    for r, h in zip(references, hypotheses):
        rt = tokenize_for_wer(r, reader)
        ht = tokenize_for_wer(h, reader)
        num += levenshtein(rt, ht)
        den += len(rt)
    return num / den if den else 0.0


# --------------------------------------------------------------------------------------
# Term recall
# --------------------------------------------------------------------------------------

def count_occurrences(text: str, term: str) -> int:
    """Non-overlapping occurrence count of ``term`` in ``text``.

    Claim: TERM-RECALL -- metric (2) is built entirely from these counts.
    """
    if not term:
        return 0
    n = 0
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            return n
        n += 1
        start = idx + len(term)


@dataclass
class TermRecall:
    """Metric (2): how many glossary-term occurrences survived into the output."""

    recovered: int = 0
    total: int = 0
    by_category: Dict[str, Tuple[int, int]] = None  # type: ignore[assignment]

    @property
    def recall(self) -> float:
        """Claim: TERM-RECALL."""
        return self.recovered / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return {
            "recovered": self.recovered,
            "total": self.total,
            "recall": self.recall,
            "by_category": {k: {"recovered": v[0], "total": v[1], "recall": (v[0] / v[1] if v[1] else 0.0)}
                            for k, v in (self.by_category or {}).items()},
        }


def term_recall(
    references: Sequence[str],
    hypotheses: Sequence[str],
    terms: Sequence[str],
    categories: Optional[Dict[str, str]] = None,
) -> TermRecall:
    """Fraction of glossary-term occurrences in the gold text present in the hypothesis.

    Counted per occurrence, capped at the gold count, so a hypothesis cannot
    inflate its recall by repeating a term.

    Claim: TERM-RECALL -- this is metric (2) and the y-axis of the headline
    glossary-size figure.
    """
    categories = categories or {}
    rec = 0
    tot = 0
    by_cat: Dict[str, List[int]] = {}
    for ref, hyp in zip(references, hypotheses):
        for term in terms:
            gold_n = count_occurrences(ref, term)
            if not gold_n:
                continue
            hyp_n = min(count_occurrences(hyp, term), gold_n)
            rec += hyp_n
            tot += gold_n
            cat = categories.get(term, "")
            slot = by_cat.setdefault(cat, [0, 0])
            slot[0] += hyp_n
            slot[1] += gold_n
    return TermRecall(rec, tot, {k: (v[0], v[1]) for k, v in by_cat.items()})


# --------------------------------------------------------------------------------------
# Damage
# --------------------------------------------------------------------------------------

@dataclass
class DamageReport:
    """Metric (3), the most important one.  Three complementary views of breakage."""

    correct_chars_before: int = 0
    broken_chars: int = 0
    fixed_chars: int = 0
    correct_terms_before: int = 0
    broken_terms: int = 0
    fixed_terms: int = 0
    edits: int = 0
    damaging_edits: int = 0
    repairing_edits: int = 0
    neutral_edits: int = 0

    @property
    def damage_rate_chars(self) -> float:
        """Headline 破壊率: originally-correct characters that we turned wrong.

        Claim: LOW-DAMAGE.
        """
        return self.broken_chars / self.correct_chars_before if self.correct_chars_before else 0.0

    @property
    def damage_rate_terms(self) -> float:
        """Glossary-term occurrences that were right before and wrong after.

        Claim: LOW-DAMAGE.
        """
        return self.broken_terms / self.correct_terms_before if self.correct_terms_before else 0.0

    @property
    def damage_rate_edits(self) -> float:
        """Of the edits we made, the fraction that made things worse.

        Claim: LOW-DAMAGE -- the y-axis of the correction-rate vs damage-rate curve.
        """
        return self.damaging_edits / self.edits if self.edits else 0.0

    @property
    def repair_rate_edits(self) -> float:
        """Of the edits we made, the fraction that made things better.

        Claim: TERM-RECALL.
        """
        return self.repairing_edits / self.edits if self.edits else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Claim: SUPPORT."""
        d = {k: float(v) for k, v in asdict(self).items()}
        d.update(
            damage_rate_chars=self.damage_rate_chars,
            damage_rate_terms=self.damage_rate_terms,
            damage_rate_edits=self.damage_rate_edits,
            repair_rate_edits=self.repair_rate_edits,
        )
        return d


def _correct_char_positions(reference: str, hypothesis: str) -> set:
    """Indices in ``hypothesis`` that align to an identical reference character.

    Claim: LOW-DAMAGE -- defines "originally correct" for the headline metric.
    """
    ok = set()
    for op, i, j in align_sequences(list(reference), list(hypothesis)):
        if op == "=" and j >= 0:
            ok.add(j)
    return ok


def damage_report(
    references: Sequence[str],
    baselines: Sequence[str],
    corrected: Sequence[str],
    terms: Sequence[str] = (),
    edits_per_item: Optional[Sequence[int]] = None,
    edit_outcomes: Optional[Sequence[Sequence[str]]] = None,
) -> DamageReport:
    """Compare raw ASR and corrected output against gold, counting what broke.

    Character view: a character is *correct before* if it aligns to an identical
    reference character in the baseline.  We then measure whether the same
    reference character is still matched after correction.  This is stable under
    the index shifts a replacement causes, because both sides are projected onto
    the reference rather than onto each other.

    Claim: LOW-DAMAGE -- metric (3).
    """
    rep = DamageReport()
    for k, (ref, base, corr) in enumerate(zip(references, baselines, corrected)):
        ref_ok_before = {i for op, i, j in align_sequences(list(ref), list(base)) if op == "=" and i >= 0}
        ref_ok_after = {i for op, i, j in align_sequences(list(ref), list(corr)) if op == "=" and i >= 0}
        rep.correct_chars_before += len(ref_ok_before)
        rep.broken_chars += len(ref_ok_before - ref_ok_after)
        rep.fixed_chars += len(ref_ok_after - ref_ok_before)

        for term in terms:
            gold_n = count_occurrences(ref, term)
            if not gold_n:
                continue
            before = min(count_occurrences(base, term), gold_n)
            after = min(count_occurrences(corr, term), gold_n)
            rep.correct_terms_before += before
            if after < before:
                rep.broken_terms += before - after
            elif after > before:
                rep.fixed_terms += after - before

    if edit_outcomes is not None:
        for outcomes in edit_outcomes:
            for o in outcomes:
                rep.edits += 1
                if o == "damage":
                    rep.damaging_edits += 1
                elif o == "repair":
                    rep.repairing_edits += 1
                else:
                    rep.neutral_edits += 1
    elif edits_per_item is not None:
        rep.edits += int(sum(edits_per_item))
    return rep


def classify_edit(reference: str, before: str, after: str) -> str:
    """Label one whole-sentence edit as repair / damage / neutral by CER delta.

    Claim: LOW-DAMAGE -- gives every edit a ground-truth outcome without any
    model in the loop.
    """
    d_before = levenshtein(list(reference), list(before))
    d_after = levenshtein(list(reference), list(after))
    if d_after < d_before:
        return "repair"
    if d_after > d_before:
        return "damage"
    return "neutral"


def classify_span_edit(
    reference: str,
    baseline: str,
    span: Tuple[int, int],
    replacement: str,
) -> str:
    """Label a span replacement as repair / damage / neutral / no-op.

    Two tests, in order:

    1. **Was the span already exactly right?**  Project it onto the reference and
       compare.  If it matched, any change is damage.  This is the precision test
       that makes the damage rate meaningful, so it is checked first -- but only
       when the projection is trustworthy (the projected reference span has to be
       about the same length as the span itself; a wildly different length means
       the alignment wandered and the projection is not evidence of anything).

    2. **Did the sentence get closer to gold?**  Apply the edit to the whole
       baseline and compare edit distance before and after.  This is robust to
       alignment wander in a way span projection is not: restoring
       「曉小圧」 to 「アカツキコア2」 projects onto only 「コア2」 and looks like
       damage span-locally, while the sentence plainly improves.

    Claim: LOW-DAMAGE -- the gate is trained on these labels and the operating
    curve is drawn from them, so a systematically wrong label would corrupt every
    LOW-DAMAGE number in the repo.
    """
    s, e = span
    original = baseline[s:e]
    if replacement == original:
        return "no-op"

    alignment = align_sequences(list(baseline), list(reference))
    ref_idx = [j for op, i, j in alignment if i is not None and s <= i < e and j >= 0]
    if ref_idx:
        ref_span = reference[min(ref_idx) : max(ref_idx) + 1]
        trustworthy = abs(len(ref_span) - len(original)) <= max(1, len(original) // 2)
        if trustworthy:
            if original == ref_span:
                return "damage"
            if replacement == ref_span:
                return "repair"

    after = baseline[:s] + replacement + baseline[e:]
    d_before = levenshtein(list(reference), list(baseline))
    d_after = levenshtein(list(reference), list(after))
    if d_after < d_before:
        return "repair"
    if d_after > d_before:
        return "damage"
    return "neutral"


# --------------------------------------------------------------------------------------
# Hallucination removal
# --------------------------------------------------------------------------------------

def hallucination_removal_rate(
    references: Sequence[str],
    baselines: Sequence[str],
    corrected: Sequence[str],
    patterns: Sequence[str],
) -> Dict[str, float]:
    """Metric (5): how many canned Whisper hallucinations we removed, and at what cost.

    ``false_removals`` counts patterns that were *genuinely spoken* (present in
    the reference) and got deleted anyway -- removing those is damage, and the
    number is reported next to the win so the trade is visible.

    Claim: LOW-DAMAGE -- deleting text is the highest-variance thing a corrector
    can do, so it gets its own precision number.
    """
    present = 0
    removed = 0
    legit = 0
    false_removals = 0
    for ref, base, corr in zip(references, baselines, corrected):
        for pat in patterns:
            in_ref = count_occurrences(ref, pat)
            in_base = count_occurrences(base, pat)
            in_corr = count_occurrences(corr, pat)
            spurious = max(0, in_base - in_ref)
            if spurious:
                present += spurious
                removed += max(0, min(spurious, in_base - in_corr))
            if in_ref:
                legit += in_ref
                false_removals += max(0, min(in_ref, in_base) - in_corr)
    return {
        "hallucinations": float(present),
        "removed": float(removed),
        "removal_rate": (removed / present) if present else 0.0,
        "legitimate_occurrences": float(legit),
        "false_removals": float(false_removals),
        "false_removal_rate": (false_removals / legit) if legit else 0.0,
    }


def summarize(
    references: Sequence[str],
    baselines: Sequence[str],
    corrected: Sequence[str],
    terms: Sequence[str] = (),
    categories: Optional[Dict[str, str]] = None,
    hallucination_patterns: Sequence[str] = (),
    edit_outcomes: Optional[Sequence[Sequence[str]]] = None,
    reader=None,
) -> Dict[str, object]:
    """Compute every metric for one experimental condition, in one pass.

    Claim: SUPPORT -- one function so that conditions (A)-(E) are guaranteed to
    be scored identically.
    """
    dmg = damage_report(references, baselines, corrected, terms, edit_outcomes=edit_outcomes)
    tr_base = term_recall(references, baselines, terms, categories)
    tr_corr = term_recall(references, corrected, terms, categories)
    out: Dict[str, object] = {
        "n": len(references),
        "cer": corpus_cer(references, corrected),
        "cer_baseline": corpus_cer(references, baselines),
        "wer": corpus_wer(references, corrected, reader),
        "wer_baseline": corpus_wer(references, baselines, reader),
        "term_recall": tr_corr.to_dict(),
        "term_recall_baseline": tr_base.to_dict(),
        "damage": dmg.to_dict(),
    }
    out["cer_improvement"] = float(out["cer_baseline"]) - float(out["cer"])  # type: ignore[arg-type]
    out["cer_improvement_rel"] = (
        out["cer_improvement"] / out["cer_baseline"] if out["cer_baseline"] else 0.0  # type: ignore[operator]
    )
    out["term_recall_gain"] = tr_corr.recall - tr_base.recall
    if hallucination_patterns:
        out["hallucination"] = hallucination_removal_rate(
            references, baselines, corrected, hallucination_patterns
        )
    return out
