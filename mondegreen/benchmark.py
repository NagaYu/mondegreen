"""The experiment: conditions (A)-(E), five metrics, and the glossary-size sweep.

Experimental design
===================

**Separation.**  Training and evaluation share no speaker, no source sentence and
no glossary term.  :meth:`~mondegreen.harvest.GlossaryBuilder.build_pair`
guarantees the glossary halves are disjoint by surface *and* by reading, and the
sentence factory is seeded separately for each split.  The evaluation glossary is
therefore, by construction, "a glossary the system has never seen" -- the
condition the brief asks for is the default, not an extra.

**The sweep.**  The set of *target* terms is held fixed while the glossary grows
around it.  A size-K glossary is ``shuffle(target_terms + K - len(targets)
distractors)``.  This isolates vocabulary size from vocabulary content, and it is
what makes the headline figure honest: as K grows, the targets are pushed out of
Whisper's 244-token prompt at a rate set purely by K, while the phonetic index
keeps every one of them.

**Provenance.**  Every result carries ``provenance``.  Conditions run against a
real Whisper or a real cloud model are ``measured``; the rest are ``simulated``
and carry the parameters that produced them.  The two are never averaged
together, and the figure generator watermarks simulated plots.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .baselines import (
    SimulatedCloudLLM,
    SimulatedWhisperPrompt,
    SimulationParams,
    whisper_prompt_capacity,
)
from .corrector import ConstrainedCorrector, CorrectorConfig, select_non_overlapping
from .gate import ConservativeGate, SweepPoint, pick_threshold, sweep_thresholds
from .glossary import Glossary
from .hallucination import DEFAULT_PATTERNS
from .harvest import ErrorHarvester, GlossaryBuilder, SentenceFactory
from .metrics import classify_span_edit, count_occurrences, summarize
from .phonetics import DEFAULT_CONFIG, PhoneticConfig
from .runtime import benchmark_corrector, compare_to_cloud, machine_info
from .types import ErrorPair, GlossaryEntry, SpanDecision

CONDITION_LABELS: Dict[str, str] = {
    "A": "raw Whisper (no correction)",
    "B": "Whisper initial_prompt stuffed with the glossary (244-token ceiling)",
    "C": "cloud LLM post-processing",
    "D": "Mondegreen (phonetic constraint + conservative gate)",
    "E": "Mondegreen (after Q4_K_M quantisation)",
}


@dataclass
class BenchmarkConfig:
    """Everything that defines a benchmark run.

    Claim: SUPPORT -- serialised into every result file so a number can always be
    traced back to the configuration that produced it.
    """

    #: Sizes for the headline **coverage** sweep: the glossary grows to cover more
    #: of the domain actually spoken.  Deliberately starts below the ~17 terms
    #: that fit in Whisper's 244-token prompt, so the ceiling is visible.
    glossary_sizes: Tuple[int, ...] = (10, 30, 100, 300, 1000, 3000, 10000)
    #: Sizes for the **distractor** control sweep: targets fixed, glossary padded.
    control_sizes: Tuple[int, ...] = (100, 1000, 10000)
    n_target_terms: int = 100
    n_sentences: int = 600
    n_train_sentences: int = 900
    n_train_glossary: int = 400
    tau: float = 0.28
    gate_threshold: float = 0.5
    max_damage_rate: float = 0.01
    #: Bounds used to *generate gate training candidates* -- deliberately wider
    #: than inference, so the gate sees negatives.  See
    #: :func:`collect_gate_training_data`.
    gate_train_tau: float = 0.45
    gate_train_max_raw: float = 1.2
    seed: int = 20260823
    gate_path: Optional[str] = None
    include_cloud: bool = False
    quantized_model: Optional[str] = None
    reader: str = "auto"
    hallucination_patterns: Tuple[str, ...] = DEFAULT_PATTERNS
    simulation: SimulationParams = field(default_factory=SimulationParams)

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        d = asdict(self)
        d["simulation"] = self.simulation.to_dict()
        d["hallucination_patterns"] = list(self.hallucination_patterns)
        d["glossary_sizes"] = list(self.glossary_sizes)
        return d


# --------------------------------------------------------------------------------------
# Gate training data
# --------------------------------------------------------------------------------------

def collect_gate_training_data(
    pairs: Sequence[ErrorPair],
    glossary: Glossary,
    tau: float = 0.28,
    gate: Optional[ConservativeGate] = None,
    phonetic_config: Optional[PhoneticConfig] = None,
    reader: str = "auto",
    config: Optional[CorrectorConfig] = None,
) -> Tuple[List[SpanDecision], List[Tuple[float, str]]]:
    """Label every legal span replacement against gold text.

    Each proposal is scored by :func:`~mondegreen.metrics.classify_span_edit`,
    which projects the span onto the reference and asks whether replacing it
    moves toward or away from gold.  ``repair`` becomes a positive label; every
    other outcome becomes a negative.  No model is involved in labelling.

    ``config`` should normally be **wider** than the shipped inference bound.  At
    the shipped bound the constraint already rejects essentially every bad
    candidate, so the training set is almost purely positive and the gate learns
    nothing (measured: AUC 0.50).  Training on a wider candidate set gives the
    gate real negatives, and it then acts as a genuine second line of defence for
    users whose glossaries collide more than synthetic ones do, or who loosen the
    bound deliberately.

    Returns ``(decisions, [(gate_probability, outcome), ...])`` -- the first for
    training, the second for the threshold sweep.

    Claim: LOW-DAMAGE.
    """
    cfg = config or CorrectorConfig(
        tau=tau, gate_threshold=0.0, reader=reader, keep_rejected=True
    )
    cfg = CorrectorConfig(**{**cfg.to_dict(), "gate_threshold": 0.0, "keep_rejected": True})
    corrector = ConstrainedCorrector(
        glossary, cfg, phonetic_config or DEFAULT_CONFIG, gate=gate
    )
    decisions: List[SpanDecision] = []
    scored: List[Tuple[float, str]] = []
    for pair in pairs:
        props = corrector.proposals(pair.hypothesis)
        for p in props:
            working = str(p["working_text"])
            outcome = classify_span_edit(
                pair.gold, working, (int(p["start"]), int(p["end"])), str(p["replacement"])
            )
            features = dict(p["features"])  # type: ignore[arg-type]
            decisions.append(
                SpanDecision(
                    features={k: float(v) for k, v in features.items()},
                    label=1 if outcome == "repair" else 0,
                    span_text=str(p["text"]),
                    candidate=str(p["replacement"]),
                    pair_id=pair.id,
                )
            )
            prob = corrector.gate.predict_proba_one(decisions[-1].features)
            scored.append((float(prob), outcome))
    return decisions, scored


@dataclass
class SweepRecord:
    """One document's worth of gated proposals, ready to be re-thresholded cheaply.

    Claim: LOW-DAMAGE -- lets the operating curve be traced over the *system's*
    behaviour without re-running retrieval a hundred times.
    """

    gold: str
    baseline: str
    terms: Tuple[str, ...]
    #: ``(probability, start, end, replacement, score)`` for every legal proposal.
    proposals: Tuple[Tuple[float, int, int, str, float], ...]


def collect_sweep_records(
    pairs: Sequence[ErrorPair],
    glossary: Glossary,
    tau: float = 0.28,
    gate: Optional[ConservativeGate] = None,
    reader: str = "auto",
    config: Optional[CorrectorConfig] = None,
) -> List[SweepRecord]:
    """Gather every legal proposal per document, with its gate probability.

    Claim: LOW-DAMAGE.
    """
    cfg = config or CorrectorConfig(tau=tau, gate_threshold=0.0, reader=reader)
    cfg = CorrectorConfig(**{**cfg.to_dict(), "gate_threshold": 0.0})
    corrector = ConstrainedCorrector(glossary, cfg, gate=gate)
    out: List[SweepRecord] = []
    for pair in pairs:
        props = corrector.proposals(pair.hypothesis)
        if not props:
            working = pair.hypothesis
            if cfg.remove_hallucinations:
                working, _ = corrector.hallucination_filter.apply(working)
            out.append(SweepRecord(pair.gold, working, tuple(pair.glossary_terms), ()))
            continue
        working = str(props[0]["working_text"])
        rows: List[Tuple[float, int, int, str, float]] = []
        for p in props:
            feats = {k: float(v) for k, v in dict(p["features"]).items()}  # type: ignore[arg-type]
            prob = float(corrector.gate.predict_proba_one(feats))
            nd = float(p["norm_distance"])  # type: ignore[arg-type]
            span_len = int(p["end"]) - int(p["start"])  # type: ignore[arg-type]
            score = prob + 0.5 * (1.0 - nd) + 0.10 * math_log1p(span_len)
            rows.append((prob, int(p["start"]), int(p["end"]), str(p["replacement"]), score))
        out.append(SweepRecord(pair.gold, working, tuple(pair.glossary_terms), tuple(rows)))
    return out


def math_log1p(x: float) -> float:
    """Local import-free ``log1p`` so this module has no math import ceremony.

    Claim: SUPPORT.
    """
    import math

    return math.log1p(max(0.0, x))


def apply_at_threshold(record: SweepRecord, threshold: float) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Re-run gating + overlap resolution + splicing at one threshold.

    Claim: LOW-DAMAGE.
    """
    accepted = [p for p in record.proposals if p[0] >= threshold]
    if not accepted:
        return record.baseline, []
    keep = select_non_overlapping([(a[1], a[2], a[4]) for a in accepted])
    chosen = [accepted[i] for i in keep]
    text = record.baseline
    edits: List[Tuple[int, int, str]] = []
    for _, s, e, repl, _sc in sorted(chosen, key=lambda a: -a[1]):
        text = text[:s] + repl + text[e:]
        edits.append((s, e, repl))
    return text, edits


def sweep_operating_points(
    records: Sequence[SweepRecord],
    thresholds: Optional[Sequence[float]] = None,
) -> List[SweepPoint]:
    """Trace correction rate against damage rate over the *whole* pipeline.

    The earlier proposal-level sweep (:func:`mondegreen.gate.sweep_thresholds`)
    counts every overlapping candidate separately and therefore badly overstates
    damage: 「量子誤り訂正」 and 「量子誤り訂正の話」 are two proposals but at most
    one becomes an edit.  This version re-gates, re-resolves overlaps and re-splices
    at every threshold, then measures term occurrences in the resulting text --
    which is the thing a user actually experiences.

    Denominators are term occurrences, fixed across the sweep:

    * correction rate = broken-in-ASR occurrences now correct / broken-in-ASR
    * damage rate     = correct-in-ASR occurrences now wrong / correct-in-ASR

    Claim: LOW-DAMAGE -- this function produces the second headline figure.
    """
    if thresholds is None:
        thresholds = [i / 100.0 for i in range(0, 101)]
    broken_total = 0
    correct_total = 0
    for r in records:
        for t in r.terms:
            gold_n = count_occurrences(r.gold, t)
            if not gold_n:
                continue
            base_n = min(count_occurrences(r.baseline, t), gold_n)
            correct_total += base_n
            broken_total += gold_n - base_n

    points: List[SweepPoint] = []
    for th in thresholds:
        fixed = damaged = 0
        edits = 0
        damaging_edits = 0
        repairing_edits = 0
        for r in records:
            text, applied = apply_at_threshold(r, th)
            edits += len(applied)
            for t in r.terms:
                gold_n = count_occurrences(r.gold, t)
                if not gold_n:
                    continue
                base_n = min(count_occurrences(r.baseline, t), gold_n)
                new_n = min(count_occurrences(text, t), gold_n)
                if new_n > base_n:
                    fixed += new_n - base_n
                elif new_n < base_n:
                    damaged += base_n - new_n
            for s, e, repl in applied:
                outcome = classify_span_edit(r.gold, r.baseline, (s, e), repl)
                if outcome == "repair":
                    repairing_edits += 1
                elif outcome == "damage":
                    damaging_edits += 1
        points.append(
            SweepPoint(
                threshold=float(th),
                correction_rate=(fixed / broken_total) if broken_total else 0.0,
                damage_rate=(damaged / correct_total) if correct_total else 0.0,
                edit_damage_rate=(damaging_edits / edits) if edits else 0.0,
                accepted=edits,
                repairs=repairing_edits,
                damages=damaging_edits,
                neutrals=max(0, edits - repairing_edits - damaging_edits),
            )
        )
    return points


def sweep_grid(
    pairs: Sequence[ErrorPair],
    glossary: Glossary,
    permissiveness: Sequence[float] = (0.2, 0.4, 0.8, 1.6, 3.2),
    gate: Optional[ConservativeGate] = None,
    reader: str = "auto",
    thresholds: Optional[Sequence[float]] = None,
    tau: float = 0.28,
) -> Dict[str, List[Dict[str, float]]]:
    """Trace the operating frontier over *both* knobs: the bound and the gate.

    The family parameter is ``max_raw_distance`` -- the absolute cap on weighted
    phonetic distance -- because that is the binding constraint once the
    length-normalised ``tau`` is set sensibly.  Sweeping ``tau`` alone produces a
    flat line, and a flat line would misrepresent where the safety actually comes
    from.

    Along each curve the gate threshold varies from 0 to 1.

    Reading the figure: moving up-and-right between curves is the *constraint*
    being loosened; moving along one curve is the *gate* being relaxed.  On this
    corpus almost all of the safety comes from the constraint -- which is the
    central design claim, drawn.

    Claim: LOW-DAMAGE -- this is the second headline figure.
    """
    out: Dict[str, List[Dict[str, float]]] = {}
    for raw in permissiveness:
        cfg = CorrectorConfig(
            tau=tau, max_raw_distance=raw, gate_threshold=0.0, reader=reader
        )
        records = collect_sweep_records(pairs, glossary, gate=gate, reader=reader, config=cfg)
        points = sweep_operating_points(records, thresholds=thresholds)
        out[f"{raw:.1f}"] = [p.to_dict() for p in points]
    return out


# --------------------------------------------------------------------------------------
# Glossary-size sweep construction
# --------------------------------------------------------------------------------------

def build_coverage_glossary(vocabulary: Glossary, size: int) -> Glossary:
    """The first ``size`` terms of a fixed, pre-shuffled vocabulary.

    This is the headline sweep's construction and it models the realistic
    question: *how much of my domain can I afford to declare?*  A bigger glossary
    covers more of what is actually said, so term recall should rise with size --
    for any method that can hold the terms.  Baseline (B) can hold about
    seventeen of them, forever.

    The vocabulary is shuffled once, outside this function, so that size-K
    glossaries are nested prefixes: every term in the 100-term glossary is also
    in the 1,000-term one.  Without nesting the sweep would confound size with
    content.

    Claim: UNBOUNDED-VOCAB -- this is the x-axis of the headline figure.
    """
    return Glossary(list(vocabulary)[:size])


def build_sized_glossary(
    targets: Glossary, distractors: Glossary, size: int, seed: int
) -> Glossary:
    """A size-``size`` glossary containing every target plus shuffled distractors.

    Shuffling matters: if targets always sat at the front, they would always fit
    in the 244-token prompt and baseline (B) would never degrade.  A real user's
    glossary is not sorted by relevance to the meeting they are about to have.

    Claim: UNBOUNDED-VOCAB -- this construction is the x-axis of the headline
    figure.
    """
    if size < len(targets):
        raise ValueError(f"glossary size {size} is smaller than the {len(targets)} target terms")
    pool = [e for e in distractors if e.surface not in targets]
    need = size - len(targets)
    if need > len(pool):
        raise ValueError(
            f"need {need} distractors but only {len(pool)} available; "
            "raise n_distractors in run_benchmark"
        )
    entries = list(targets) + pool[:need]
    random.Random(seed).shuffle(entries)
    return Glossary(entries)


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
    out_dir: str = "benchmarks/results",
    verbose: bool = True,
) -> Dict[str, object]:
    """Run conditions (A)-(E) over every glossary size and write the results.

    Claim: TERM-RECALL, LOW-DAMAGE, UNBOUNDED-VOCAB, LOCAL-SPEED -- all four, and
    the figures are generated from this function's output alone.
    """
    cfg = config or BenchmarkConfig()
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---- 1. vocabularies -------------------------------------------------
    max_size = max(max(cfg.glossary_sizes), max(cfg.control_sizes))
    builder = GlossaryBuilder(seed=cfg.seed)
    log(f"[1/6] building vocabularies (train {cfg.n_train_glossary}, "
        f"evaluation vocabulary {max_size})")
    train_glossary = builder.build(cfg.n_train_glossary, seed=cfg.seed)
    # One vocabulary, shuffled once, so every size-K glossary is a nested prefix.
    vocabulary_all = builder.build(max_size, seed=cfg.seed + 7919, exclude=train_glossary)
    ordered = list(vocabulary_all)
    random.Random(cfg.seed + 555).shuffle(ordered)
    vocabulary = Glossary(ordered)

    overlap = set(train_glossary.surfaces()) & set(vocabulary.surfaces())
    assert not overlap, f"train/eval glossary leak: {sorted(overlap)[:5]}"

    # ---- 2. corpora ------------------------------------------------------
    log(f"[2/6] generating corpora ({cfg.n_train_sentences} train, {cfg.n_sentences} eval)")
    train_sents = SentenceFactory(seed=cfg.seed).build(train_glossary, cfg.n_train_sentences)
    # Eval sentences draw terms from the *whole* vocabulary, so that a size-K
    # glossary covers roughly K/|V| of what is spoken.
    eval_sents = SentenceFactory(seed=cfg.seed + 31337).build(vocabulary, cfg.n_sentences)
    train_texts = {s for s, _ in train_sents}
    eval_sents = [(s, t) for s, t in eval_sents if s not in train_texts]

    harvester = ErrorHarvester(seed=cfg.seed)
    train_pairs = harvester.harvest_simulated(train_sents, train_glossary, split="train")
    eval_pairs = harvester.harvest_simulated(eval_sents, vocabulary, split="test")
    provenance = "simulated"

    # ---- 3. gate ---------------------------------------------------------
    if cfg.gate_path and os.path.exists(cfg.gate_path):
        log(f"[3/6] loading gate from {cfg.gate_path}")
        gate = ConservativeGate.load(cfg.gate_path)
        gate_report: Dict[str, object] = {"loaded_from": cfg.gate_path}
    else:
        log(f"[3/6] training gate on {len(train_pairs)} train pairs "
            f"(glossary disjoint from eval)")
        wide = CorrectorConfig(
            tau=cfg.gate_train_tau,
            max_raw_distance=cfg.gate_train_max_raw,
            tau_common_word=min(0.12, cfg.gate_train_tau),
            gate_threshold=0.0,
            reader=cfg.reader,
        )
        decisions, scored = collect_gate_training_data(
            train_pairs, train_glossary, reader=cfg.reader, config=wide
        )
        gate = ConservativeGate()
        rep = gate.fit(decisions)
        # Choose the operating threshold on the *wide* distribution, where damage
        # actually occurs.  Picking it on the shipped-bound distribution -- where
        # the constraint alone already reaches zero damage -- yields a threshold
        # of 0.0, i.e. a gate that never refuses anything.
        train_records = collect_sweep_records(
            train_pairs, train_glossary, gate=gate, reader=cfg.reader, config=wide
        )
        points = sweep_operating_points(train_records)
        gate.threshold = pick_threshold(points, cfg.max_damage_rate)
        gate_report = rep.to_dict()
        gate_report["chosen_threshold"] = gate.threshold
        gate_report["train_sweep"] = [p.to_dict() for p in points]
        gate_report["n_decisions"] = len(decisions)
        log(f"      AUC {rep.auc:.3f}  ECE {rep.ece:.3f}  "
            f"threshold {gate.threshold:.2f} (damage budget {cfg.max_damage_rate:.1%})")
        gate.save(os.path.join(out_dir, "gate.json"))

    refs = [p.gold for p in eval_pairs]
    raws = [p.hypothesis for p in eval_pairs]
    # Targets = every vocabulary term that actually occurs in the evaluation gold.
    spoken = {t for p in eval_pairs for t in p.glossary_terms}
    target_surfaces = [e.surface for e in vocabulary if e.surface in spoken]
    categories = {e.surface: e.category for e in vocabulary}
    log(f"      {len(target_surfaces)} distinct vocabulary terms are actually spoken "
        f"in the evaluation set")

    # ---- 4. conditions ---------------------------------------------------
    rows: List[Dict[str, object]] = []
    throughput: List[Dict[str, object]] = []
    ccfg_base = dict(tau=cfg.tau, gate_threshold=gate.threshold, reader=cfg.reader)

    def evaluate_sweep(
        sweep_name: str, sizes: Sequence[int], make_glossary,
        targets: Sequence[str],
    ) -> None:
        """Run conditions (A)-(E) across one sweep's glossary sizes.

                Claim: TERM-RECALL, LOW-DAMAGE, UNBOUNDED-VOCAB, LOCAL-SPEED.
                """
        base = summarize(refs, raws, raws, targets, categories,
                         cfg.hallucination_patterns)
        for size in sizes:
            rows.append({
                "condition": "A", "label": CONDITION_LABELS["A"], "glossary_size": size,
                "sweep": sweep_name, "provenance": provenance, "metrics": base,
            })
        for size in sizes:
            g = make_glossary(size)
            covered = sum(1 for t in targets if t in g)
            log(f"      [{sweep_name}] glossary {size} "
                f"(covers {covered}/{len(targets)} spoken terms)")

            cap = whisper_prompt_capacity(g)
            simB = SimulatedWhisperPrompt(g, cfg.simulation)
            outB, infoB = simB.run(eval_pairs)
            rows.append({
                "condition": "B", "label": CONDITION_LABELS["B"], "glossary_size": size,
                "sweep": sweep_name, "provenance": "simulated",
                "metrics": summarize(refs, raws, outB, targets, categories,
                                     cfg.hallucination_patterns),
                "extra": {**infoB,
                          "prompt_capacity": {k: v for k, v in cap.items() if k != "included"},
                          "glossary_covers_spoken_terms": covered,
                          "targets_in_prompt": sum(1 for t in targets
                                                   if t in set(cap["included"]))},
            })

            if cfg.include_cloud:
                from .baselines import AnthropicPostProcessor

                proc = AnthropicPostProcessor(g)
                outC = [proc(h) for h in raws]
                infoC: Dict[str, object] = {"condition": "C", "provenance": "measured",
                                            "model": proc.model}
                provC = "measured"
            else:
                simC = SimulatedCloudLLM(g, cfg.simulation)
                outC, infoC = simC.run(eval_pairs, cfg.hallucination_patterns)
                provC = "simulated"
            rows.append({
                "condition": "C", "label": CONDITION_LABELS["C"], "glossary_size": size,
                "sweep": sweep_name, "provenance": provC,
                "metrics": summarize(refs, raws, outC, targets, categories,
                                     cfg.hallucination_patterns),
                "extra": {**infoC, "glossary_covers_spoken_terms": covered},
            })

            ccfg = CorrectorConfig(**ccfg_base)
            corrector = ConstrainedCorrector(g, ccfg, gate=gate)
            t0 = time.perf_counter()
            resultsD = [corrector.correct(h) for h in raws]
            secD = time.perf_counter() - t0
            outD = [r.text for r in resultsD]
            outcomesD = _edit_outcomes(refs, raws, resultsD)
            rows.append({
                "condition": "D", "label": CONDITION_LABELS["D"], "glossary_size": size,
                "sweep": sweep_name, "provenance": provenance,
                "metrics": summarize(refs, raws, outD, targets, categories,
                                     cfg.hallucination_patterns, edit_outcomes=outcomesD),
                "extra": {
                    "seconds": secD, "index": corrector.index.stats(), "tau": cfg.tau,
                    "gate_threshold": gate.threshold,
                    "glossary_covers_spoken_terms": covered,
                    "edits": sum(len(r.corrections) for r in resultsD),
                    "hallucinations_removed": sum(len(r.removed_hallucinations) for r in resultsD),
                },
            })
            if sweep_name == "coverage":
                fresh = ConstrainedCorrector(g, ccfg, gate=gate)
                throughput.append(
                    benchmark_corrector(fresh, raws[: min(150, len(raws))],
                                        label=f"D@{size}").to_dict()
                )

            if cfg.quantized_model:
                from .runtime import build_reranker

                lm = build_reranker(cfg.quantized_model)
                corrE = ConstrainedCorrector(g, ccfg, gate=gate, lm=lm)
                resultsE = [corrE.correct(h) for h in raws]
                outE = [r.text for r in resultsE]
                outcomesE = _edit_outcomes(refs, raws, resultsE)
                extraE: Dict[str, object] = {"model": cfg.quantized_model,
                                             "lm": getattr(lm, "name", "?")}
                provE = "measured"
            else:
                outE, outcomesE = outD, outcomesD
                extraE = {
                    "note": "no quantised checkpoint supplied; (E) reported as (D) with "
                            "the LM re-ranking term absent -- which is quantisation's "
                            "worst case, bounded by tests/test_quantization.py. Pass "
                            "--quantized-model to measure the real number.",
                }
                provE = "not-measured"
            rows.append({
                "condition": "E", "label": CONDITION_LABELS["E"], "glossary_size": size,
                "sweep": sweep_name, "provenance": provE,
                "metrics": summarize(refs, raws, outE, targets, categories,
                                     cfg.hallucination_patterns, edit_outcomes=outcomesE),
                "extra": extraE,
            })

    log(f"[4/6] coverage sweep {list(cfg.glossary_sizes)} "
        f"(glossary grows to cover more of what is spoken)")
    evaluate_sweep("coverage", cfg.glossary_sizes,
                   lambda k: build_coverage_glossary(vocabulary, k), target_surfaces)

    log(f"      distractor control {list(cfg.control_sizes)} "
        f"(targets fixed, glossary padded with unrelated terms)")
    control_targets = Glossary([e for e in vocabulary if e.surface in spoken][: cfg.n_target_terms])
    # The pool must contain **no** spoken term, or padding the glossary would also
    # increase coverage and the control would stop controlling for anything.
    control_pool = Glossary([e for e in vocabulary if e.surface not in spoken])
    control_sizes = [k for k in cfg.control_sizes
                     if k <= len(control_targets) + len(control_pool)]
    if len(control_sizes) < len(cfg.control_sizes):
        log(f"      (control sizes clamped to {control_sizes}: only "
            f"{len(control_pool)} non-spoken distractors exist)")
    evaluate_sweep("distractor", control_sizes,
                   lambda k: build_sized_glossary(control_targets, control_pool, k,
                                                  seed=cfg.seed + k),
                   list(control_targets.surfaces()))

    # ---- 5. threshold sweep on the eval set ------------------------------
    log("[5/6] sweeping the gate threshold on held-out data")
    mid = max(cfg.glossary_sizes)
    g_mid = build_coverage_glossary(vocabulary, mid)
    eval_records = collect_sweep_records(
        eval_pairs, g_mid, tau=cfg.tau, gate=gate, reader=cfg.reader
    )
    eval_sweep = sweep_operating_points(eval_records)
    # 21 threshold steps is ample for a curve and five times cheaper than 101.
    eval_grid = sweep_grid(eval_pairs, g_mid, gate=gate, reader=cfg.reader, tau=cfg.tau,
                           thresholds=[i / 20 for i in range(21)])

    # ---- 6. retrieval sanity + write -------------------------------------
    log("[6/6] checking retrieval exactness and writing results")
    from .phonetics import kana_to_phonemes

    corrector_mid = ConstrainedCorrector(
        g_mid, CorrectorConfig(tau=cfg.tau, gate_threshold=gate.threshold, reader=cfg.reader),
        gate=gate,
    )
    probe_queries = [kana_to_phonemes(e.reading) for e in list(vocabulary)[:60]]
    retrieval = corrector_mid.index.retrieval_recall(probe_queries, cfg.tau)

    results: Dict[str, object] = {
        "config": cfg.to_dict(),
        "machine": machine_info(),
        "provenance": provenance,
        "provenance_note": (
            "Conditions marked 'simulated' come from mondegreen.simulate / "
            "mondegreen.baselines, whose stated assumptions are in config.simulation. "
            "Re-run scripts/harvest_errors.py --mode real and mondegreen bench --cloud "
            "to replace them with measurements."
        ),
        "separation": {
            "train_glossary_terms": len(train_glossary),
            "eval_vocabulary_terms": len(vocabulary),
            "eval_target_terms": len(target_surfaces),
            "glossary_overlap": 0,
            "train_sentences": len(train_pairs),
            "eval_sentences": len(eval_pairs),
            "sentence_overlap": 0,
            "eval_glossary_seen_in_training": False,
        },
        "gate": gate_report,
        "rows": rows,
        "eval_sweep": [p.to_dict() for p in eval_sweep],
        "eval_grid": eval_grid,
        "throughput": throughput,
        "retrieval_recall": retrieval,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
    results["summary"] = _summarize_rows(rows, eval_sweep, throughput)

    stamp = f"{provenance}"
    path = os.path.join(out_dir, f"benchmark.{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    log(f"wrote {path}  ({results['elapsed_seconds']:.1f}s)")
    return results


def _edit_outcomes(refs, raws, results) -> List[List[str]]:
    """Label every applied edit as repair / damage / neutral.

    Claim: LOW-DAMAGE -- metric (3)'s edit-level denominator.
    """
    out: List[List[str]] = []
    for ref, raw, res in zip(refs, raws, results):
        labels: List[str] = []
        for c in res.corrections:
            labels.append(classify_span_edit(ref, res.source, (c.start, c.end), c.replacement))
        out.append(labels)
    return out


def _summarize_rows(
    rows: Sequence[Dict[str, object]],
    eval_sweep: Sequence[SweepPoint],
    throughput: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Condense the full result set into the numbers that go in the README.

    Claim: SUPPORT.
    """
    table: Dict[str, Dict[int, Dict[str, float]]] = {}
    for r in rows:
        cond = str(r["condition"])
        if str(r.get("sweep", "coverage")) != "coverage":
            continue
        size = int(r["glossary_size"])  # type: ignore[arg-type]
        m = r["metrics"]  # type: ignore[assignment]
        table.setdefault(cond, {})[size] = {
            "cer": float(m["cer"]),                     # type: ignore[index]
            "wer": float(m["wer"]),                     # type: ignore[index]
            "term_recall": float(m["term_recall"]["recall"]),        # type: ignore[index]
            "damage_rate_chars": float(m["damage"]["damage_rate_chars"]),  # type: ignore[index]
            "damage_rate_terms": float(m["damage"]["damage_rate_terms"]),  # type: ignore[index]
            "damage_rate_edits": float(m["damage"]["damage_rate_edits"]),  # type: ignore[index]
            "hallucination_removal": float(
                m.get("hallucination", {}).get("removal_rate", 0.0)   # type: ignore[union-attr]
            ),
            "provenance": r["provenance"],
        }
    best = min(
        (p for p in eval_sweep if p.correction_rate > 0),
        key=lambda p: (p.damage_rate, -p.correction_rate),
        default=None,
    )
    return {
        "by_condition": table,
        "operating_point": best.to_dict() if best else None,
        "throughput": throughput[-1] if throughput else None,
    }
