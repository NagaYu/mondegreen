"""``mondegreen`` command line.

The headline command is the one from the brief::

    mondegreen fix transcript.txt --glossary terms.csv

Everything else exists to make that command's behaviour inspectable: ``explain``
shows the phoneme-level receipt for each edit, ``sweep`` draws the correction-rate
vs damage-rate curve on your own data, ``info`` says which backends are actually
in use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

from . import __version__


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _read_text(path: str) -> str:
    """Read a transcript from a path, or from stdin when given ``-``.

        Claim: SUPPORT.
        """
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _write_text(path: Optional[str], text: str) -> None:
    """Write to a path, or stdout when given ``-``/nothing.

        Claim: SUPPORT.
        """
    if not path or path == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _load_pieces(args) -> tuple:
    """Assemble glossary, gate, configs and optional LM from parsed arguments.

        Claim: SUPPORT -- one construction path so every subcommand behaves alike.
        """
    from .corrector import ConstrainedCorrector, CorrectorConfig
    from .gate import ConservativeGate
    from .glossary import Glossary, load_glossary
    from .phonetics import PhoneticConfig

    glossary = load_glossary(args.glossary) if args.glossary else Glossary()
    gate = ConservativeGate.load(args.gate) if getattr(args, "gate", None) else None
    pcfg = PhoneticConfig()
    if getattr(args, "phonetic_config", None):
        with open(args.phonetic_config, encoding="utf-8") as fh:
            pcfg = PhoneticConfig.from_dict(json.load(fh))
    cfg = CorrectorConfig(
        tau=args.tau,
        gate_threshold=args.gate_threshold,
        remove_hallucinations=not getattr(args, "keep_hallucinations", False),
        reader=getattr(args, "reader", "auto"),
        max_span_tokens=getattr(args, "max_span_tokens", 8),
    )
    lm = None
    if getattr(args, "lm", None):
        from .runtime import build_reranker

        lm = build_reranker(args.lm)
    corrector = ConstrainedCorrector(glossary, cfg, pcfg, gate=gate, lm=lm)
    return glossary, corrector


def _diff_lines(result) -> List[str]:
    """Render each applied edit as a coloured diff line with its phonetic receipt.

        Claim: LOW-DAMAGE -- the terminal output shows the evidence, not just the result.
        """
    out: List[str] = []
    for c in result.corrections:
        out.append(
            f"  \033[31m- {c.original}\033[0m  ->  \033[32m+ {c.replacement}\033[0m   "
            f"[{' '.join(c.original_phonemes)}] ~ [{' '.join(c.candidate_phonemes)}]   "
            f"d={c.norm_distance:.3f} <= tau={c.threshold:.2f}   p={c.gate_prob:.3f}"
        )
    for s, e, txt in result.removed_hallucinations:
        out.append(f"  \033[31m- {txt}\033[0m   [hallucination removed]")
    return out


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------

def cmd_fix(args) -> int:
    """``mondegreen fix`` -- correct a transcript against a glossary.

    Claim: TERM-RECALL + LOW-DAMAGE + LOCAL-SPEED -- the whole product in one
    command, offline.
    """
    glossary, corrector = _load_pieces(args)
    text = _read_text(args.transcript)
    t0 = time.perf_counter()
    result = corrector.correct(text)
    elapsed = time.perf_counter() - t0

    if args.json:
        _write_text(args.output, json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    _write_text(args.output, result.text)
    if args.output and args.output != "-":
        print(f"wrote {args.output}", file=sys.stderr)
    if not args.quiet:
        n = len(result.corrections)
        print(
            f"\n{n} correction{'' if n == 1 else 's'}, "
            f"{len(result.removed_hallucinations)} hallucination(s) removed, "
            f"{len(glossary)} glossary terms, {elapsed * 1000:.0f} ms",
            file=sys.stderr,
        )
        for line in _diff_lines(result):
            print(line, file=sys.stderr)
    return 0


def cmd_explain(args) -> int:
    """``mondegreen explain`` -- per-span evidence, accepted and rejected.

    Claim: LOW-DAMAGE -- shows why each edit cleared (or failed) the bound.
    """
    _, corrector = _load_pieces(args)
    rows = corrector.explain(_read_text(args.transcript))
    if args.json:
        _write_text(args.output, json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    lines: List[str] = []
    for r in rows:
        mark = "ACCEPT" if r["accepted"] else "reject"
        lines.append(
            f"[{mark}] {r['original']} -> {r['replacement']}\n"
            f"    span phonemes : {r['original_phonemes']}\n"
            f"    term phonemes : {r['candidate_phonemes']}\n"
            f"    distance      : {r['norm_distance']:.4f}  (threshold {r['threshold']:.2f}, "
            f"raw {r['distance']:.3f})\n"
            f"    gate          : p={r['gate_prob']:.4f}  margin={r['margin']:.3f}  {r['reason']}\n"
            f"    alignment     : "
            + " ".join(
                f"{o['op']}{o['from'] or '-'}/{o['to'] or '-'}" for o in r["alignment"]
            )
        )
    _write_text(args.output, "\n".join(lines) if lines else "(no candidate spans)")
    return 0


def cmd_build_glossary(args) -> int:
    """``mondegreen build-glossary`` -- synthesise a glossary with readings.

    Claim: UNBOUNDED-VOCAB.
    """
    from .glossary import save_glossary
    from .harvest import GlossaryBuilder

    gb = GlossaryBuilder(seed=args.seed)
    if args.test_out:
        train, test = gb.build_pair(args.n, args.n_test)
        save_glossary(train, args.output)
        save_glossary(test, args.test_out)
        print(f"wrote {len(train)} train terms -> {args.output}")
        print(f"wrote {len(test)} disjoint test terms -> {args.test_out}")
    else:
        g = gb.build(args.n)
        save_glossary(g, args.output)
        print(f"wrote {len(g)} terms -> {args.output}")
    return 0


def cmd_harvest(args) -> int:
    """``mondegreen harvest`` -- build an (error, gold) dataset.

    Claim: SUPPORT.
    """
    from .glossary import load_glossary
    from .harvest import (
        ErrorHarvester, SentenceFactory, build_asr, build_tts, write_jsonl,
    )

    glossary = load_glossary(args.glossary)
    sentences = SentenceFactory(seed=args.seed).build(glossary, args.n)
    tts = asr = None
    if args.mode == "real":
        tts = build_tts(args.tts)
        asr = build_asr(args.whisper_size, args.asr)
    harvester = ErrorHarvester(tts=tts, asr=asr, seed=args.seed)
    if args.mode == "real":
        pairs = harvester.harvest_real(sentences, glossary, split=args.split, corpus=args.corpus,
                                       asr_model=getattr(asr, "name", "whisper"))
        pairs += harvester.harvest_hallucinations(split=args.split, corpus=args.corpus)
    else:
        pairs = harvester.harvest_simulated(sentences, glossary, split=args.split, corpus=args.corpus)
    write_jsonl(pairs, args.output)
    print(f"wrote {len(pairs)} pairs ({args.mode}) -> {args.output}")
    return 0


def cmd_train_gate(args) -> int:
    """``mondegreen train-gate`` -- fit and calibrate the conservative gate.

    Claim: LOW-DAMAGE.
    """
    from .benchmark import collect_gate_training_data, collect_sweep_records, sweep_operating_points
    from .gate import ConservativeGate, pick_threshold
    from .glossary import load_glossary
    from .harvest import read_jsonl

    pairs = read_jsonl(args.pairs)
    glossary = load_glossary(args.glossary)
    from .corrector import CorrectorConfig

    wide = CorrectorConfig(tau=max(args.tau, 0.45), max_raw_distance=1.2,
                           tau_common_word=0.12, gate_threshold=0.0)
    decisions, _ = collect_gate_training_data(pairs, glossary, config=wide)
    if not decisions:
        print("no labelled span decisions were produced; is the glossary right?", file=sys.stderr)
        return 1
    gate = ConservativeGate()
    report = gate.fit(decisions, class_weight_negative=args.negative_weight)
    records = collect_sweep_records(pairs, glossary, tau=args.tau, gate=gate)
    points = sweep_operating_points(records)
    gate.threshold = pick_threshold(points, args.max_damage_rate)
    gate.metadata["sweep"] = [p.to_dict() for p in points]
    gate.metadata["max_damage_rate"] = args.max_damage_rate
    gate.save(args.output)
    print(f"trained on {report.n_train} spans "
          f"(AUC {report.auc:.3f}, ECE {report.ece:.3f}), "
          f"threshold {gate.threshold:.2f} for damage <= {args.max_damage_rate:.1%}")
    print(f"wrote {args.output}")
    return 0


def cmd_bench(args) -> int:
    """``mondegreen bench`` -- run conditions (A)-(E) and write results JSON.

    Claim: TERM-RECALL, LOW-DAMAGE, UNBOUNDED-VOCAB, LOCAL-SPEED.
    """
    from .benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        glossary_sizes=tuple(args.glossary_sizes),
        tau=args.tau,
        gate_path=args.gate,
        n_sentences=args.n,
        seed=args.seed,
        include_cloud=args.cloud,
        quantized_model=args.quantized_model,
    )
    results = run_benchmark(cfg, out_dir=args.output)
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    return 0


def cmd_sweep(args) -> int:
    """``mondegreen sweep`` -- correction rate vs damage rate on your own data.

    Claim: LOW-DAMAGE.
    """
    from .benchmark import collect_sweep_records, sweep_operating_points
    from .gate import ConservativeGate
    from .glossary import load_glossary
    from .harvest import read_jsonl

    pairs = read_jsonl(args.pairs)
    glossary = load_glossary(args.glossary)
    gate = ConservativeGate.load(args.gate) if args.gate else ConservativeGate()
    records = collect_sweep_records(pairs, glossary, tau=args.tau, gate=gate)
    points = sweep_operating_points(records)
    rows = [p.to_dict() for p in points]
    if args.json:
        _write_text(args.output, json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"{'thr':>6} {'correction':>11} {'damage':>8} {'edit-dmg':>9} {'accepted':>9}")
    for p in points[::5]:
        print(f"{p.threshold:6.2f} {p.correction_rate:11.3f} {p.damage_rate:8.4f} "
              f"{p.edit_damage_rate:9.4f} {p.accepted:9d}")
    return 0


def cmd_info(args) -> int:
    """``mondegreen info`` -- which backends are actually installed and in use.

    Claim: SUPPORT -- results are not interpretable without this.
    """
    from .reading import get_reader
    from .runtime import machine_info

    info: Dict[str, object] = {"version": __version__, "machine": machine_info()}
    try:
        info["reader"] = get_reader().name
    except Exception as exc:
        info["reader"] = f"unavailable: {exc}"
    backends: Dict[str, bool] = {}
    for mod in ("pyopenjtalk", "fugashi", "pykakasi", "faster_whisper", "whisper",
                "torch", "transformers", "peft", "trl", "outlines", "xgrammar",
                "llama_cpp", "mlx_lm", "gradio", "datasets", "matplotlib"):
        try:
            __import__(mod)
            backends[mod] = True
        except Exception:
            backends[mod] = False
    info["backends"] = backends
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser.

    Claim: SUPPORT.
    """
    p = argparse.ArgumentParser(
        prog="mondegreen",
        description="Local ASR post-correction under a hard phonetic constraint.",
    )
    p.add_argument("--version", action="version", version=f"mondegreen {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        """Attach the flags shared by ``fix`` and ``explain``.

                Claim: SUPPORT.
                """
        sp.add_argument("--glossary", "-g", help="CSV/TSV/JSON glossary (surface,reading,...)")
        sp.add_argument("--gate", help="trained gate JSON (mondegreen train-gate)")
        sp.add_argument("--tau", type=float, default=0.28,
                        help="hard phonetic threshold; replacements above it are impossible")
        sp.add_argument("--gate-threshold", type=float, default=0.5)
        sp.add_argument("--reader", default="auto", choices=["auto", "pyopenjtalk", "fugashi", "fallback"])
        sp.add_argument("--max-span-tokens", type=int, default=8)
        sp.add_argument("--phonetic-config", help="JSON PhoneticConfig override")
        sp.add_argument("--lm", help="GGUF file or MLX dir used to re-rank inside the candidate set")
        sp.add_argument("--keep-hallucinations", action="store_true")
        sp.add_argument("--output", "-o", default="-")
        sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("fix", help="correct a transcript")
    sp.add_argument("transcript", help="path, or - for stdin")
    sp.add_argument("--quiet", "-q", action="store_true")
    add_common(sp)
    sp.set_defaults(func=cmd_fix)

    sp = sub.add_parser("explain", help="show the evidence behind every candidate edit")
    sp.add_argument("transcript")
    add_common(sp)
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("build-glossary", help="synthesise a glossary with readings")
    sp.add_argument("-n", type=int, default=1000)
    sp.add_argument("--n-test", type=int, default=200)
    sp.add_argument("--test-out", help="also write a strictly disjoint test glossary here")
    sp.add_argument("--seed", type=int, default=20260823)
    sp.add_argument("--output", "-o", default="data/glossary.csv")
    sp.set_defaults(func=cmd_build_glossary)

    sp = sub.add_parser("harvest", help="build an (error, gold) dataset")
    sp.add_argument("--glossary", "-g", required=True)
    sp.add_argument("-n", type=int, default=500)
    sp.add_argument("--mode", choices=["simulated", "real"], default="simulated")
    sp.add_argument("--split", default="train")
    sp.add_argument("--corpus", default="synthetic")
    sp.add_argument("--tts", default="auto")
    sp.add_argument("--asr", default="auto")
    sp.add_argument("--whisper-size", default="small")
    sp.add_argument("--seed", type=int, default=20260823)
    sp.add_argument("--output", "-o", default="data/pairs.jsonl")
    sp.set_defaults(func=cmd_harvest)

    sp = sub.add_parser("train-gate", help="fit and calibrate the conservative gate")
    sp.add_argument("--pairs", required=True)
    sp.add_argument("--glossary", "-g", required=True)
    sp.add_argument("--tau", type=float, default=0.28)
    sp.add_argument("--negative-weight", type=float, default=3.0)
    sp.add_argument("--max-damage-rate", type=float, default=0.01)
    sp.add_argument("--output", "-o", default="models/gate.json")
    sp.set_defaults(func=cmd_train_gate)

    sp = sub.add_parser("bench", help="run conditions (A)-(E)")
    sp.add_argument("--glossary-sizes", type=int, nargs="+",
                    default=[10, 30, 100, 300, 1000, 3000, 10000])
    sp.add_argument("-n", type=int, default=600)
    sp.add_argument("--tau", type=float, default=0.28)
    sp.add_argument("--gate")
    sp.add_argument("--cloud", action="store_true", help="call a real cloud LLM for condition (C)")
    sp.add_argument("--quantized-model", help="GGUF/MLX path for condition (E)")
    sp.add_argument("--seed", type=int, default=20260823)
    sp.add_argument("--output", "-o", default="benchmarks/results")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("sweep", help="correction rate vs damage rate")
    sp.add_argument("--pairs", required=True)
    sp.add_argument("--glossary", "-g", required=True)
    sp.add_argument("--gate")
    sp.add_argument("--tau", type=float, default=0.28)
    sp.add_argument("--output", "-o", default="-")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("info", help="report installed backends and machine")
    sp.set_defaults(func=cmd_info)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``mondegreen`` console script.

    Claim: LOCAL-SPEED -- one command, no network, no API key.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except BrokenPipeError:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
