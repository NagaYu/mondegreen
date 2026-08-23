#!/usr/bin/env python3
"""Run conditions (A)-(E) and write the results and figures.

    (A) raw Whisper
    (B) Whisper initial_prompt stuffed with the glossary  [244-token ceiling]
    (C) cloud LLM post-processing
    (D) Mondegreen  (phonetic constraint + conservative gate)
    (E) Mondegreen after Q4_K_M quantisation

Across glossary sizes 100 / 1,000 / 10,000, with training and evaluation
separated by speaker, source sentence and glossary.

Example
-------
    python scripts/run_benchmarks.py --n 500 --figures
    python scripts/run_benchmarks.py --cloud            # real (C), needs an API key
    python scripts/run_benchmarks.py --quantized-model models/quantized/mondegreen-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mondegreen.benchmark import BenchmarkConfig, run_benchmark


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glossary-sizes", type=int, nargs="+",
                    default=[10, 30, 100, 300, 1000, 3000, 10000],
                    help="coverage sweep sizes; starts below the ~17-term prompt ceiling")
    ap.add_argument("--control-sizes", type=int, nargs="+", default=[100, 1000, 10000],
                    help="distractor control sweep sizes")
    ap.add_argument("-n", "--n-sentences", type=int, default=500)
    ap.add_argument("--target-terms", type=int, default=100)
    ap.add_argument("--train-sentences", type=int, default=700)
    ap.add_argument("--tau", type=float, default=0.28)
    ap.add_argument("--max-damage-rate", type=float, default=0.01)
    ap.add_argument("--gate", help="reuse an existing gate instead of training one")
    ap.add_argument("--cloud", action="store_true", help="call a real cloud LLM for (C)")
    ap.add_argument("--quantized-model", help="GGUF/MLX path for a measured (E)")
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--figures", action="store_true", help="also render figures/")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    cfg = BenchmarkConfig(
        glossary_sizes=tuple(args.glossary_sizes),
        control_sizes=tuple(args.control_sizes),
        n_sentences=args.n_sentences,
        n_target_terms=args.target_terms,
        n_train_sentences=args.train_sentences,
        tau=args.tau,
        max_damage_rate=args.max_damage_rate,
        gate_path=args.gate,
        include_cloud=args.cloud,
        quantized_model=args.quantized_model,
        seed=args.seed,
    )
    results = run_benchmark(cfg, out_dir=args.out)

    print("\n" + "=" * 96)
    print(f"{'cond':<5}{'glossary':>10}{'CER':>9}{'WER':>9}{'term recall':>13}"
          f"{'damage(ch)':>12}{'damage(term)':>14}{'provenance':>14}")
    print("-" * 96)
    for cond in ("A", "B", "C", "D", "E"):
        table = results["summary"]["by_condition"].get(cond, {})
        for size in sorted(table):
            m = table[size]
            print(f"{cond:<5}{size:>10,}{m['cer']:>9.4f}{m['wer']:>9.4f}"
                  f"{m['term_recall']:>13.3f}{m['damage_rate_chars']:>12.5f}"
                  f"{m['damage_rate_terms']:>14.5f}{m['provenance']:>14}")
    print("=" * 96)
    op = results["summary"].get("operating_point")
    if op:
        print(f"operating point: gate={op['threshold']:.2f} -> "
              f"correction {op['correction_rate']:.1%} at damage {op['damage_rate']:.3%}")
    tp = results["summary"].get("throughput")
    if tp:
        print(f"local throughput: {tp['chars_per_second']:.0f} chars/s "
              f"({tp['seconds_per_hour_of_audio']:.0f}s per hour of audio, "
              f"{tp['peak_rss_mb']:.0f} MB peak)")
    print(f"retrieval recall vs exhaustive scan: {results['retrieval_recall']['recall']:.4f}")

    if args.figures:
        from mondegreen.figures import make_all

        paths = make_all(results, args.figures_dir)
        print("\nfigures:")
        for p in paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
