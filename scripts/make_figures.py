#!/usr/bin/env python3
"""Render the figures from a benchmark results JSON file.

    python scripts/make_figures.py benchmarks/results/benchmark.simulated.json

Figures built from simulated data are watermarked SIMULATED. That is deliberate:
a plot without its provenance attached will eventually be quoted as a measurement.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mondegreen.figures import load_results, make_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", help="benchmark JSON (default: newest in benchmarks/results)")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    path = args.results
    if not path:
        candidates = sorted(glob.glob("benchmarks/results/benchmark.*.json"), key=os.path.getmtime)
        if not candidates:
            print("no benchmark results found; run scripts/run_benchmarks.py first", file=sys.stderr)
            return 1
        path = candidates[-1]
    print(f"reading {path}")
    results = load_results(path)
    for p in make_all(results, args.out):
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
