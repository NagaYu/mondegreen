#!/usr/bin/env python3
"""Merge the LoRA adapter and export GGUF (Q4_K_M, Q8_0) and MLX.

    LoRA adapter -> merged fp16 checkpoint -> {GGUF Q4_K_M, GGUF Q8_0, MLX 4-bit}

Then measure, on this machine: throughput, peak memory, and how long an hour of
transcription actually takes.

Requirements
------------
GGUF needs a llama.cpp checkout (``git clone https://github.com/ggml-org/llama.cpp``
and ``export LLAMA_CPP_DIR=...``, built with ``cmake -B build && cmake --build build -j``).
MLX needs ``pip install mlx-lm`` and Apple silicon.  Each step reports the exact
command it ran, and a missing tool is reported rather than silently skipped.

Example
-------
    python scripts/quantize.py --base Qwen/Qwen2.5-0.5B --adapter models/lora \\
        --out models/quantized --benchmark --glossary data/glossary_test.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mondegreen import __version__
from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig
from mondegreen.glossary import load_glossary
from mondegreen.harvest import SentenceFactory, ErrorHarvester
from mondegreen.runtime import (
    CHARS_PER_HOUR_OF_AUDIO, benchmark_corrector, compare_to_cloud, export_gguf,
    export_mlx, machine_info, merge_lora,
)


def model_card(args, exports: List[Dict[str, object]], bench: Dict[str, object]) -> str:
    """HF model card for the published artefacts.

    Claim: SUPPORT -- LOCAL-SPEED numbers are meaningless without the machine.
    """
    rows = "\n".join(
        f"| `{e['kind']}` | {'ok' if e['ok'] else 'FAILED'} | "
        f"{e['size_mb']:.0f} MB | `{str(e.get('command', ''))[:90]}` |"
        for e in exports
    )
    m = bench.get("machine", {}) if bench else {}
    perf = ""
    if bench:
        perf = f"""
## Measured on this machine

| | |
| --- | --- |
| machine | {m.get('cpu_brand', m.get('platform'))} , {m.get('memory_gb', '?')} GB |
| throughput | **{bench['chars_per_second']:.0f} characters/second** |
| 1 hour of audio | **{bench['seconds_per_hour_of_audio']:.0f} seconds** (1 h ≈ {CHARS_PER_HOUR_OF_AUDIO:,} chars) |
| peak memory | {bench['peak_rss_mb']:.0f} MB |
| glossary | {bench['glossary_terms']:,} terms |
| network | none — nothing leaves the machine |
"""
    return f"""---
license: apache-2.0
language:
- ja
library_name: transformers
tags:
- asr-error-correction
- japanese
- gguf
- mlx
- quantized
base_model: {args.base}
---

# Mondegreen re-ranker (`v{__version__}`)

A small LoRA-tuned LM used by [Mondegreen](https://github.com/mondegreen/mondegreen)
to **re-rank candidates inside a hard phonetic constraint**.

## What this model does and does not do

It does **not** decide what a span may become. That is decided by
`mondegreen.index.PhoneticIndex`, which computes a finite candidate list under a
weighted phonetic edit-distance bound. This model only orders that list when it
contains more than one entry.

Consequently:

- it cannot introduce a term that is not in your glossary;
- it cannot rewrite grammar, drop hedges or normalise numbers;
- removing it entirely degrades term recall by **less than 2 points**
  (asserted by `tests/test_quantization.py`), which is why 4-bit quantisation is
  safe here.

On synthetic glossaries only ~1% of spans have more than one legal candidate at
the shipped bound, so most of the time this model is not consulted at all.

## Artefacts

| artefact | status | size | command |
| --- | --- | --- | --- |
{rows}
{perf}
## Use

```python
from mondegreen import ConstrainedCorrector, load_glossary
from mondegreen.runtime import build_reranker

corrector = ConstrainedCorrector(
    load_glossary("terms.csv"),
    lm=build_reranker("mondegreen-Q4_K_M.gguf"),
)
print(corrector.correct(transcript).text)
```

## Privacy

Your glossary and your transcripts stay on your machine. The whole point of this
project is that the audio that produces these errors is usually audio you are not
allowed to upload.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--adapter", default="models/lora")
    ap.add_argument("--out", default="models/quantized")
    ap.add_argument("--merged", default=None, help="where to write the merged fp16 model")
    ap.add_argument("--quantizations", nargs="+", default=["Q4_K_M", "Q8_0"])
    ap.add_argument("--mlx-bits", type=int, default=4)
    ap.add_argument("--skip-merge", action="store_true", help="--adapter is already merged")
    ap.add_argument("--skip-gguf", action="store_true")
    ap.add_argument("--skip-mlx", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="measure local throughput and memory after exporting")
    ap.add_argument("--glossary", help="glossary CSV to benchmark against")
    ap.add_argument("--bench-sentences", type=int, default=200)
    ap.add_argument("--push-to", help="HF model repo id")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    merged = args.merged or os.path.join(args.out, "merged-fp16")
    exports: List[Dict[str, object]] = []

    if args.skip_merge:
        merged = args.adapter
        print(f"[1/4] skipping merge, using {merged}")
    else:
        print(f"[1/4] merging {args.adapter} into {args.base}")
        r = merge_lora(args.base, args.adapter, merged)
        exports.append(r.to_dict())
        print(f"      {'ok' if r.ok else 'FAILED: ' + r.message}")
        if not r.ok:
            return 1

    if args.skip_gguf:
        print("[2/4] skipping GGUF")
    else:
        print(f"[2/4] GGUF export ({', '.join(args.quantizations)})")
        for r in export_gguf(merged, args.out, args.quantizations):
            exports.append(r.to_dict())
            print(f"      {r.kind}: {'ok' if r.ok else 'FAILED'} "
                  f"{r.size_mb:.0f}MB {'' if r.ok else '- ' + r.message[:200]}")

    if args.skip_mlx:
        print("[3/4] skipping MLX")
    else:
        print(f"[3/4] MLX export ({args.mlx_bits}-bit)")
        r = export_mlx(merged, os.path.join(args.out, "mlx"), args.mlx_bits)
        exports.append(r.to_dict())
        print(f"      {'ok' if r.ok else 'FAILED: ' + r.message[:200]}")

    bench: Dict[str, object] = {}
    if args.benchmark:
        print("[4/4] measuring local throughput")
        if not args.glossary:
            print("      --benchmark needs --glossary", file=sys.stderr)
            return 1
        glossary = load_glossary(args.glossary)
        sents = SentenceFactory(seed=7).build(glossary, args.bench_sentences)
        pairs = ErrorHarvester(seed=7).harvest_simulated(sents, glossary, split="test")
        texts = [p.hypothesis for p in pairs]
        corrector = ConstrainedCorrector(glossary, CorrectorConfig())
        res = benchmark_corrector(corrector, texts, label="mondegreen-local")
        bench = res.to_dict()
        print(f"      {res.chars_per_second:.0f} chars/s | "
              f"{res.seconds_per_hour_of_audio:.0f}s per hour of audio | "
              f"{res.peak_rss_mb:.0f} MB peak")
        cloud = compare_to_cloud(res, n_requests=len(texts))
        bench["cloud_comparison"] = cloud
        print(f"      vs cloud: {cloud['speedup']:.1f}x, "
              f"and applicable to confidential audio: local=True cloud=False")
    else:
        print("[4/4] skipping benchmark (pass --benchmark --glossary ...)")

    report = {"exports": exports, "benchmark": bench, "machine": machine_info()}
    with open(os.path.join(args.out, "export_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    card = model_card(args, exports, bench)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    print(f"wrote {args.out}/export_report.json and README.md")

    if args.push_to:
        try:
            from huggingface_hub import HfApi  # type: ignore
        except ImportError:
            print("  need: pip install huggingface_hub", file=sys.stderr)
            return 1
        print(f"pushing to https://huggingface.co/{args.push_to}")
        api = HfApi()
        api.create_repo(args.push_to, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push_to,
                          ignore_patterns=["merged-fp16/*", "*-f16.gguf"])
        print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
