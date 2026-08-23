#!/usr/bin/env python3
"""Publish the Model, Dataset and Space to the Hugging Face Hub.

Three repos, each self-contained:

``model``    the trained conservative gate, the LoRA adapter, and the quantised
             GGUF / MLX exports, with a card that is explicit about what the LM
             does (re-rank inside the constraint) and what it cannot do (widen it).
``dataset``  the ``(ASR hypothesis, gold text)`` pairs, the two disjoint
             glossaries, and the pathology-label taxonomy.
``space``    the Gradio evidence viewer, with the package vendored so the Space
             does not depend on PyPI or on GitHub staying up.

Nothing is uploaded without a licence field, and nothing that could contain a
real person's name is uploaded at all -- every glossary here is synthetic.

    python scripts/publish_hf.py --user NagaYu --what model dataset space
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mondegreen import __version__

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _api():
    """Authenticated Hub client, or a clear error.

    Claim: SUPPORT.
    """
    from huggingface_hub import HfApi, whoami

    try:
        who = whoami()
    except Exception as exc:
        raise SystemExit(
            "not logged in to Hugging Face. Run `huggingface-cli login` first.\n"
            f"({type(exc).__name__}: {exc})"
        )
    return HfApi(), who


def _bench() -> Dict[str, object]:
    """Load the newest benchmark result, for the cards.

    Claim: SUPPORT -- a model card without its numbers is an advertisement.
    """
    import glob

    files = sorted(
        glob.glob(os.path.join(REPO_ROOT, "benchmarks/results/benchmark.*.json")),
        key=os.path.getmtime,
    )
    if not files:
        return {}
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


def _cond_table(bench: Dict[str, object], size: int = 10000) -> str:
    """Render the (A)-(E) table for a card.

    Claim: SUPPORT.
    """
    by = bench.get("summary", {}).get("by_condition", {})  # type: ignore[union-attr]
    names = {
        "A": "(A) raw Whisper", "B": "(B) Whisper `initial_prompt`",
        "C": "(C) cloud LLM post-processing", "D": "**(D) Mondegreen**",
        "E": "(E) Mondegreen, quantised",
    }
    rows = []
    for c in "ABCDE":
        m = {int(k): v for k, v in by.get(c, {}).items()}.get(size)
        if not m:
            continue
        rows.append(
            f"| {names[c]} | {m['cer']:.4f} | {m['wer']:.4f} | "
            f"{m['term_recall']:.1%} | {m['damage_rate_chars']:.5f} |"
        )
    if not rows:
        return "_(no benchmark results found; run `make bench`)_"
    return (
        "| condition | CER | WER | term recall | **damage rate** |\n"
        "| --- | ---: | ---: | ---: | ---: |\n" + "\n".join(rows)
    )


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

def model_card(user: str, bench: Dict[str, object], artefacts: List[str]) -> str:
    """The model card.

    Claim: SUPPORT.
    """
    tp = (bench.get("summary", {}) or {}).get("throughput") or {}
    machine = bench.get("machine", {}) or {}
    gate = bench.get("gate", {}) or {}
    have = "\n".join(f"- `{a}`" for a in artefacts) or "- _(none built yet)_"
    perf = ""
    if tp:
        perf = (
            f"| throughput | **{tp['chars_per_second']:.0f} characters/second** "
            f"(10,000-term glossary) |\n"
            f"| 1 hour of transcription | **{tp['seconds_per_hour_of_audio']:.0f} seconds** |\n"
            f"| peak memory | {tp['peak_rss_mb']:.0f} MB |\n"
            f"| machine | {machine.get('cpu_brand', machine.get('platform', '?'))}, "
            f"{machine.get('memory_gb', '?')} GB |\n"
            f"| network | **none** |\n"
        )
    return f"""---
license: apache-2.0
language:
- ja
library_name: transformers
pipeline_tag: text2text-generation
base_model: Qwen/Qwen2.5-0.5B
tags:
- asr-error-correction
- japanese
- whisper
- phonetics
- constrained-decoding
- on-device
- gguf
- mlx
---

# Mondegreen — `v{__version__}`

**用語集を、お願いではなく制約にする。**
*A private glossary, compiled into a hard phonetic constraint.*

Whisper does not know your colleagues' names, your product names or your team's
jargon — and 10,000 of them do not fit in a 244-token prompt. Mondegreen corrects
them **afterwards, locally**, as a span replacement that is structurally unable to
leave your glossary.

- Code: https://github.com/{user}/mondegreen
- Space: https://huggingface.co/spaces/{user}/mondegreen
- Dataset: https://huggingface.co/datasets/{user}/mondegreen-asr-errors

## What is in this repo

{have}

## Read this before assuming what the LM does

The **hard constraint is not learned and not in these weights.** The legal
replacement set for a span is computed by `mondegreen.index.PhoneticIndex` as a
finite list, under bounds evaluated before any model runs:

| bound | what it stops |
| --- | --- |
| normalised phonetic distance ≤ τ (0.28) | unrelated words |
| absolute distance ≤ 0.25 + 0.20·√mora | long terms reached via many cheap edits |
| mora-count difference ≤ 34% | invented syllables |
| common dictionary words need near-exact homophony | 「稼働率」→「加藤率」 |
| containment guard | 「新藤さん」→「新藤」 (deleting an honorific) |

The LoRA adapter **only re-ranks candidates already inside that set**. It cannot
add to it, cannot introduce a term that is not in your glossary, and cannot
rewrite grammar. On synthetic glossaries only ~1% of spans have more than one
legal candidate, so most of the time it is not consulted at all — which is
precisely why 4-bit quantisation is safe here, and is asserted by
`tests/test_quantization.py` (removing the LM entirely costs < 2 recall points).

`gate.json` is the calibrated conservative gate — a logistic regression over 18
interpretable span features (AUC {gate.get('auc', 0):.3f}, ECE {gate.get('ece', 0):.3f},
threshold {gate.get('chosen_threshold', 0.82)}). It is 3 KB of JSON and it is the
component whose job is to say *no*.

## Results

400 held-out sentences, 10,000-term glossary, evaluation glossary strictly
disjoint from training by surface **and** by reading:

{_cond_table(bench)}

**(C) wins on term recall (83.0% vs 66.3%) and does 73× the damage** (0.00657 vs
0.00009), needs the transcript to leave the machine, and is therefore unusable on
the confidential audio that motivates this project. That trade is the finding, not
a footnote.

{("| | |\n| --- | --- |\n" + perf) if perf else ""}
> **Provenance.** These numbers are `simulated`: condition (D) is always the real
> system, but the error generator and baselines (B)/(C) are explicit models whose
> parameters are printed in the results file. They are **not** measured Whisper
> numbers. See `benchmarks/README.md` in the repo for how to replace them with
> measurements.

## Use

```bash
pip install git+https://github.com/{user}/mondegreen
```

```python
from mondegreen import ConstrainedCorrector, load_glossary
corrector = ConstrainedCorrector(load_glossary("terms.csv"))
print(corrector.correct("進藤さんが両氏誤り訂正の話をしました。").text)
# 新藤さんが量子誤り訂正の話をしました。
```

With the quantised re-ranker:

```python
from mondegreen.runtime import build_reranker
corrector = ConstrainedCorrector(
    load_glossary("terms.csv"),
    lm=build_reranker("mondegreen-Q4_K_M.gguf"),
)
```

CLI, with the evidence for every edit:

```bash
mondegreen fix transcript.txt --glossary terms.csv
mondegreen explain transcript.txt --glossary terms.csv
```

## Training data

Synthetic. Glossaries are generated by `mondegreen.harvest.GlossaryBuilder`;
carrier sentences by `SentenceFactory`; errors by the phonetic corruption model in
`mondegreen.simulate`, which perturbs the *reading* using the same confusion
classes the distance function discounts and re-renders it as a homophone kanji
spelling. **No real audio, no real person's name, and no LLM grading anywhere.**

## Limitations

- Japanese only. The mora table, the confusion costs and the POS rules are all
  Japanese-specific.
- Without `fugashi`/`pyopenjtalk` the bundled 4,030-kanji fallback table is used.
  It has no part-of-speech information, so the common-word protection cannot fire
  and the damage rate rises. Install `mondegreen[g2p]`.
- The n-gram candidate accelerator is not exact (99.67% recall vs exhaustive at
  10,000 terms). Misses can only cause a *missed* correction, never an illegal
  one — the bound is re-verified on every scored candidate.
- Evaluation is on synthetic glossaries and simulated ASR errors.

## ⚠️ Intended use

A glossary can contain personal names. **Use this only on data under your own
control.** Do not run it over other people's meeting records, or with a glossary
of names you have no relationship to. Mondegreen handles text only, never audio,
and makes no network calls.

## Citation

```bibtex
@software{{mondegreen,
  title  = {{Mondegreen: private glossaries as hard phonetic constraints for local ASR correction}},
  author = {{{user}}},
  year   = {{2026}},
  url    = {{https://github.com/{user}/mondegreen}},
  license = {{Apache-2.0}}
}}
```
"""


def publish_model(api, user: str, repo: str, bench: Dict[str, object], private: bool) -> str:
    """Upload gate + adapter + quantised exports.

    Claim: LOCAL-SPEED -- the published artefacts are the ones that make local
    inference possible.
    """
    repo_id = f"{user}/{repo}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    artefacts: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        gate_src = os.path.join(REPO_ROOT, "models/gate.json")
        if os.path.exists(gate_src):
            shutil.copy(gate_src, os.path.join(tmp, "gate.json"))
            artefacts.append("gate.json — the calibrated conservative gate (3 KB, ships with the package)")
        lora = os.path.join(REPO_ROOT, "models/lora")
        if os.path.isdir(lora):
            dst = os.path.join(tmp, "lora")
            shutil.copytree(
                lora, dst,
                ignore=shutil.ignore_patterns("checkpoint-*", "runs", "*.log", "optimizer.pt"),
            )
            artefacts.append("lora/ — LoRA adapter for the candidate re-ranker (Qwen2.5-0.5B base)")
        quant = os.path.join(REPO_ROOT, "models/quantized")
        if os.path.isdir(quant):
            for name in sorted(os.listdir(quant)):
                src = os.path.join(quant, name)
                if name.endswith(".gguf") and "f16" not in name:
                    shutil.copy(src, os.path.join(tmp, name))
                    artefacts.append(f"{name} — llama.cpp quantised re-ranker")
                elif name == "mlx" and os.path.isdir(src):
                    shutil.copytree(src, os.path.join(tmp, "mlx"))
                    artefacts.append("mlx/ — 4-bit MLX build for Apple silicon")
        for extra in ("export_report.json",):
            src = os.path.join(quant, extra)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(tmp, extra))
        bench_src = os.path.join(REPO_ROOT, "benchmarks/results")
        if os.path.isdir(bench_src):
            os.makedirs(os.path.join(tmp, "benchmarks"), exist_ok=True)
            for f in os.listdir(bench_src):
                if f.endswith(".json"):
                    shutil.copy(os.path.join(bench_src, f), os.path.join(tmp, "benchmarks", f))
        figs = os.path.join(REPO_ROOT, "figures")
        if os.path.isdir(figs):
            os.makedirs(os.path.join(tmp, "figures"), exist_ok=True)
            for f in os.listdir(figs):
                if f.endswith(".png"):
                    shutil.copy(os.path.join(figs, f), os.path.join(tmp, "figures", f))
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(model_card(user, bench, artefacts))
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="model",
                          commit_message=f"Mondegreen v{__version__}")
    return f"https://huggingface.co/{repo_id}"


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

def publish_dataset(api, user: str, repo: str, data_dir: str, private: bool) -> Optional[str]:
    """Upload the (error, gold) pairs and both glossaries.

    Claim: SUPPORT.
    """
    pairs = os.path.join(data_dir, "pairs.jsonl")
    if not os.path.exists(pairs):
        print(f"  no {pairs}; run `make data` first — skipping dataset")
        return None
    from mondegreen.harvest import read_jsonl

    records = read_jsonl(pairs)
    for r in records:
        if not r.source_license or "REQUIRED" in r.source_license:
            raise SystemExit(
                f"refusing to publish: record {r.id} has an unverified licence "
                f"({r.source_license!r}). Verify the source and set it in "
                f"mondegreen.harvest.CORPUS_LICENSES first."
            )
    repo_id = f"{user}/{repo}"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    for name in ("pairs.jsonl", "glossary_train.csv", "glossary_test.csv",
                 "stats.json", "README.md"):
        src = os.path.join(data_dir, name)
        if os.path.exists(src):
            api.upload_file(path_or_fileobj=src, path_in_repo=name,
                            repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"


# --------------------------------------------------------------------------------------
# Space
# --------------------------------------------------------------------------------------

def publish_space(api, user: str, repo: str, private: bool) -> str:
    """Upload the Space as a **static**, fully in-browser build.

    Two reasons it is static rather than a server-backed Gradio Space. The
    practical one: server-backed Gradio Spaces now require a paid plan. The good
    one: under gradio-lite the Python runs in the visitor's browser, so a
    glossary of colleagues' names never reaches Hugging Face's servers. For a
    project whose entire argument is "the audio cannot leave the machine", that
    is the honest deployment.

    Claim: LOCAL-SPEED.
    """
    repo_id = f"{user}/{repo}"
    api.create_repo(repo_id, repo_type="space", exist_ok=True, private=private,
                    space_sdk="static")
    build_dir = os.path.join(REPO_ROOT, "hf/static_space")
    index = os.path.join(build_dir, "index.html")
    if not os.path.exists(index):
        raise SystemExit(
            "no static build found. Run:\n"
            "  python scripts/build_static_space.py"
        )
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(index, os.path.join(tmp, "index.html"))
        shutil.copy(os.path.join(REPO_ROOT, "hf/space_README.md"),
                    os.path.join(tmp, "README.md"))
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          commit_message=f"Mondegreen Space v{__version__} (in-browser)")
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> int:
    """Claim: SUPPORT."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", help="HF user or org (default: whoami)")
    ap.add_argument("--what", nargs="+", default=["model", "dataset", "space"],
                    choices=["model", "dataset", "space"])
    ap.add_argument("--model-repo", default="mondegreen")
    ap.add_argument("--dataset-repo", default="mondegreen-asr-errors")
    ap.add_argument("--space-repo", default="mondegreen")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api, who = _api()
    user = args.user or who.get("name")
    print(f"authenticated as {who.get('name')}; publishing under {user}")
    bench = _bench()
    urls: Dict[str, str] = {}
    if "dataset" in args.what:
        print("dataset...")
        u = publish_dataset(api, user, args.dataset_repo, args.data_dir, args.private)
        if u:
            urls["dataset"] = u
    if "model" in args.what:
        print("model...")
        urls["model"] = publish_model(api, user, args.model_repo, bench, args.private)
    if "space" in args.what:
        print("space...")
        urls["space"] = publish_space(api, user, args.space_repo, args.private)
    print()
    for k, v in urls.items():
        print(f"  {k:8s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
