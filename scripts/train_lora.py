#!/usr/bin/env python3
"""LoRA-train the optional re-ranker, and push it to the Hub.

What is being trained, and what is not
--------------------------------------

**Not trained:** the ASR model.  Mondegreen never touches Whisper.

**Not trained:** the hard constraint.  The legal candidate set for a span is
computed by the phonetic index and is not learned, not learnable, and not
influenced by anything in this file.

**Trained:** a small causal LM whose only job is to *re-rank* candidates that are
already legal.  The training objective is therefore deliberately narrow: raise
the likelihood of the correct glossary term in its sentence context, so that when
two terms are phonetically indistinguishable (山田/山下 after a dropped mora),
context breaks the tie.

Because the LM only re-ranks, the worst case of quantising it is bounded by "no
LM at all", which is exactly what tests/test_quantization.py measures.

Example
-------
    pip install 'mondegreen[train]'
    python scripts/train_lora.py \\
        --pairs data/pairs.jsonl --glossary data/glossary_train.csv \\
        --base Qwen/Qwen2.5-0.5B --out models/lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig
from mondegreen.glossary import load_glossary
from mondegreen.harvest import read_jsonl
from mondegreen.metrics import classify_span_edit

PROMPT = (
    "音声認識の後処理です。文脈に最も合う語を候補から選びます。\n"
    "文脈: {prefix}【?】{suffix}\n"
    "候補: {candidates}\n"
    "答え: "
)


def build_examples(
    pairs, glossary, tau: float, max_examples: int,
    candidate_tau: float = 0.40, candidate_max_raw: float = 1.2,
) -> List[Dict[str, str]]:
    """Turn labelled spans into next-token targets over the legal candidate set.

    Only spans with **more than one** candidate are kept.  A span with a single
    legal candidate needs no model, and training on it would just teach the LM to
    copy -- burning capacity on cases the constraint already solved.

    Note the two different bounds.  Candidates for *training* are generated at
    ``candidate_tau`` / ``candidate_max_raw``, which are deliberately looser than
    the shipped inference defaults.  At the shipped bound only about 1% of spans
    have a competitor at all (measured: 7 of 800 synthetic glossary terms have any
    phonetic neighbour within max_raw=0.40), so training there would yield almost
    no data and would only ever show the model easy cases.  Training on the wider
    set teaches the discrimination that matters when a user's real glossary *does*
    contain 佐藤 / 佐東 / 左藤.

    Claim: TERM-RECALL -- tie-breaking is the only thing the LM is asked to
    contribute, so it should be trained on ties.
    """
    corrector = ConstrainedCorrector(
        glossary,
        CorrectorConfig(
            tau=candidate_tau, gate_threshold=0.0,
            max_raw_distance=candidate_max_raw,
            tau_common_word=min(0.12, candidate_tau),
        ),
    )
    out: List[Dict[str, str]] = []
    for pair in pairs:
        if len(out) >= max_examples:
            break
        for p in corrector.proposals(pair.hypothesis):
            working = str(p["working_text"])
            s, e = int(p["start"]), int(p["end"])
            cands = [c.entry.surface for c in corrector.candidate_set(str(p["text"]))]
            if len(cands) < 2:
                continue
            outcome = classify_span_edit(pair.gold, working, (s, e), str(p["replacement"]))
            # The target is whichever candidate actually matches gold; if none
            # does, the target is "leave it alone".
            target = None
            for c in cands:
                if classify_span_edit(pair.gold, working, (s, e), c) == "repair":
                    target = c
                    break
            if target is None:
                target = str(p["text"]) if outcome in ("damage", "no-op") else None
            if target is None:
                continue
            options = list(dict.fromkeys([str(p["text"]), *cands]))
            out.append({
                "text": PROMPT.format(
                    prefix=working[max(0, s - 48):s],
                    suffix=working[e:e + 48],
                    candidates=" / ".join(options),
                ) + target,
            })
            if len(out) >= max_examples:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--glossary", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                    help="small base LM; must be convertible to GGUF and MLX")
    ap.add_argument("--out", default="models/lora")
    ap.add_argument("--tau", type=float, default=0.28,
                    help="inference-time bound (recorded in the model card)")
    ap.add_argument("--candidate-tau", type=float, default=0.40,
                    help="wider bound used only to generate training candidate sets")
    ap.add_argument("--candidate-max-raw", type=float, default=1.2,
                    help="wider absolute bound used only for training candidates")
    ap.add_argument("--max-examples", type=int, default=20000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the dataset, then stop (no torch needed)")
    ap.add_argument("--push-to", help="HF model repo id")
    args = ap.parse_args()

    print(f"[1/4] loading {args.pairs}")
    pairs = [p for p in read_jsonl(args.pairs) if p.split == "train"]
    glossary = load_glossary(args.glossary)
    print(f"      {len(pairs)} train pairs, {len(glossary)} glossary terms")

    print("[2/4] building re-ranking examples (ambiguous spans only)")
    examples = build_examples(
        pairs, glossary, args.tau, args.max_examples,
        candidate_tau=args.candidate_tau, candidate_max_raw=args.candidate_max_raw,
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_examples.jsonl"), "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"      {len(examples)} examples -> {args.out}/train_examples.jsonl")
    if not examples:
        print("      no ambiguous spans found even at the widened training bound: "
              "the phonetic constraint resolves every case in this corpus, so there "
              "is nothing for an LM to learn. That is a valid outcome -- it means "
              "condition (E) will equal (D) on this data.")
        return 0
    if args.dry_run:
        print("[3/4] --dry-run: stopping before training")
        return 0

    print(f"[3/4] LoRA training on {args.base}")
    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError as exc:
        print(f"  need: pip install 'mondegreen[train]'  ({exc})", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(examples),
        peft_config=peft_config,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.max_seq_len,
            logging_steps=25,
            save_strategy="epoch",
            seed=args.seed,
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"      adapter -> {args.out}")

    meta = {
        "base_model": args.base, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "examples": len(examples), "epochs": args.epochs, "tau": args.tau,
        "candidate_tau": args.candidate_tau, "candidate_max_raw": args.candidate_max_raw,
        "role": "re-ranks candidates inside the hard phonetic constraint; "
                "does not and cannot widen the candidate set",
    }
    with open(os.path.join(args.out, "mondegreen_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if args.push_to:
        print(f"[4/4] pushing to https://huggingface.co/{args.push_to}")
        trainer.model.push_to_hub(args.push_to)
        tokenizer.push_to_hub(args.push_to)
    else:
        print("[4/4] not pushing (pass --push-to <repo-id>)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
