# Benchmarks

Results land in `results/benchmark.<provenance>.json`. Figures are rendered from
those files by `scripts/make_figures.py`.

## Reproduce

```bash
python scripts/run_benchmarks.py -n 500 --figures
```

That runs entirely offline in a few minutes and produces `provenance: simulated`
results — conditions (B) and (C) come from the models in `mondegreen/baselines.py`
whose parameters are printed in `config.simulation`.

## Replace the simulated parts with measurements

| condition | what makes it `measured` |
| --- | --- |
| (A) raw Whisper | `scripts/harvest_errors.py --mode real` (needs TTS + faster-whisper) |
| (B) prompt stuffing | same, plus `build_asr(..., initial_prompt=<glossary>)` |
| (C) cloud LLM | `python scripts/run_benchmarks.py --cloud` (needs an API key) |
| (D) Mondegreen | always measured — it is the system under test |
| (E) quantised | `--quantized-model models/quantized/mondegreen-Q4_K_M.gguf` |

The **244-token prompt ceiling** in (B) is never simulated: it is computed from a
real tokeniser by `mondegreen.baselines.whisper_prompt_capacity`, and it is the
mechanism behind the headline figure.

## Experimental separation

Enforced in code, asserted at run time, and recorded in every results file under
`separation`:

* train and evaluation **glossaries** are disjoint by surface *and* by reading
  (`GlossaryBuilder.build_pair`);
* train and evaluation **sentences** are generated from separate seeds and
  de-duplicated;
* the evaluation glossary is therefore, by construction, one the gate has never
  been trained on — the "unseen glossary" condition is the default, not an extra.

## What each metric means

| metric | meaning |
| --- | --- |
| `cer` / `wer` | overall error rate against gold |
| `term_recall.recall` | fraction of glossary-term occurrences present in the output |
| `damage.damage_rate_chars` | **headline 破壊率** — already-correct characters we broke |
| `damage.damage_rate_terms` | already-correct term occurrences we broke |
| `damage.damage_rate_edits` | of the edits we made, the fraction that made things worse |
| `hallucination.removal_rate` | canned hallucinations removed |
| `hallucination.false_removal_rate` | genuinely-spoken phrases wrongly deleted |
| `retrieval_recall.recall` | what the n-gram accelerator costs vs an exhaustive scan |
