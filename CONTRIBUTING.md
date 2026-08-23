# Contributing

## The one rule

Every change must keep this true:

> A span may only be replaced by a glossary term it actually sounds like.

`tests/test_hard_constraint.py` is the executable form of that sentence. If a
change makes it fail, the change is wrong — not the test. The same goes for
`test_harmlessness.py` (an empty glossary is the identity function) and
`test_damage_rate.py` (breakage stays under budget).

```bash
pytest tests/ -q -m invariant     # the tests that guard a claim
```

## Setup

```bash
pip install -e '.[dev]' fugashi unidic-lite
pytest tests/ -q
```

`fugashi` + `unidic-lite` are optional but strongly recommended for development:
without part-of-speech information the corrector cannot tell 稼働 (common noun,
protect) from 進藤 (proper noun, correctable), and the damage rate rises. CI runs
both with and without them, and both legs must pass.

## Docstrings

Every function says which claim it substantiates:

```python
def phonetic_distance(a, b, cfg=DEFAULT_CONFIG) -> float:
    """One-line summary.

    Claim: TERM-RECALL + UNBOUNDED-VOCAB.
    """
```

Valid claims: `TERM-RECALL`, `LOW-DAMAGE`, `UNBOUNDED-VOCAB`, `LOCAL-SPEED`,
`SUPPORT`. This is checked to 100% coverage; please keep it there. It is not
ceremony — it is what stops the codebase from accumulating machinery nobody can
connect to a result.

## Changing the phonetics

`mondegreen/phonetics.py` is load-bearing for every number in the repo. If you
change the mora table or a confusion cost:

1. update the hand-checked expectations in `tests/test_phonetics.py`;
2. re-run `make bench` and commit the regenerated `benchmarks/results/` and
   `figures/`;
3. say in the PR what moved and why.

Do not "fix" the distance to satisfy the triangle inequality. It is deliberately
not a metric (context-dependent indel costs), nothing relies on it being one, and
`test_phonetics.py` pins that down.

## Tuning a threshold

Tune on **training** data and report **held-out** numbers. The defaults in
`CorrectorConfig` were set that way and the docstrings record the measurements;
please follow the same pattern rather than tuning against the benchmark you are
about to quote.

## Adding a corpus

Add it to `mondegreen.harvest.CORPUS_LICENSES` with a *verified* licence first.
The harvester raises on unknown corpora and `scripts/harvest_errors.py` refuses to
push a dataset with an unverified licence field. "Aozora Bunko" is not a licence —
individual works there carry their own copyright status.

## Scope

Out of scope, deliberately:

- training or fine-tuning the ASR model itself;
- free-form rewriting, grammar "improvement", punctuation normalisation;
- anything that requires sending the transcript off the machine at inference time.
