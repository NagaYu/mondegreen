"""The three things Mondegreen is measured against.

(A) Raw ASR
    No post-processing.  The floor.

(B) Whisper ``initial_prompt`` stuffing
    Whisper accepts a text prompt that conditions the decoder.  It is capped at
    **244 tokens** -- half of the model's 448-token context is reserved for the
    prompt, minus the special tokens.  Everything past that is silently dropped
    from the *front*.  This is the mechanism behind the headline figure: a
    glossary is not a context you can grow.

(C) Cloud LLM post-processing
    The current recommended workaround: send the transcript and the glossary to
    a large hosted model and ask it to fix things.  Accurate-ish, unbounded in
    vocabulary, and requires shipping the transcript off the machine -- which for
    the confidential meeting audio this problem actually arises in is not a
    latency trade-off, it is a hard no.

Every simulator in this module stamps ``provenance="simulated"`` on its output
and records the exact parameters it used.  The benchmark runner refuses to label
simulated results as measured, and figures built from them carry a visible
watermark.  A simulated baseline is a stated assumption, not a finding.
"""

from __future__ import annotations

import functools
import math
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .glossary import Glossary
from .phonetics import DEFAULT_CONFIG, PhoneticConfig, kana_to_phonemes, normalized_distance
from .types import ErrorPair

#: Whisper's decoder reserves half of its 448-token context for the prompt.
WHISPER_PROMPT_TOKEN_LIMIT = 244


# --------------------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _load_whisper_tokenizer(name: str):
    """Cache the real Whisper tokenizer; falls back to the heuristic counter.

        Claim: UNBOUNDED-VOCAB -- the 244-token ceiling must be counted correctly.
        """
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(name)


def count_tokens(text: str, tokenizer: str = "heuristic") -> int:
    """Count tokens the way Whisper's multilingual BPE would.

    ``tokenizer`` may be ``"heuristic"`` (default, no downloads), ``"auto"``
    (try the real Whisper tokenizer, fall back), or a HuggingFace model id.

    The heuristic is calibrated on Japanese: Whisper's BPE spends roughly one
    token per kanji, a bit less per kana, and far less per latin character.  It
    is an approximation and is *labelled* as one -- :func:`whisper_prompt_capacity`
    reports which counter produced its number.

    Claim: UNBOUNDED-VOCAB -- the 244-token ceiling is the entire mechanism
    behind baseline (B)'s plateau, so counting has to be defensible.
    """
    if not text:
        return 0
    if tokenizer not in ("heuristic", ""):
        name = "openai/whisper-small" if tokenizer == "auto" else tokenizer
        try:
            tok = _load_whisper_tokenizer(name)
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            if tokenizer != "auto":
                raise
    from .reading import char_kind

    total = 0.0
    for ch in text:
        k = char_kind(ch)
        if k == "kanji":
            total += 1.0
        elif k in ("hiragana", "katakana"):
            total += 0.75
        elif k == "latin":
            total += 0.25
        elif k == "digit":
            total += 0.5
        else:
            total += 1.0
    return max(1, int(round(total)))


def whisper_prompt_capacity(
    glossary: Glossary,
    token_limit: int = WHISPER_PROMPT_TOKEN_LIMIT,
    tokenizer: str = "heuristic",
    template: str = "{surface}({reading})、",
) -> Dict[str, object]:
    """How much of a glossary actually fits in Whisper's ``initial_prompt``.

    Returns the number of terms that fit, the number dropped, and the tokenizer
    used.  This is a *measurement*, not a simulation -- the ceiling is a hard
    property of the decoder, and it is the reason the (B) curve in the headline
    figure flattens.

    Claim: UNBOUNDED-VOCAB.
    """
    used = 0
    included: List[str] = []
    dropped: List[str] = []
    for e in glossary:
        piece = template.format(surface=e.surface, reading=e.reading)
        cost = count_tokens(piece, tokenizer)
        if used + cost > token_limit:
            dropped.append(e.surface)
            continue
        included.append(e.surface)
        used += cost
    return {
        "token_limit": token_limit,
        "tokenizer": tokenizer,
        "tokens_used": used,
        "terms_included": len(included),
        "terms_dropped": len(dropped),
        "coverage": len(included) / len(glossary) if len(glossary) else 0.0,
        "included": tuple(included),
    }


# --------------------------------------------------------------------------------------
# (C) Cloud LLM post-processing -- real client
# --------------------------------------------------------------------------------------

_CLOUD_SYSTEM_PROMPT = (
    "あなたは日本語音声認識の後処理を行います。与えられた用語集の語が、"
    "音的に近い誤りとして認識結果に現れている場合にのみ、その語へ置き換えてください。\n"
    "制約:\n"
    "1. 用語集に無い語を新たに導入しない。\n"
    "2. 文法や言い回しを改善しない。言い直し・フィラーもそのまま残す。\n"
    "3. 句読点や数字表記を勝手に正規化しない。\n"
    "4. 出力は訂正後の本文のみ。説明・前置き・コードブロックを付けない。\n"
)


@dataclass
class AnthropicPostProcessor:
    """Condition (C): post-process the transcript with a hosted Claude model.

    Implemented with the official ``anthropic`` SDK.  Note the two things this
    baseline needs that Mondegreen does not: an API key and a network round trip
    carrying the transcript.  For the confidential-meeting case that motivates
    this whole project, the second one is disqualifying regardless of accuracy.

    Claim: LOW-DAMAGE -- (C) is the strongest realistic competitor, and (D) only
    means something measured against it; LOCAL-SPEED -- the latency and the
    "cannot send it at all" column of metric (6) come from here.
    """

    glossary: Glossary
    model: str = "claude-opus-5"
    max_tokens: int = 8000
    effort: str = "low"
    name: str = "anthropic-cloud"
    client: object = None

    def _get_client(self):
        """Lazily construct the Anthropic client so importing this module needs no key.

            Claim: SUPPORT.
            """
        if self.client is not None:
            return self.client
        import anthropic  # type: ignore

        self.client = anthropic.Anthropic()
        return self.client

    def _glossary_block(self) -> str:
        """Render the glossary for a cloud prompt: no budget, no truncation.

            Claim: UNBOUNDED-VOCAB -- (C) has no token ceiling, which is what makes its
            damage rate the interesting comparison rather than its recall.
            """
        return "、".join(f"{e.surface}({e.reading})" for e in self.glossary)

    def __call__(self, transcript: str) -> str:
        """Correct one transcript via the Messages API.

        Note for future editors: ``temperature`` is **not** passed.  It was
        removed on Claude Opus 5 and sending it returns a 400.  Determinism is
        instead approached with a tightly constrained system prompt and low
        effort.

        Claim: SUPPORT.
        """
        import anthropic  # type: ignore

        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_CLOUD_SYSTEM_PROMPT + "\n用語集: " + self._glossary_block(),
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": "認識結果:\n" + transcript}],
            )
        except anthropic.RateLimitError:
            raise
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"cloud baseline failed ({exc.status_code}): {exc.message}") from exc
        out = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return out.strip() or transcript


# --------------------------------------------------------------------------------------
# Simulators
# --------------------------------------------------------------------------------------

@dataclass
class SimulationParams:
    """Stated assumptions behind a simulated baseline.

    These are *not* measurements.  They are an explicit, inspectable model of how
    each baseline behaves, so that a reader can disagree with the numbers by
    disagreeing with the parameters.  Running ``scripts/harvest_errors.py`` with
    real audio replaces every one of them with observed behaviour.

    Claim: SUPPORT -- honesty about provenance is load-bearing for every other
    claim in this repo.
    """

    #: Peak probability that prompt conditioning recovers a term that is *in* the
    #: prompt and phonetically adjacent to the error.
    prompt_recovery_max: float = 0.55
    #: Decay of that probability with normalised phonetic distance.
    prompt_recovery_decay: float = 4.0
    #: Probability of recovering a term that did not fit in the prompt.
    prompt_recovery_offprompt: float = 0.02
    #: Per-sentence chance that prompt conditioning inserts a prompt term that
    #: was never spoken (a real and documented Whisper failure mode).
    prompt_intrusion: float = 0.015

    #: Cloud LLM: peak recovery for a term it can see in the glossary.
    cloud_recovery_max: float = 0.78
    cloud_recovery_decay: float = 2.2
    #: Per-sentence chance the LLM rewrites something that was already correct
    #: (grammar "improvement", number normalisation, hedge removal).
    cloud_rewrite_rate: float = 0.08
    #: Chance the LLM removes a canned hallucination it recognises.
    cloud_hallucination_removal: float = 0.80

    seed: int = 20260823

    def to_dict(self) -> Dict[str, float]:
        """Claim: SUPPORT."""
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class SimulatedBaseline:
    """Shared machinery for the (B) and (C) simulators.

    Claim: SUPPORT.
    """

    glossary: Glossary
    params: SimulationParams = field(default_factory=SimulationParams)
    phonetic_config: PhoneticConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    provenance: str = "simulated"

    def _rng(self, key: str) -> random.Random:
        """Deterministic per-record RNG, so simulated baselines are reproducible.

            Claim: SUPPORT.
            """
        return random.Random(f"{self.params.seed}:{key}")

    def _recover_prob(self, error_span: str, term_reading: str, kind: str) -> float:
        """Recovery probability as a function of phonetic distance.

        Both baselines are modelled as *phonetically* biased -- a prompt or a
        glossary in context helps most when the ASR error already sounds like the
        term.  That is a charitable model: it gives the baselines the same
        information Mondegreen uses.

        Claim: SUPPORT.
        """
        try:
            a = kana_to_phonemes(error_span, self.phonetic_config)
            b = kana_to_phonemes(term_reading, self.phonetic_config)
            d = normalized_distance(a, b, self.phonetic_config) if a and b else 1.0
        except Exception:
            d = 1.0
        if kind == "prompt":
            return self.params.prompt_recovery_max * math.exp(-self.params.prompt_recovery_decay * d)
        return self.params.cloud_recovery_max * math.exp(-self.params.cloud_recovery_decay * d)


class SimulatedWhisperPrompt(SimulatedBaseline):
    """Condition (B) without a GPU: glossary-in-``initial_prompt``, with the ceiling.

    The ceiling is real (measured by :func:`whisper_prompt_capacity`); the
    recovery behaviour inside the ceiling is modelled.  That split is the point:
    the shape of the curve -- flat after 244 tokens -- does not depend on the
    modelled part at all.

    Claim: UNBOUNDED-VOCAB.
    """

    def run(
        self, pairs: Sequence[ErrorPair], tokenizer: str = "heuristic"
    ) -> Tuple[List[str], Dict[str, object]]:
        """Apply simulated prompt conditioning to a list of ASR hypotheses.

        Claim: UNBOUNDED-VOCAB.
        """
        cap = whisper_prompt_capacity(self.glossary, tokenizer=tokenizer)
        in_prompt = set(cap["included"])  # type: ignore[arg-type]
        by_surface = {e.surface: e for e in self.glossary}
        out: List[str] = []
        recovered = 0
        intrusions = 0
        for pair in pairs:
            rng = self._rng(f"B:{pair.id}")
            text = pair.hypothesis
            for surface in pair.glossary_terms:
                entry = by_surface.get(surface)
                if entry is None or surface in text:
                    continue
                span = _closest_error_span(text, entry.reading, self.phonetic_config)
                if span is None:
                    continue
                s, e, span_text = span
                if surface in in_prompt:
                    p = self._recover_prob(_reading_of(span_text), entry.reading, "prompt")
                else:
                    p = self.params.prompt_recovery_offprompt
                if rng.random() < p:
                    text = text[:s] + surface + text[e:]
                    recovered += 1
            if rng.random() < self.params.prompt_intrusion and in_prompt:
                victim = sorted(in_prompt)[rng.randrange(len(in_prompt))]
                text = _intrude(text, victim, rng)
                intrusions += 1
            out.append(text)
        return out, {
            "condition": "B",
            "provenance": self.provenance,
            "capacity": {k: v for k, v in cap.items() if k != "included"},
            "params": self.params.to_dict(),
            "recovered_spans": recovered,
            "intrusions": intrusions,
        }


class SimulatedCloudLLM(SimulatedBaseline):
    """Condition (C) without an API key: unbounded glossary, but it rewrites things.

    Claim: LOW-DAMAGE -- the modelled rewrite rate is what makes (C) interesting;
    a cloud LLM's damage does not come from misunderstanding the glossary, it
    comes from helpfully improving text nobody asked it to touch.
    """

    def run(
        self,
        pairs: Sequence[ErrorPair],
        hallucination_patterns: Sequence[str] = (),
    ) -> Tuple[List[str], Dict[str, object]]:
        """Apply the simulated cloud post-processor.

        Claim: LOW-DAMAGE.
        """
        by_surface = {e.surface: e for e in self.glossary}
        out: List[str] = []
        recovered = 0
        rewrites = 0
        removed = 0
        for pair in pairs:
            rng = self._rng(f"C:{pair.id}")
            text = pair.hypothesis
            for surface in pair.glossary_terms:
                entry = by_surface.get(surface)
                if entry is None or surface in text:
                    continue
                span = _closest_error_span(text, entry.reading, self.phonetic_config)
                if span is None:
                    continue
                s, e, span_text = span
                if rng.random() < self._recover_prob(_reading_of(span_text), entry.reading, "cloud"):
                    text = text[:s] + surface + text[e:]
                    recovered += 1
            for pat in hallucination_patterns:
                if pat in text and rng.random() < self.params.cloud_hallucination_removal:
                    text = text.replace(pat, "", 1).strip()
                    removed += 1
            if rng.random() < self.params.cloud_rewrite_rate:
                text2 = _gratuitous_rewrite(text, rng)
                if text2 != text:
                    rewrites += 1
                    text = text2
            out.append(text)
        return out, {
            "condition": "C",
            "provenance": self.provenance,
            "params": self.params.to_dict(),
            "recovered_spans": recovered,
            "gratuitous_rewrites": rewrites,
            "hallucinations_removed": removed,
            "note": "requires shipping the transcript to a third party",
        }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def _tokenize_cached(text: str) -> tuple:
    """Tokenise once per distinct string.

    The simulators score many candidate spans of the same sentence, and a naive
    implementation re-runs the morphological analyser on every substring -- which
    made a 60-sentence benchmark take minutes.

    Claim: SUPPORT.
    """
    from .reading import get_reader

    return tuple(get_reader().tokenize(text))


@functools.lru_cache(maxsize=32768)
def _reading_of(text: str) -> str:
    """Best-effort kana reading of a string, memoised.

    Claim: SUPPORT.
    """
    try:
        toks = _tokenize_cached(text)
    except Exception:  # pragma: no cover
        return text
    out = []
    for t in toks:
        out.append(t.readings[0] if t.readings else t.surface)
    return "".join(out)


def _closest_error_span(
    text: str, reading: str, cfg: PhoneticConfig, max_norm: float = 0.5
) -> Optional[Tuple[int, int, str]]:
    """Find where in ``text`` a mangled version of ``reading`` most likely sits.

    Enumerates *token* spans rather than character substrings, so the cost is
    linear in tokens rather than quadratic in characters, and the readings come
    straight off the tokeniser instead of being recomputed.

    Used only by the simulators, to decide what a baseline would have fixed.

    Claim: SUPPORT.
    """
    target = kana_to_phonemes(reading, cfg)
    if not target:
        return None
    try:
        toks = _tokenize_cached(text)
    except Exception:  # pragma: no cover
        return None
    n = len(toks)
    best: Optional[Tuple[float, int, int, str]] = None
    for i in range(n):
        if toks[i].kind in ("punct", "space"):
            continue
        acc = ""
        for j in range(i, min(i + 6, n)):
            if toks[j].kind in ("punct", "space"):
                break
            acc += toks[j].readings[0] if toks[j].readings else toks[j].surface
            ph = kana_to_phonemes(acc, cfg)
            if not ph:
                continue
            d = normalized_distance(ph, target, cfg)
            if d <= max_norm and (best is None or d < best[0]):
                s, e = toks[i].start, toks[j].end
                best = (d, s, e, text[s:e])
    if best is None:
        return None
    return best[1], best[2], best[3]


def _intrude(text: str, term: str, rng: random.Random) -> str:
    """Insert a prompt term that was never spoken -- Whisper's prompt-bleed failure.

    Claim: LOW-DAMAGE -- baseline (B) has a damage mode too, and hiding it would
    flatter (D) unfairly.
    """
    if not text:
        return term
    parts = text.split("。")
    i = rng.randrange(len(parts))
    if parts[i]:
        parts[i] = term + parts[i]
    return "。".join(parts)


_REWRITE_RULES: Tuple[Tuple[str, str], ...] = (
    ("えーと、", ""), ("あの、", ""), ("まあ、", ""), ("そのー、", ""),
    ("だと思うんですけど", "だと思います"),
    ("ですけれども", "ですが"),
    ("という形で", "として"),
    ("させていただきます", "します"),
)


def _gratuitous_rewrite(text: str, rng: random.Random) -> str:
    """Apply one unrequested "improvement" -- the cloud LLM's characteristic damage.

    Claim: LOW-DAMAGE.
    """
    applicable = [(a, b) for a, b in _REWRITE_RULES if a in text]
    if not applicable:
        return text
    a, b = applicable[rng.randrange(len(applicable))]
    return text.replace(a, b, 1)
