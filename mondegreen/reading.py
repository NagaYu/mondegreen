"""Turning ASR text into tokens that carry candidate kana readings.

Mondegreen never sees audio.  All it gets is a transcript, so before it can
compare anything phonetically it has to guess how each span of that transcript
*sounds*.  Three backends, in descending order of quality:

1. ``pyopenjtalk``  -- the real Open JTalk frontend, token-aligned readings.
2. ``fugashi``/UniDic -- MeCab readings, token-aligned.
3. :class:`FallbackReader` -- pure Python.  Kana is exact (that is most of what
   ASR emits when it mangles a proper noun), and kanji falls back to the bundled
   425-character table in ``data/kanji_readings.json`` with automatic rendaku
   variants.

The fallback is not as good as a real morphological analyser, and it is honest
about it: characters it cannot read become :data:`~mondegreen.phonetics.UNKNOWN`,
which is *expensive* to align, so an unreadable span simply fails the hard
constraint instead of matching something by accident.
"""

from __future__ import annotations

import functools
import json
import os
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Token

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Characters that end a correction span no matter what.
PUNCTUATION = set("、。，．,.!?！？「」『』（）()［］[]｛｝{}〈〉《》・…‥:：;；\"'“”‘’/／\\|｜~〜ー" .replace("ー", ""))
SPACE = set(" 　\t\n\r")


@functools.lru_cache(maxsize=1)
def kanji_readings() -> Dict[str, Tuple[str, ...]]:
    """Load the bundled kanji -> katakana reading table.

    Claim: SUPPORT -- without readings for kanji there is no phonetic distance to
    compute, and TERM-RECALL on kanji-rendered ASR errors would be zero.
    """
    path = os.path.join(_DATA_DIR, "kanji_readings.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: tuple(v) for k, v in raw.items()}


@functools.lru_cache(maxsize=1)
def common_kanji() -> frozenset:
    """The subset of the reading table that a real ASR system would plausibly emit.

    Derived from the hand-authored core of the table (name kanji, business and
    technical vocabulary) intersected with what UniDic recognises.  Used by the
    error simulator so that a corrupted proper noun comes out as 「進藤」 and not
    as an obscure variant no decoder would ever produce.

    Claim: SUPPORT -- a simulator whose errors are implausible would make
    TERM-RECALL look easier than it is.
    """
    path = os.path.join(_DATA_DIR, "common_kanji.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return frozenset(json.load(fh))
    except FileNotFoundError:  # pragma: no cover
        return frozenset()


# --------------------------------------------------------------------------------------
# Character classification
# --------------------------------------------------------------------------------------

def char_kind(ch: str) -> str:
    """Classify one character as kanji / hiragana / katakana / latin / digit / punct / space.

    Claim: SUPPORT -- span enumeration only merges characters of compatible
    classes, which keeps the candidate count (and therefore LOW-DAMAGE risk)
    under control.
    """
    if ch in SPACE:
        return "space"
    if ch in PUNCTUATION:
        return "punct"
    cp = ord(ch)
    if 0x3041 <= cp <= 0x3096 or ch in "ゝゞ":
        return "hiragana"
    if 0x30A1 <= cp <= 0x30FA or ch in "ーヽヾ・":
        return "katakana"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or ch in "々〆ヶ":
        return "kanji"
    if ch.isdigit():
        return "digit"
    if ("a" <= ch.lower() <= "z") or (0xFF21 <= cp <= 0xFF5A):
        return "latin"
    if unicodedata.category(ch).startswith("P"):
        return "punct"
    return "other"


_UNVOICED_TO_VOICED = {
    "カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
    "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
    "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
    "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ",
}


def rendaku_variants(reading: str) -> Tuple[str, ...]:
    """Add the sequential-voicing form of a reading (中村 = ナカ + *ム*ラ, 山田 = ヤマ + *ダ*).

    Rendaku is not optional in Japanese compounding, and a table that only stores
    the citation reading would miss half of all surnames.

    Claim: TERM-RECALL -- surnames are the highest-value glossary category and
    most of them are rendaku compounds.
    """
    if not reading:
        return ()
    head = reading[0]
    voiced = _UNVOICED_TO_VOICED.get(head)
    if voiced is None:
        return (reading,)
    return (reading, voiced + reading[1:])


_DIGIT_KANA = {
    "0": ("ゼロ", "レイ"), "1": ("イチ",), "2": ("ニ",), "3": ("サン",),
    "4": ("ヨン", "シ"), "5": ("ゴ",), "6": ("ロク",), "7": ("ナナ", "シチ"),
    "8": ("ハチ",), "9": ("キュウ", "ク"),
}
_LATIN_KANA = {
    "a": "エー", "b": "ビー", "c": "シー", "d": "ディー", "e": "イー", "f": "エフ",
    "g": "ジー", "h": "エイチ", "i": "アイ", "j": "ジェー", "k": "ケー", "l": "エル",
    "m": "エム", "n": "エヌ", "o": "オー", "p": "ピー", "q": "キュー", "r": "アール",
    "s": "エス", "t": "ティー", "u": "ユー", "v": "ブイ", "w": "ダブリュー",
    "x": "エックス", "y": "ワイ", "z": "ゼット",
}


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------

class FallbackReader:
    """Dependency-free reader: exact for kana, table-driven for kanji.

    Kana and kanji are emitted one character per token so that correction spans
    can start and end anywhere; runs of latin letters and digits are kept whole
    because splitting an initialism produces nonsense readings.

    Claim: LOCAL-SPEED -- this backend has no model, no dictionary download and
    no startup cost, which is what lets ``mondegreen fix`` run anywhere.
    """

    name = "fallback"
    needs_variants = True
    #: No morphological analysis, so no part of speech. The corrector compensates
    #: by demanding near-exact homophony on kanji spans -- see
    #: :meth:`~mondegreen.corrector.ConstrainedCorrector._propose`.
    has_pos = False

    def tokenize(self, text: str) -> List[Token]:
        """Split ``text`` into reading-bearing tokens.

        Claim: SUPPORT (feeds TERM-RECALL).
        """
        table = kanji_readings()
        tokens: List[Token] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            kind = char_kind(ch)
            if kind in ("latin", "digit"):
                j = i
                while j < n and char_kind(text[j]) == kind:
                    j += 1
                chunk = text[i:j]
                if kind == "digit":
                    readings = ("".join(_DIGIT_KANA.get(c, ("",))[0] for c in chunk),)
                else:
                    readings = ("".join(_LATIN_KANA.get(c.lower(), "") for c in chunk),)
                tokens.append(Token(chunk, i, j, readings, kind))
                i = j
                continue
            if kind in ("hiragana", "katakana"):
                tokens.append(Token(ch, i, i + 1, (ch,), kind))
                i += 1
                continue
            if kind == "kanji":
                readings = table.get(ch, ())
                tokens.append(Token(ch, i, i + 1, tuple(readings), "kanji"))
                i += 1
                continue
            tokens.append(Token(ch, i, i + 1, (), kind))
            i += 1
        return tokens


class FugashiReader:
    """MeCab/UniDic reader.  Token-aligned, dictionary-quality readings.

    Claim: TERM-RECALL -- a real analyser reads kanji compounds the fallback
    table cannot, which raises the ceiling on how many mangled spans are even
    eligible for correction.
    """

    name = "fugashi"
    needs_variants = False
    has_pos = True

    def __init__(self) -> None:
        """Initialise the reader backend.

                Claim: TERM-RECALL.
                """
        import fugashi  # noqa: F401  (import error is the caller's signal)

        self._tagger = fugashi.Tagger()

    def tokenize(self, text: str) -> List[Token]:
        """Claim: SUPPORT (feeds TERM-RECALL)."""
        tokens: List[Token] = []
        pos = 0
        for word in self._tagger(text):
            surface = word.surface
            start = text.find(surface, pos)
            if start < 0:
                start = pos
            end = start + len(surface)
            # Emit any skipped characters (whitespace) as their own tokens.
            if start > pos:
                gap = text[pos:start]
                tokens.append(Token(gap, pos, start, (), char_kind(gap[0])))
            reading = ""
            feat = getattr(word, "feature", None)
            for attr in ("kana", "pron", "reading"):
                reading = getattr(feat, attr, "") or reading
                if reading:
                    break
            readings = (reading,) if reading and reading != "*" else ()
            if not readings and all(char_kind(c) in ("hiragana", "katakana") for c in surface):
                readings = (surface,)
            kind = char_kind(surface[0]) if surface else "other"
            pos_tag = getattr(feat, "pos1", "") or ""
            pos2_tag = getattr(feat, "pos2", "") or ""
            tokens.append(Token(surface, start, end, readings, kind, pos_tag, pos2_tag))
            pos = end
        if pos < len(text):
            rest = text[pos:]
            tokens.append(Token(rest, pos, len(text), (), char_kind(rest[0])))
        return tokens


class PyOpenJTalkReader:
    """Open JTalk frontend reader.  The best readings available offline.

    Claim: TERM-RECALL -- Open JTalk is also what the harvester's TTS uses, so
    training and inference share one phonology.
    """

    name = "pyopenjtalk"
    needs_variants = False
    has_pos = True

    def __init__(self) -> None:
        """Initialise the reader backend.

                Claim: TERM-RECALL.
                """
        import pyopenjtalk  # noqa: F401

        self._pyopenjtalk = pyopenjtalk

    def tokenize(self, text: str) -> List[Token]:
        """Claim: SUPPORT (feeds TERM-RECALL)."""
        tokens: List[Token] = []
        pos = 0
        for feat in self._pyopenjtalk.run_frontend(text):
            surface = feat.get("string", "")
            if not surface:
                continue
            start = text.find(surface, pos)
            if start < 0:
                start = pos
            end = start + len(surface)
            if start > pos:
                gap = text[pos:start]
                tokens.append(Token(gap, pos, start, (), char_kind(gap[0])))
            pron = feat.get("pron", "") or feat.get("read", "")
            pron = pron.replace("’", "").replace("'", "")
            readings = (pron,) if pron and pron != "*" else ()
            kind = char_kind(surface[0])
            pos_parts = (feat.get("pos", "") or "").split(",")
            pos_tag = pos_parts[0] if pos_parts else ""
            pos2_tag = pos_parts[1] if len(pos_parts) > 1 else ""
            tokens.append(Token(surface, start, end, readings, kind, pos_tag, pos2_tag))
            pos = end
        if pos < len(text):
            rest = text[pos:]
            tokens.append(Token(rest, pos, len(text), (), char_kind(rest[0])))
        return tokens


_READER_ORDER = ("pyopenjtalk", "fugashi", "fallback")
_READER_CLASSES = {
    "pyopenjtalk": PyOpenJTalkReader,
    "fugashi": FugashiReader,
    "fallback": FallbackReader,
}


@functools.lru_cache(maxsize=8)
def get_reader(prefer: str = "auto"):
    """Return the best available reader, degrading quietly to the pure-Python one.

    ``prefer`` may name a backend explicitly ("fallback" is useful in tests,
    where determinism matters more than reading quality).

    Claim: LOCAL-SPEED -- the whole pipeline must still work when the user has
    installed nothing but numpy.
    """
    order = _READER_ORDER if prefer in ("auto", "", None) else (prefer,)
    errors: List[str] = []
    for name in order:
        cls = _READER_CLASSES.get(name)
        if cls is None:
            errors.append(f"unknown reader {name!r}")
            continue
        try:
            return cls()
        except Exception as exc:  # pragma: no cover - depends on the environment
            errors.append(f"{name}: {type(exc).__name__}")
    raise RuntimeError("no reader available: " + "; ".join(errors))


# --------------------------------------------------------------------------------------
# Span readings
# --------------------------------------------------------------------------------------

def span_reading_variants(
    tokens: Sequence[Token],
    i: int,
    j: int,
    max_variants: int = 12,
    per_token: int = 3,
    apply_rendaku: bool = True,
) -> Tuple[str, ...]:
    """Enumerate plausible kana readings for ``tokens[i:j]``, best first.

    A multi-kanji span is genuinely ambiguous (中 is ナカ or チュウ, 正 is マサ or
    セイ), so we take a beam rather than committing to one reading.

    The two sources of variation are kept on **separate axes**, and that matters.
    Sequential voicing (rendaku: 小林 = コ + *バ*ヤシ) applies to almost every
    token, so mixing it into the same ranking as alternate base readings lets
    rendaku forms of the top reading crowd out genuine alternatives. Measured:
    with a single mixed beam, 「両氏誤り訂正」 needed 20 variants before the
    correct リョウシアヤマリテイセイ appeared, because rendaku forms of the wrong
    base reading occupied the first 19 slots. Ranking base combinations first and
    only then expanding rendaku finds it within 12 — at less than half the index
    queries.

    Claim: TERM-RECALL (the correct reading has to be reachable) and LOCAL-SPEED
    (every extra variant is another index query per span).
    """
    # --- axis 1: base readings ------------------------------------------
    beam: List[Tuple[str, int]] = [("", 0)]
    hard_cap = max(max_variants * 4, 32)
    for idx in range(i, j):
        tok = tokens[idx]
        if tok.kind in ("punct", "space"):
            return ()
        readings = list(tok.readings[:per_token]) if tok.readings else []
        if not readings:
            readings = ["?" * max(1, len(tok.surface))]
        nxt: List[Tuple[str, int]] = []
        for base, brank in beam:
            for rank, form in enumerate(readings):
                nxt.append((base + form, brank + rank))
        nxt.sort(key=lambda x: x[1])
        seen: Dict[str, int] = {}
        for text, rank in nxt:
            if text not in seen:
                seen[text] = rank
            if len(seen) >= hard_cap:
                break
        beam = sorted(seen.items(), key=lambda x: x[1])[:hard_cap]

    ordered = [t for t, _ in beam if t][:max_variants]
    if not apply_rendaku:
        return tuple(ordered)

    # --- axis 2: rendaku, applied to the surviving base combinations -----
    out: List[str] = []
    seen_out: Dict[str, None] = {}

    def _add(v: str) -> None:
        if v and v not in seen_out:
            seen_out[v] = None
            out.append(v)

    for v in ordered:
        _add(v)
    for idx in range(i + 1, j):
        tok = tokens[idx]
        base = tok.readings[0] if tok.readings else ""
        voiced = rendaku_variants(base)
        if len(voiced) < 2:
            continue
        for v in list(ordered):
            pos = v.find(base)
            if pos >= 0:
                _add(v[:pos] + voiced[1] + v[pos + len(base):])
        if len(out) >= max_variants * 2:
            break
    return tuple(out[: max_variants * 2])


def text_to_reading(text: str, reader=None) -> str:
    """Single best kana reading for a whole string (used for display and for TTS).

    Claim: SUPPORT.
    """
    reader = reader or get_reader()
    toks = reader.tokenize(text)
    out: List[str] = []
    for k, tok in enumerate(toks):
        if tok.kind in ("punct", "space"):
            out.append(tok.surface)
        elif tok.readings:
            r = tok.readings[0]
            out.append(r)
        else:
            out.append(tok.surface)
    return "".join(out)


# --------------------------------------------------------------------------------------
# Span boundary guards
# --------------------------------------------------------------------------------------

#: UniDic coarse POS tags that must never begin or end a correction span.
#: Particles, auxiliaries and suffixes are grammatically *attached* to the word
#: before them, so a span that swallows one is a span whose replacement will
#: delete grammar.
BOUNDARY_BLOCK_POS: frozenset = frozenset({
    "助詞", "助動詞", "接尾辞", "補助記号", "記号", "空白",
    "接続詞", "感動詞", "副詞", "連体詞",
})

#: Surface-form fallback for readers with no POS information.  Honorifics and
#: particles, i.e. exactly the things a naive corrector strips off a name.
AFFIX_SURFACES: frozenset = frozenset({
    "さん", "サン", "様", "さま", "君", "くん", "ちゃん", "氏", "殿", "どの", "先生",
    "の", "は", "が", "を", "に", "へ", "と", "で", "も", "や", "か", "ね", "よ", "な",
    "から", "まで", "より", "など", "とか", "ので", "のに", "けど", "けれど",
    "です", "です", "ます", "した", "する", "して", "され", "せる", "れる", "られ",
    "だ", "た", "て", "い", "る", "ら", "り", "ば", "ず", "ぬ", "ん", "っ",
    "そう", "よう", "たち", "ども", "がた", "側", "的", "性", "化", "式", "系",
})


def is_boundary_blocked(token: Token) -> bool:
    """May this token *not* start or end a correction span?

    Two independent tests, because the answer must be right on every reader
    backend: the morphological POS tag when we have one, and a surface-form
    affix list when we do not.

    Claim: LOW-DAMAGE -- letting a span end on 「さん」 is how a corrector ends up
    deleting an honorific in order to "restore" a name that was already correct.
    This single guard removes an entire class of damage.
    """
    if token.kind in ("punct", "space"):
        return True
    if token.pos and token.pos in BOUNDARY_BLOCK_POS:
        return True
    if token.surface in AFFIX_SURFACES:
        return True
    return False


#: POS combinations that mark a token as an ordinary dictionary word rather than a
#: name or an unknown string.
_COMMON_POS1 = frozenset({"動詞", "形容詞", "形状詞", "副詞", "連体詞", "接続詞", "感動詞"})
_COMMON_POS2 = frozenset({"普通名詞", "数詞"})
_PROPER_POS2 = frozenset({"固有名詞"})


def is_common_word(token: Token) -> bool:
    """Is this token an ordinary word of Japanese, as opposed to a name?

    UniDic separates 名詞-固有名詞 (進藤, 中村) from 名詞-普通名詞 (稼働, 維持), and
    that distinction is exactly the one that matters here: a glossary of private
    proper nouns should be able to rewrite an unknown or proper-noun span freely,
    and should be nearly forbidden from rewriting a common word.

    Returns ``False`` when the reader supplies no POS -- the dependency-free
    fallback reader cannot make this call, which is the main reason the ``[g2p]``
    extra measurably lowers the damage rate.

    Claim: LOW-DAMAGE -- 「稼働率」 becoming 「加藤率」 because the glossary contains
    the surname 加藤 is the single most damaging thing a phonetic corrector can do,
    and this predicate is what stops it.
    """
    if not token.pos:
        return False
    if token.pos in _COMMON_POS1:
        return True
    return token.pos == "名詞" and token.pos2 in _COMMON_POS2


def is_proper_noun(token: Token) -> bool:
    """Is this token tagged as a proper noun?

    Claim: TERM-RECALL -- a proper-noun span is the single best place to spend the
    correction budget, because that is where a private glossary lives.
    """
    return token.pos == "名詞" and token.pos2 in _PROPER_POS2
