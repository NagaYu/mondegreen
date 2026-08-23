"""Kana -> phoneme conversion and the weighted phonetic edit distance.

This module is the *only* place where "how close do two things sound" is
defined, and everything downstream (the index, the hard constraint, the gate
features, the evidence panel in the Space) reads its answer from here.

Design notes on the four tricky pieces of Japanese mora phonology:

長音 (chouon, long vowels)
    ``ー`` becomes ``R`` at parse time and is then *expanded into a copy of the
    preceding vowel*, so ``コンピューター`` and ``コンピュータ`` differ by exactly one
    cheap vowel insertion rather than by an exotic symbol.  ``オウ``/``エイ``
    sequences are optionally folded to ``オオ``/``エエ`` (default on) so that
    ``トウキョウ`` == ``トーキョー``.

促音 (sokuon, geminates)
    ``ッ`` becomes a standalone ``Q``.  ASR drops and hallucinates geminates
    constantly, so inserting/deleting ``Q`` is deliberately cheap (0.25) while
    substituting ``Q`` for a real consonant is mid-priced.

拗音 (youon, palatalised morae)
    ``キャ`` is parsed as the two symbols ``ky a`` -- one palatalised onset plus a
    vowel -- never as ``k y a``.  ``k`` <-> ``ky`` is then a cheap substitution,
    which is what actually happens when ASR mishears 「シュン」 as 「スン」.

撥音 (hatsuon, moraic nasal)
    ``ン`` becomes ``N``, a first-class symbol.  ``N`` <-> ``n``/``m`` is cheap
    because the moraic nasal assimilates, and ``N`` indel is cheap because
    far-field audio swallows it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Phonemes

# --------------------------------------------------------------------------------------
# Kana normalisation
# --------------------------------------------------------------------------------------

_HIRA_START, _HIRA_END = 0x3041, 0x3096
_KATA_OFFSET = 0x60

VOWELS = frozenset("aiueo")
SPECIALS = frozenset({"N", "Q", "R"})
#: Symbols that stand in for a reading we could not resolve (unknown kanji).
UNKNOWN = "?"


def to_katakana(text: str) -> str:
    """Fold hiragana to katakana so the mora table only needs one alphabet.

    Half-width katakana and the wave dash family are folded too, because ASR
    output and hand-written glossaries disagree about them constantly.

    Claim: SUPPORT -- normalisation failures show up as spurious phonetic
    distance, which would depress TERM-RECALL for no linguistic reason.
    """
    out: List[str] = []
    for ch in _fold_halfwidth(text):
        cp = ord(ch)
        if _HIRA_START <= cp <= _HIRA_END:
            out.append(chr(cp + _KATA_OFFSET))
        elif ch in "〜~－—―–ｰ":
            out.append("ー")
        else:
            out.append(ch)
    return "".join(out)


def to_hiragana(text: str) -> str:
    """Inverse of :func:`to_katakana`, for display in the CLI/Space.

    Claim: SUPPORT.
    """
    out: List[str] = []
    for ch in text:
        cp = ord(ch)
        if _HIRA_START + _KATA_OFFSET <= cp <= _HIRA_END + _KATA_OFFSET:
            out.append(chr(cp - _KATA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


_HALFWIDTH_KATAKANA = {
    "ｱ": "ア", "ｲ": "イ", "ｳ": "ウ", "ｴ": "エ", "ｵ": "オ",
    "ｶ": "カ", "ｷ": "キ", "ｸ": "ク", "ｹ": "ケ", "ｺ": "コ",
    "ｻ": "サ", "ｼ": "シ", "ｽ": "ス", "ｾ": "セ", "ｿ": "ソ",
    "ﾀ": "タ", "ﾁ": "チ", "ﾂ": "ツ", "ﾃ": "テ", "ﾄ": "ト",
    "ﾅ": "ナ", "ﾆ": "ニ", "ﾇ": "ヌ", "ﾈ": "ネ", "ﾉ": "ノ",
    "ﾊ": "ハ", "ﾋ": "ヒ", "ﾌ": "フ", "ﾍ": "ヘ", "ﾎ": "ホ",
    "ﾏ": "マ", "ﾐ": "ミ", "ﾑ": "ム", "ﾒ": "メ", "ﾓ": "モ",
    "ﾔ": "ヤ", "ﾕ": "ユ", "ﾖ": "ヨ",
    "ﾗ": "ラ", "ﾘ": "リ", "ﾙ": "ル", "ﾚ": "レ", "ﾛ": "ロ",
    "ﾜ": "ワ", "ｦ": "ヲ", "ﾝ": "ン",
    "ｧ": "ァ", "ｨ": "ィ", "ｩ": "ゥ", "ｪ": "ェ", "ｫ": "ォ",
    "ｬ": "ャ", "ｭ": "ュ", "ｮ": "ョ", "ｯ": "ッ", "ｰ": "ー",
}
_DAKUTEN = {
    "カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
    "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
    "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
    "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ", "ウ": "ヴ",
}
_HANDAKUTEN = {"ハ": "パ", "ヒ": "ピ", "フ": "プ", "ヘ": "ペ", "ホ": "ポ"}


def _fold_halfwidth(text: str) -> str:
    """Fold half-width katakana and combining dakuten into composed full-width kana.

        Claim: SUPPORT -- ASR output and hand-written glossaries disagree about these.
        """
    out: List[str] = []
    for ch in text:
        base = _HALFWIDTH_KATAKANA.get(ch, ch)
        if base in ("ﾞ", "゙") and out:
            out[-1] = _DAKUTEN.get(out[-1], out[-1])
            continue
        if base in ("ﾟ", "゚") and out:
            out[-1] = _HANDAKUTEN.get(out[-1], out[-1])
            continue
        out.append(base)
    return "".join(out)


# --------------------------------------------------------------------------------------
# Mora table
# --------------------------------------------------------------------------------------

#: Two-character digraphs (youon and the borrowed-sound set).  Checked first.
_DIGRAPHS: Dict[str, Tuple[str, str]] = {
    "キャ": ("ky", "a"), "キュ": ("ky", "u"), "キョ": ("ky", "o"), "キェ": ("ky", "e"),
    "ギャ": ("gy", "a"), "ギュ": ("gy", "u"), "ギョ": ("gy", "o"), "ギェ": ("gy", "e"),
    "シャ": ("sh", "a"), "シュ": ("sh", "u"), "ショ": ("sh", "o"), "シェ": ("sh", "e"),
    "ジャ": ("j", "a"),  "ジュ": ("j", "u"),  "ジョ": ("j", "o"),  "ジェ": ("j", "e"),
    "チャ": ("ch", "a"), "チュ": ("ch", "u"), "チョ": ("ch", "o"), "チェ": ("ch", "e"),
    "ヂャ": ("j", "a"),  "ヂュ": ("j", "u"),  "ヂョ": ("j", "o"),
    "ニャ": ("ny", "a"), "ニュ": ("ny", "u"), "ニョ": ("ny", "o"), "ニェ": ("ny", "e"),
    "ヒャ": ("hy", "a"), "ヒュ": ("hy", "u"), "ヒョ": ("hy", "o"), "ヒェ": ("hy", "e"),
    "ビャ": ("by", "a"), "ビュ": ("by", "u"), "ビョ": ("by", "o"),
    "ピャ": ("py", "a"), "ピュ": ("py", "u"), "ピョ": ("py", "o"),
    "ミャ": ("my", "a"), "ミュ": ("my", "u"), "ミョ": ("my", "o"),
    "リャ": ("ry", "a"), "リュ": ("ry", "u"), "リョ": ("ry", "o"), "リェ": ("ry", "e"),
    # borrowed sounds
    "ファ": ("f", "a"), "フィ": ("f", "i"), "フェ": ("f", "e"), "フォ": ("f", "o"),
    "フュ": ("fy", "u"), "フャ": ("fy", "a"), "フョ": ("fy", "o"),
    "ヴァ": ("v", "a"), "ヴィ": ("v", "i"), "ヴェ": ("v", "e"), "ヴォ": ("v", "o"),
    "ヴュ": ("vy", "u"), "ヴャ": ("vy", "a"), "ヴョ": ("vy", "o"),
    "ツァ": ("ts", "a"), "ツィ": ("ts", "i"), "ツェ": ("ts", "e"), "ツォ": ("ts", "o"),
    "ティ": ("t", "i"), "トゥ": ("t", "u"), "テュ": ("ty", "u"),
    "ディ": ("d", "i"), "ドゥ": ("d", "u"), "デュ": ("dy", "u"),
    "ウィ": ("w", "i"), "ウェ": ("w", "e"), "ウォ": ("w", "o"), "ウャ": ("y", "a"),
    "クァ": ("kw", "a"), "クィ": ("kw", "i"), "クェ": ("kw", "e"), "クォ": ("kw", "o"),
    "グァ": ("gw", "a"), "グィ": ("gw", "i"), "グェ": ("gw", "e"), "グォ": ("gw", "o"),
    "スィ": ("s", "i"), "ズィ": ("z", "i"), "イェ": ("y", "e"),
    "テァ": ("t", "a"), "シィ": ("sh", "i"), "ジィ": ("j", "i"),
}

#: Single characters.  ``None`` onset means the mora is a bare vowel.
_MONOGRAPHS: Dict[str, Tuple[str, str]] = {
    "ア": ("", "a"), "イ": ("", "i"), "ウ": ("", "u"), "エ": ("", "e"), "オ": ("", "o"),
    "カ": ("k", "a"), "キ": ("k", "i"), "ク": ("k", "u"), "ケ": ("k", "e"), "コ": ("k", "o"),
    "ガ": ("g", "a"), "ギ": ("g", "i"), "グ": ("g", "u"), "ゲ": ("g", "e"), "ゴ": ("g", "o"),
    "サ": ("s", "a"), "シ": ("sh", "i"), "ス": ("s", "u"), "セ": ("s", "e"), "ソ": ("s", "o"),
    "ザ": ("z", "a"), "ジ": ("j", "i"), "ズ": ("z", "u"), "ゼ": ("z", "e"), "ゾ": ("z", "o"),
    "タ": ("t", "a"), "チ": ("ch", "i"), "ツ": ("ts", "u"), "テ": ("t", "e"), "ト": ("t", "o"),
    "ダ": ("d", "a"), "ヂ": ("j", "i"), "ヅ": ("z", "u"), "デ": ("d", "e"), "ド": ("d", "o"),
    "ナ": ("n", "a"), "ニ": ("n", "i"), "ヌ": ("n", "u"), "ネ": ("n", "e"), "ノ": ("n", "o"),
    "ハ": ("h", "a"), "ヒ": ("h", "i"), "フ": ("f", "u"), "ヘ": ("h", "e"), "ホ": ("h", "o"),
    "バ": ("b", "a"), "ビ": ("b", "i"), "ブ": ("b", "u"), "ベ": ("b", "e"), "ボ": ("b", "o"),
    "パ": ("p", "a"), "ピ": ("p", "i"), "プ": ("p", "u"), "ペ": ("p", "e"), "ポ": ("p", "o"),
    "マ": ("m", "a"), "ミ": ("m", "i"), "ム": ("m", "u"), "メ": ("m", "e"), "モ": ("m", "o"),
    "ヤ": ("y", "a"), "ユ": ("y", "u"), "ヨ": ("y", "o"),
    "ラ": ("r", "a"), "リ": ("r", "i"), "ル": ("r", "u"), "レ": ("r", "e"), "ロ": ("r", "o"),
    "ワ": ("w", "a"), "ヲ": ("", "o"), "ヰ": ("", "i"), "ヱ": ("", "e"),
    "ヴ": ("v", "u"),
    "ヷ": ("v", "a"), "ヸ": ("v", "i"), "ヹ": ("v", "e"), "ヺ": ("v", "o"),
    # small kana standing alone (common in stylised product names)
    "ァ": ("", "a"), "ィ": ("", "i"), "ゥ": ("", "u"), "ェ": ("", "e"), "ォ": ("", "o"),
    "ャ": ("y", "a"), "ュ": ("y", "u"), "ョ": ("y", "o"), "ヮ": ("w", "a"),
    "ヵ": ("k", "a"), "ヶ": ("k", "a"),
}

#: Digits and a few units, so the "numbers and units" pathology class is reachable.
_DIGIT_READINGS: Dict[str, Tuple[str, ...]] = {
    "0": ("ゼロ", "レイ"), "1": ("イチ",), "2": ("ニ",), "3": ("サン",), "4": ("ヨン", "シ"),
    "5": ("ゴ",), "6": ("ロク",), "7": ("ナナ", "シチ"), "8": ("ハチ",), "9": ("キュウ", "ク"),
}

#: Latin letters read as they are spoken in Japanese (initialisms in product names).
_LATIN_READINGS: Dict[str, str] = {
    "a": "エー", "b": "ビー", "c": "シー", "d": "ディー", "e": "イー", "f": "エフ",
    "g": "ジー", "h": "エイチ", "i": "アイ", "j": "ジェー", "k": "ケー", "l": "エル",
    "m": "エム", "n": "エヌ", "o": "オー", "p": "ピー", "q": "キュー", "r": "アール",
    "s": "エス", "t": "ティー", "u": "ユー", "v": "ブイ", "w": "ダブリュー",
    "x": "エックス", "y": "ワイ", "z": "ゼット",
}


# --------------------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------------------

def _pairs(spec: Sequence[Tuple[str, str, float]]) -> Dict[frozenset, float]:
    """Build the symmetric confusion lookup from the (a, b, cost) spec table.

        Claim: SUPPORT.
        """
    return {frozenset((a, b)): c for a, b, c in spec}


#: Weighted substitution costs.  Anything not listed falls back to the class
#: defaults in :class:`PhoneticConfig`.  The numbers encode "these two sounds get
#: swapped by real ASR systems", which is what makes the constraint *phonetic*
#: rather than a generic string edit distance.
_CONFUSION_SPEC: Tuple[Tuple[str, str, float], ...] = (
    # --- voicing: by far the most frequent Japanese ASR confusion -----------------
    ("k", "g", 0.30), ("ky", "gy", 0.30), ("s", "z", 0.30), ("sh", "j", 0.30),
    ("t", "d", 0.30), ("ch", "j", 0.32), ("ts", "z", 0.40), ("h", "b", 0.38),
    ("hy", "by", 0.38), ("b", "p", 0.30), ("by", "py", 0.30), ("f", "v", 0.30),
    ("kw", "gw", 0.30), ("t", "ts", 0.40), ("d", "z", 0.40),
    # --- sibilants and affricates ------------------------------------------------
    ("s", "sh", 0.35), ("z", "j", 0.30), ("sh", "ch", 0.35), ("ch", "ts", 0.35),
    ("s", "ts", 0.40), ("t", "ch", 0.40), ("d", "j", 0.40), ("s", "h", 0.50),
    ("sh", "h", 0.50), ("sh", "hy", 0.45),
    # --- palatalisation ----------------------------------------------------------
    ("k", "ky", 0.40), ("g", "gy", 0.40), ("n", "ny", 0.38), ("h", "hy", 0.40),
    ("b", "by", 0.40), ("p", "py", 0.40), ("m", "my", 0.40), ("r", "ry", 0.40),
    ("f", "fy", 0.40), ("v", "vy", 0.40), ("t", "ty", 0.40), ("d", "dy", 0.40),
    ("ty", "ch", 0.35), ("dy", "j", 0.35),
    # --- nasals ------------------------------------------------------------------
    ("n", "m", 0.35), ("n", "N", 0.28), ("m", "N", 0.30), ("ny", "N", 0.42),
    ("my", "N", 0.45), ("N", "u", 0.62), ("N", "o", 0.66), ("N", "i", 0.70),
    # --- liquids and glides ------------------------------------------------------
    ("r", "d", 0.45), ("r", "n", 0.50), ("r", "w", 0.55), ("r", "l", 0.10),
    ("y", "i", 0.40), ("w", "u", 0.40), ("y", "ny", 0.45), ("w", "v", 0.45),
    ("w", "b", 0.50), ("h", "w", 0.55), ("y", "r", 0.55),
    # --- fricatives --------------------------------------------------------------
    ("h", "f", 0.38), ("f", "p", 0.45), ("f", "b", 0.50), ("h", "k", 0.55),
    ("v", "b", 0.35), ("f", "h", 0.38), ("f", "s", 0.50),
    # --- place of articulation for stops ----------------------------------------
    ("t", "k", 0.60), ("k", "p", 0.65), ("t", "p", 0.65),
    ("d", "g", 0.60), ("b", "d", 0.60), ("b", "g", 0.65),
    ("k", "kw", 0.30), ("g", "gw", 0.30),
    # --- vowels ------------------------------------------------------------------
    ("i", "e", 0.35), ("u", "o", 0.35), ("a", "o", 0.45), ("a", "e", 0.50),
    ("u", "i", 0.45), ("o", "e", 0.50), ("a", "u", 0.55), ("a", "i", 0.55),
    ("i", "o", 0.60), ("e", "u", 0.50),
    # --- geminate heard as its own consonant ------------------------------------
    ("Q", "t", 0.55), ("Q", "k", 0.55), ("Q", "s", 0.55), ("Q", "ts", 0.55),
    ("Q", "ch", 0.58), ("Q", "p", 0.58), ("Q", "sh", 0.58),
    # --- long-vowel marker vs a plain vowel (only reachable with expansion off) ---
    ("R", "a", 0.20), ("R", "i", 0.20), ("R", "u", 0.20), ("R", "e", 0.20), ("R", "o", 0.20),
)

CONFUSION: Dict[frozenset, float] = _pairs(_CONFUSION_SPEC)

#: Consonants after which an epenthetic /u/ or /i/ routinely appears or vanishes
#: in loanwords ("ストライク" <-> "ストライク" heard as "スtライク").
_EPENTHESIS_U = frozenset({"s", "sh", "ts", "z", "j", "k", "g", "t", "d", "ch", "b", "p", "f", "v"})
_EPENTHESIS_I = frozenset({"ch", "j", "sh", "k", "g", "r", "t", "d"})


@dataclass
class PhoneticConfig:
    """Every tunable knob of the phonetic model, in one serialisable object.

    The exact config used for a run is written into benchmark result files and
    the model card, because a phonetic threshold is meaningless without the cost
    table it was measured against.

    Claim: SUPPORT -- reproducibility of TERM-RECALL / LOW-DAMAGE numbers.
    """

    # --- parsing -------------------------------------------------------------
    expand_long_vowel_mark: bool = True   # ー -> copy of previous vowel
    fold_ou_to_oo: bool = True            # トウキョウ == トーキョー
    fold_ei_to_ee: bool = True            # セイ == セー
    collapse_long_vowels: bool = False    # aggressive: aa -> a (off by default)

    # --- substitution --------------------------------------------------------
    vowel_vowel_default: float = 0.60
    consonant_consonant_default: float = 1.00
    cross_class_default: float = 1.10
    unknown_sub_cost: float = 0.95        # cost of aligning against '?' (unread kanji)

    # --- insertion / deletion ------------------------------------------------
    indel_consonant: float = 1.00
    indel_vowel: float = 0.85
    indel_Q: float = 0.25
    indel_N: float = 0.40
    indel_R: float = 0.15
    indel_long_vowel: float = 0.15        # duplicate vowel next to an identical one
    indel_glide: float = 0.50             # y / w
    indel_epenthetic: float = 0.50        # u after s/k/t..., i after ch/j...
    indel_unknown: float = 0.95

    # --- normalisation -------------------------------------------------------
    normalize_by: str = "max"             # max | term | mean

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PhoneticConfig":
        """Claim: SUPPORT."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})  # type: ignore[arg-type]


DEFAULT_CONFIG = PhoneticConfig()


def _is_vowel(p: str) -> bool:
    """Is this phoneme a vowel? Drives class-based default costs.

        Claim: SUPPORT.
        """
    return p in VOWELS


def substitution_cost(a: str, b: str, cfg: PhoneticConfig = DEFAULT_CONFIG) -> float:
    """Weighted cost of hearing phoneme ``a`` as phoneme ``b``.

    Confusable pairs (voicing, sibilants, palatalisation, moraic nasal) are
    discounted; everything else pays a full-class default.  Lowering the cost of
    genuinely confusable sounds is what lets a mangled span still reach its
    glossary term, while keeping unrelated sounds expensive is what stops the
    corrector from rewriting healthy text.

    Claim: TERM-RECALL (cheap real confusions) + LOW-DAMAGE (expensive fake ones).
    """
    if a == b:
        return 0.0
    if a == UNKNOWN or b == UNKNOWN:
        return cfg.unknown_sub_cost
    hit = CONFUSION.get(frozenset((a, b)))
    if hit is not None:
        return hit
    av, bv = _is_vowel(a), _is_vowel(b)
    if av and bv:
        return cfg.vowel_vowel_default
    if not av and not bv and a not in SPECIALS and b not in SPECIALS:
        return cfg.consonant_consonant_default
    return cfg.cross_class_default


def indel_cost(
    seq: Sequence[str],
    i: int,
    cfg: PhoneticConfig = DEFAULT_CONFIG,
) -> float:
    """Cost of inserting/deleting ``seq[i]``, using its neighbours as context.

    Context matters: dropping the second ``o`` of ``オオ`` is a length change
    (cheap), dropping a lone ``o`` is a mora loss (expensive).  Same for the
    epenthetic vowels of loanwords and for ``Q``/``N``, which far-field audio
    swallows routinely.

    Claim: TERM-RECALL -- length and geminate slips are the single most common
    way a correct term comes back wrong from ASR.
    """
    p = seq[i]
    if p == UNKNOWN:
        return cfg.indel_unknown
    if p == "Q":
        return cfg.indel_Q
    if p == "N":
        return cfg.indel_N
    if p == "R":
        return cfg.indel_R
    if _is_vowel(p):
        prev = seq[i - 1] if i > 0 else None
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        if prev == p or nxt == p:
            return cfg.indel_long_vowel
        if p == "u" and prev in _EPENTHESIS_U:
            return cfg.indel_epenthetic
        if p == "i" and prev in _EPENTHESIS_I:
            return cfg.indel_epenthetic
        return cfg.indel_vowel
    if p in ("y", "w"):
        return cfg.indel_glide
    return cfg.indel_consonant


# --------------------------------------------------------------------------------------
# Kana -> phonemes
# --------------------------------------------------------------------------------------

def kana_to_phonemes(kana: str, cfg: PhoneticConfig = DEFAULT_CONFIG) -> Phonemes:
    """Convert a kana reading into a mora-phoneme sequence.

    Unresolvable characters become :data:`UNKNOWN` rather than being dropped, so
    a span whose reading we only partly know can never masquerade as a perfect
    phonetic match.

    Claim: TERM-RECALL -- this is the representation every glossary term is
    indexed under; LOW-DAMAGE -- ``?`` keeps unreadable spans honest.
    """
    text = to_katakana(kana)
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in (" ", "　", "・", "=", "＝", "-", "‐"):
            i += 1
            continue
        if ch == "ッ":
            out.append("Q")
            i += 1
            continue
        if ch == "ン":
            out.append("N")
            i += 1
            continue
        if ch == "ー":
            out.append("R")
            i += 1
            continue
        if i + 1 < n:
            di = _DIGRAPHS.get(text[i : i + 2])
            if di is not None:
                onset, vowel = di
                if onset:
                    out.append(onset)
                out.append(vowel)
                i += 2
                continue
        mono = _MONOGRAPHS.get(ch)
        if mono is not None:
            onset, vowel = mono
            if onset:
                out.append(onset)
            out.append(vowel)
            i += 1
            continue
        # Latin / digits inside a reading: expand via their spoken forms.
        low = ch.lower()
        if low in _LATIN_READINGS:
            out.extend(kana_to_phonemes(_LATIN_READINGS[low], cfg))
            i += 1
            continue
        if ch in _DIGIT_READINGS:
            out.extend(kana_to_phonemes(_DIGIT_READINGS[ch][0], cfg))
            i += 1
            continue
        out.append(UNKNOWN)
        i += 1
    return canonicalize(tuple(out), cfg)


def canonicalize(phonemes: Sequence[str], cfg: PhoneticConfig = DEFAULT_CONFIG) -> Phonemes:
    """Apply the long-vowel normalisations to a raw phoneme sequence.

    Runs in three passes: ``R`` -> copy of the preceding vowel, then ``ou``/``ei``
    folding, then (optionally) collapsing of all doubled vowels.

    Claim: TERM-RECALL -- Japanese orthography spells one sound several ways and
    ASR picks whichever it likes; canonicalising removes that free variation
    before the distance is ever computed.
    """
    seq = list(phonemes)

    if cfg.expand_long_vowel_mark:
        out: List[str] = []
        for p in seq:
            if p == "R":
                if out and _is_vowel(out[-1]):
                    out.append(out[-1])
                # a leading 'ー' has nothing to lengthen; drop it
                continue
            out.append(p)
        seq = out

    if cfg.fold_ou_to_oo or cfg.fold_ei_to_ee:
        out = []
        for p in seq:
            if out and _is_vowel(out[-1]):
                if cfg.fold_ou_to_oo and out[-1] == "o" and p == "u":
                    out.append("o")
                    continue
                if cfg.fold_ei_to_ee and out[-1] == "e" and p == "i":
                    out.append("e")
                    continue
            out.append(p)
        seq = out

    if cfg.collapse_long_vowels:
        out = []
        for p in seq:
            if out and out[-1] == p and _is_vowel(p):
                continue
            out.append(p)
        seq = out

    return tuple(seq)


def phoneme_string(phonemes: Sequence[str]) -> str:
    """Space-separated rendering used in the CLI output and the Space evidence panel.

    Claim: SUPPORT -- the evidence display is what makes a correction auditable.
    """
    return " ".join(phonemes)


def mora_count(phonemes: Sequence[str]) -> int:
    """Number of morae, i.e. vowels plus the special morae ``N``/``Q``/``R``.

    Claim: SUPPORT -- used as a gate feature, since short spans are much riskier
    to replace than long ones.
    """
    return sum(1 for p in phonemes if _is_vowel(p) or p in SPECIALS)


# --------------------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------------------

def _normalizer(a: Sequence[str], b: Sequence[str], cfg: PhoneticConfig) -> float:
    """Denominator for length normalisation, per ``cfg.normalize_by``.

        Claim: SUPPORT.
        """
    if cfg.normalize_by == "term":
        return float(max(1, len(b)))
    if cfg.normalize_by == "mean":
        return max(1.0, (len(a) + len(b)) / 2.0)
    return float(max(1, len(a), len(b)))


def phonetic_distance(
    a: Sequence[str],
    b: Sequence[str],
    cfg: PhoneticConfig = DEFAULT_CONFIG,
) -> float:
    """Raw weighted phonetic edit distance between two phoneme sequences.

    Plain Levenshtein DP with the context-sensitive costs above.  ``O(len(a) *
    len(b))`` with a tiny constant -- fast enough that a 10,000-term glossary is
    still interactive once the n-gram index has pruned the candidate set.

    This is symmetric and non-negative, and zero only for identical sequences,
    but it is **not a metric**: context-dependent indel costs break the triangle
    inequality (see tests/test_phonetics.py).  Nothing here needs it to be one.
    The only structural property the index relies on is
    ``d(a, b) >= |len(a) - len(b)| * min_indel``, which holds because every
    alignment contains at least that many indels.

    Claim: TERM-RECALL + UNBOUNDED-VOCAB (cost is per candidate, not per token of
    a prompt, so glossary size never hits a context ceiling).
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return sum(indel_cost(b, j, cfg) for j in range(m))
    if m == 0:
        return sum(indel_cost(a, i, cfg) for i in range(n))

    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + indel_cost(b, j - 1, cfg)
    cur = [0.0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = prev[0] + indel_cost(a, i - 1, cfg)
        ai = a[i - 1]
        del_c = indel_cost(a, i - 1, cfg)
        for j in range(1, m + 1):
            sub = prev[j - 1] + substitution_cost(ai, b[j - 1], cfg)
            dele = prev[j] + del_c
            ins = cur[j - 1] + indel_cost(b, j - 1, cfg)
            cur[j] = sub if sub <= dele and sub <= ins else (dele if dele <= ins else ins)
        prev, cur = cur, prev
    return prev[m]


def normalized_distance(
    a: Sequence[str],
    b: Sequence[str],
    cfg: PhoneticConfig = DEFAULT_CONFIG,
) -> float:
    """Length-normalised phonetic distance in ``[0, ~1.05]``.

    This is the quantity the hard constraint is expressed in: a replacement is
    *structurally impossible* unless this value is <= tau.

    Claim: LOW-DAMAGE -- one scalar, one threshold, one invariant that
    tests/test_hard_constraint.py asserts can never be violated.
    """
    return phonetic_distance(a, b, cfg) / _normalizer(a, b, cfg)


def bounded_normalized_distance(
    a: Sequence[str],
    b: Sequence[str],
    max_norm: float,
    cfg: PhoneticConfig = DEFAULT_CONFIG,
) -> Optional[float]:
    """Like :func:`normalized_distance` but abandons hopeless pairs early.

    Returns ``None`` as soon as the whole DP row exceeds the absolute budget
    implied by ``max_norm``.  This is what keeps a 10,000-term glossary within a
    few milliseconds per span.

    Claim: UNBOUNDED-VOCAB + LOCAL-SPEED -- the retrieval cost has to stay flat
    as the glossary grows, or the headline figure is unreachable.
    """
    n, m = len(a), len(b)
    budget = max_norm * _normalizer(a, b, cfg)
    # Cheapest possible length change: no alignment can cost less than this.
    if abs(n - m) * cfg.indel_R > budget:
        return None
    if n == 0 or m == 0:
        d = phonetic_distance(a, b, cfg)
        return d / _normalizer(a, b, cfg) if d <= budget else None

    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + indel_cost(b, j - 1, cfg)
    cur = [0.0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = prev[0] + indel_cost(a, i - 1, cfg)
        ai = a[i - 1]
        del_c = indel_cost(a, i - 1, cfg)
        row_min = cur[0]
        for j in range(1, m + 1):
            sub = prev[j - 1] + substitution_cost(ai, b[j - 1], cfg)
            dele = prev[j] + del_c
            ins = cur[j - 1] + indel_cost(b, j - 1, cfg)
            v = sub if sub <= dele and sub <= ins else (dele if dele <= ins else ins)
            cur[j] = v
            if v < row_min:
                row_min = v
        # Every remaining alignment must pass through this row.
        if row_min > budget:
            return None
        prev, cur = cur, prev
    d = prev[m]
    if d > budget:
        return None
    return d / _normalizer(a, b, cfg)


# --------------------------------------------------------------------------------------
# Precomputed fast path
# --------------------------------------------------------------------------------------

def phoneme_alphabet() -> Tuple[str, ...]:
    """Every symbol :func:`kana_to_phonemes` can emit.

    Claim: SUPPORT -- lets the cost tables be fully precomputed instead of
    branched through on every DP cell.
    """
    syms = set(VOWELS) | set(SPECIALS) | {UNKNOWN}
    for onset, vowel in list(_MONOGRAPHS.values()) + list(_DIGRAPHS.values()):
        if onset:
            syms.add(onset)
        syms.add(vowel)
    return tuple(sorted(syms))


_SUB_TABLE_CACHE: Dict[Tuple, Dict[Tuple[str, str], float]] = {}


def substitution_table(cfg: PhoneticConfig = DEFAULT_CONFIG) -> Dict[Tuple[str, str], float]:
    """Dense ``(a, b) -> cost`` table over the whole phoneme alphabet.

    The alphabet has fewer than fifty symbols, so the full table is ~2,000
    entries and turns the innermost DP operation into one dict lookup.  This is
    the single biggest constant-factor win in the system.

    Claim: LOCAL-SPEED.
    """
    key = (cfg.vowel_vowel_default, cfg.consonant_consonant_default,
           cfg.cross_class_default, cfg.unknown_sub_cost)
    hit = _SUB_TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    alpha = phoneme_alphabet()
    table = {(a, b): substitution_cost(a, b, cfg) for a in alpha for b in alpha}
    _SUB_TABLE_CACHE[key] = table
    return table


def indel_costs(seq: Sequence[str], cfg: PhoneticConfig = DEFAULT_CONFIG) -> Tuple[float, ...]:
    """Context-dependent indel cost for every position of ``seq``, computed once.

    Indel cost depends on neighbours, so it is a property of the *sequence*, not
    of the DP cell.  Hoisting it out of the loop removes millions of redundant
    calls per benchmark.

    Claim: LOCAL-SPEED.
    """
    return tuple(indel_cost(seq, i, cfg) for i in range(len(seq)))


def bounded_distance_pre(
    a: Sequence[str],
    a_indel: Sequence[float],
    b: Sequence[str],
    b_indel: Sequence[float],
    sub: Dict[Tuple[str, str], float],
    budget: float,
    min_indel: float,
) -> Optional[float]:
    """Early-abandoning weighted DP over precomputed cost arrays.

    Returns the raw distance, or ``None`` once every alignment left is provably
    over ``budget``.  Semantically identical to
    :func:`bounded_normalized_distance`; tests assert the two agree exactly.

    Claim: LOCAL-SPEED + UNBOUNDED-VOCAB.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        total = sum(a_indel) if m == 0 else sum(b_indel)
        return total if total <= budget else None
    if abs(n - m) * min_indel > budget:
        return None

    prev = [0.0] * (m + 1)
    acc = 0.0
    for j in range(m):
        acc += b_indel[j]
        prev[j + 1] = acc
    cur = [0.0] * (m + 1)
    for i in range(n):
        ai = a[i]
        del_c = a_indel[i]
        cur[0] = prev[0] + del_c
        row_min = cur[0]
        prev_j = prev[0]
        for j in range(m):
            v = prev_j + sub[(ai, b[j])]
            d = prev[j + 1] + del_c
            if d < v:
                v = d
            ins = cur[j] + b_indel[j]
            if ins < v:
                v = ins
            prev_j = prev[j + 1]
            cur[j + 1] = v
            if v < row_min:
                row_min = v
        if row_min > budget:
            return None
        prev, cur = cur, prev
    d = prev[m]
    return d if d <= budget else None


def align(
    a: Sequence[str],
    b: Sequence[str],
    cfg: PhoneticConfig = DEFAULT_CONFIG,
) -> Tuple[float, Tuple[Tuple[str, str, str, float], ...]]:
    """Full DP with backtrace, returning ``(distance, operations)``.

    Each operation is ``(op, a_symbol, b_symbol, cost)`` with ``op`` in
    ``{"=", "~", "-", "+"}``.  The Space prints this table so a user can see
    *why* 「シンドウ」 was allowed to become 「新藤」.

    Claim: LOW-DAMAGE -- a correction a human can audit is a correction a human
    can veto; opaque rewrites are how cloud post-processing does damage.
    """
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + indel_cost(a, i - 1, cfg)
        bt[i][0] = "-"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + indel_cost(b, j - 1, cfg)
        bt[0][j] = "+"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + substitution_cost(a[i - 1], b[j - 1], cfg)
            dele = dp[i - 1][j] + indel_cost(a, i - 1, cfg)
            ins = dp[i][j - 1] + indel_cost(b, j - 1, cfg)
            best = min(sub, dele, ins)
            dp[i][j] = best
            bt[i][j] = "s" if best == sub else ("-" if best == dele else "+")

    ops: List[Tuple[str, str, str, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j] if (i > 0 and j > 0) else ("-" if j == 0 else "+")
        if move == "s":
            c = substitution_cost(a[i - 1], b[j - 1], cfg)
            ops.append(("=" if a[i - 1] == b[j - 1] else "~", a[i - 1], b[j - 1], c))
            i, j = i - 1, j - 1
        elif move == "-":
            ops.append(("-", a[i - 1], "", indel_cost(a, i - 1, cfg)))
            i -= 1
        else:
            ops.append(("+", "", b[j - 1], indel_cost(b, j - 1, cfg)))
            j -= 1
    ops.reverse()
    return dp[n][m], tuple(ops)


def phoneme_ngrams(phonemes: Sequence[str], n: int = 2) -> List[str]:
    """Padded phoneme n-grams, the retrieval keys of :class:`~mondegreen.index.PhoneticIndex`.

    Claim: UNBOUNDED-VOCAB -- an inverted index over these is what makes
    candidate generation sublinear in glossary size.
    """
    padded = ["^", *phonemes, "$"]
    if len(padded) < n:
        return ["".join(padded)]
    return ["\x1f".join(padded[i : i + n]) for i in range(len(padded) - n + 1)]


def describe_cost_model() -> str:
    """Human-readable dump of the confusion table, for the README and model card.

    Claim: SUPPORT.
    """
    rows = sorted(
        ((sorted(k), v) for k, v in CONFUSION.items()), key=lambda kv: (kv[1], kv[0])
    )
    lines = ["| pair | cost |", "| --- | --- |"]
    lines += [f"| {p[0]} ~ {p[1]} | {c:.2f} |" for p, c in rows]
    return "\n".join(lines)
