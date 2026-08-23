"""Phonology: 長音, 促音, 拗音, 撥音, and the weighted distance.

These are unit tests with hand-checked expected values.  If the mora table or the
canonicalisation rules drift, every downstream number drifts with them, so the
expectations here are written out in full rather than computed.
"""

from __future__ import annotations

import pytest

from mondegreen.phonetics import (
    DEFAULT_CONFIG,
    PhoneticConfig,
    align,
    describe_cost_model,
    indel_cost,
    kana_to_phonemes,
    mora_count,
    normalized_distance,
    phoneme_ngrams,
    phoneme_string,
    phonetic_distance,
    substitution_cost,
    to_hiragana,
    to_katakana,
)


def ph(kana: str) -> str:
    return phoneme_string(kana_to_phonemes(kana))


class TestMoraTable:
    def test_basic_gojuon(self):
        assert ph("カキクケコ") == "k a k i k u k e k o"
        assert ph("サシスセソ") == "s a sh i s u s e s o"
        assert ph("タチツテト") == "t a ch i ts u t e t o"
        assert ph("ハヒフヘホ") == "h a h i f u h e h o"

    def test_youon_is_one_palatalised_onset(self):
        """拗音: キャ is ``ky a``, never ``k y a``."""
        assert ph("キャ") == "ky a"
        assert ph("シュ") == "sh u"
        assert ph("チョ") == "ch o"
        assert ph("ジャ") == "j a"
        assert ph("リョウ") == "ry o o"

    def test_sokuon_is_its_own_symbol(self):
        """促音: ッ becomes Q."""
        assert ph("ガッコウ") == "g a Q k o o"
        assert ph("キャッシュ") == "ky a Q sh u"
        assert ph("イッポン") == "i Q p o N"

    def test_hatsuon_is_its_own_symbol(self):
        """撥音: ン becomes N."""
        assert ph("シンドウ") == "sh i N d o o"
        assert ph("カンパン") == "k a N p a N"

    def test_chouon_expands_to_the_previous_vowel(self):
        """長音: ー copies the preceding vowel, and オウ folds to オオ."""
        assert ph("トーキョー") == ph("トウキョウ") == "t o o ky o o"
        assert ph("コンピューター") == "k o N py u u t a a"
        assert ph("セイ") == ph("セー") == "s e e"

    def test_leading_chouon_is_dropped_not_crashed(self):
        assert ph("ーアイ") == "a i"

    def test_foreign_sounds(self):
        assert ph("ファイル") == "f a i r u"
        assert ph("ヴァイオリン") == "v a i o r i N"
        assert ph("ツァー") == "ts a a"
        assert ph("ウィーク") == "w i i k u"
        assert ph("ジェット") == "j e Q t o"

    def test_hiragana_and_katakana_agree(self):
        for h, k in [("しんどう", "シンドウ"), ("がっこう", "ガッコウ"), ("きゃっしゅ", "キャッシュ")]:
            assert ph(h) == ph(k)

    def test_halfwidth_and_dakuten_folding(self):
        assert to_katakana("ｼﾝﾄﾞｳ") == "シンドウ"
        assert ph("ｼﾝﾄﾞｳ") == "sh i N d o o"

    def test_latin_and_digits_get_spoken_forms(self):
        assert ph("NHK") == ph("エヌエイチケー")
        assert ph("3") == ph("サン")

    def test_unreadable_characters_become_unknown(self):
        out = kana_to_phonemes("蘇")
        assert "?" in out

    def test_roundtrip_kana_case(self):
        assert to_hiragana(to_katakana("しんどう")) == "しんどう"

    def test_mora_count(self):
        assert mora_count(kana_to_phonemes("シンドウ")) == 4      # シ ン ド ウ
        assert mora_count(kana_to_phonemes("ガッコウ")) == 4      # ガ ッ コ ウ
        assert mora_count(kana_to_phonemes("キャッシュ")) == 3    # キャ ッ シュ


class TestDistance:
    def test_identical_is_zero(self):
        for w in ["シンドウ", "ナカムラ", "ミライドライブ"]:
            assert normalized_distance(kana_to_phonemes(w), kana_to_phonemes(w)) == 0.0

    def test_orthographic_variants_are_free(self):
        pairs = [("トウキョウ", "トーキョー"), ("シンドウ", "シンドー"), ("セイサク", "セーサク")]
        for a, b in pairs:
            assert normalized_distance(kana_to_phonemes(a), kana_to_phonemes(b)) == 0.0

    def test_voicing_is_cheap(self):
        d = normalized_distance(kana_to_phonemes("サトウ"), kana_to_phonemes("サドウ"))
        assert 0 < d < 0.10

    def test_unrelated_words_are_expensive(self):
        d = normalized_distance(kana_to_phonemes("タナカ"), kana_to_phonemes("ヤマダ"))
        assert d > 0.30

    def test_symmetry(self):
        a, b = kana_to_phonemes("ナカムラ"), kana_to_phonemes("ナカムタ")
        assert phonetic_distance(a, b) == pytest.approx(phonetic_distance(b, a))

    def test_is_not_a_metric_and_does_not_claim_to_be(self):
        """The triangle inequality does *not* hold, by design -- and nothing needs it.

        Indel costs are context-dependent (deleting one of a doubled vowel is
        cheap; deleting a lone vowel is not), which breaks the inequality.  This
        test pins that down so nobody later "fixes" the cost model to satisfy a
        property the system never uses.  What the index actually relies on is the
        length lower bound, tested below.
        """
        a = kana_to_phonemes("シントウ")
        b = kana_to_phonemes("シンドウ")
        c = kana_to_phonemes("ナカムラ")
        assert phonetic_distance(a, c) > phonetic_distance(a, b) + phonetic_distance(b, c)

    def test_length_lower_bound_holds(self):
        """``d(a,b) >= |len(a)-len(b)| * min_indel`` -- the index's pruning bound.

        Every alignment contains at least ``|len(a)-len(b)|`` indels, so this is
        what makes the length prefilter admissible.  If it ever failed, the index
        would silently start dropping legal candidates.
        """
        min_indel = min(
            DEFAULT_CONFIG.indel_R, DEFAULT_CONFIG.indel_long_vowel,
            DEFAULT_CONFIG.indel_Q, DEFAULT_CONFIG.indel_N,
            DEFAULT_CONFIG.indel_epenthetic, DEFAULT_CONFIG.indel_glide,
            DEFAULT_CONFIG.indel_vowel, DEFAULT_CONFIG.indel_consonant,
        )
        words = ["シンドウ", "シントウ", "ナカムラ", "ミライドライブ", "リョウシアヤマリテイセイ",
                 "ア", "コンピューター", "ガッコウ"]
        for x in words:
            for y in words:
                a, b = kana_to_phonemes(x), kana_to_phonemes(y)
                assert phonetic_distance(a, b) >= abs(len(a) - len(b)) * min_indel - 1e-9

    def test_non_negative_and_zero_only_for_equal_sequences(self):
        words = ["シンドウ", "ナカムラ", "ガッコウ", "ミライドライブ"]
        for x in words:
            for y in words:
                a, b = kana_to_phonemes(x), kana_to_phonemes(y)
                d = phonetic_distance(a, b)
                assert d >= 0.0
                assert (d == 0.0) == (a == b)

    def test_substitution_costs_are_symmetric_and_zero_on_diagonal(self):
        for a in ("k", "g", "s", "sh", "N", "a", "o", "Q"):
            assert substitution_cost(a, a) == 0.0
            for b in ("k", "g", "s", "sh", "N", "a", "o", "Q"):
                assert substitution_cost(a, b) == substitution_cost(b, a)

    def test_long_vowel_indel_is_cheap_in_context(self):
        """Deleting one of a doubled vowel is a length change, not a lost mora."""
        seq = ["t", "o", "o", "ky", "o", "o"]
        assert indel_cost(seq, 2) < indel_cost(["t", "o", "ky", "a"], 1)

    def test_alignment_reconstructs_the_distance(self):
        a, b = kana_to_phonemes("シントウ"), kana_to_phonemes("シンドウ")
        total, ops = align(a, b)
        assert total == pytest.approx(sum(op[3] for op in ops))
        assert total == pytest.approx(phonetic_distance(a, b))

    def test_config_changes_change_the_answer(self):
        strict = PhoneticConfig(fold_ou_to_oo=False, expand_long_vowel_mark=False)
        assert kana_to_phonemes("トーキョー", strict) != kana_to_phonemes("トウキョウ", strict)
        assert kana_to_phonemes("トーキョー") == kana_to_phonemes("トウキョウ")

    def test_config_roundtrip(self):
        assert PhoneticConfig.from_dict(DEFAULT_CONFIG.to_dict()) == DEFAULT_CONFIG
        custom = PhoneticConfig(vowel_vowel_default=0.42, indel_Q=0.11)
        assert PhoneticConfig.from_dict(custom.to_dict()) == custom


class TestNgrams:
    def test_ngrams_are_padded(self):
        grams = phoneme_ngrams(("a", "b"), 2)
        assert len(grams) == 3  # ^a ab b$

    def test_short_sequences_do_not_crash(self):
        assert phoneme_ngrams((), 2)
        assert phoneme_ngrams(("a",), 3)


def test_cost_model_is_documentable():
    md = describe_cost_model()
    assert md.startswith("| pair | cost |")
    assert "k ~ g" in md or "g ~ k" in md
