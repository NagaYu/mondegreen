"""Shared fixtures.  Everything here is deterministic and dependency-free."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig
from mondegreen.glossary import Glossary, loads_glossary
from mondegreen.harvest import ErrorHarvester, GlossaryBuilder, SentenceFactory
from mondegreen.index import PhoneticIndex

SMALL_CSV = """surface,reading,category
新藤,シンドウ,person
中村,ナカムラ,person
量子誤り訂正,リョウシアヤマリテイセイ,jargon
ミライドライブ,ミライドライブ,product
逐次復号,チクジフクゴウ,jargon
佐藤太郎,サトウタロウ,person
"""


@pytest.fixture(scope="session")
def small_glossary() -> Glossary:
    return loads_glossary(SMALL_CSV, ".csv")


@pytest.fixture(scope="session")
def empty_glossary() -> Glossary:
    return Glossary()


@pytest.fixture(scope="session")
def small_index(small_glossary) -> PhoneticIndex:
    return PhoneticIndex(small_glossary)


@pytest.fixture
def corrector(small_glossary) -> ConstrainedCorrector:
    return ConstrainedCorrector(small_glossary, CorrectorConfig(gate_threshold=0.5))


@pytest.fixture(scope="session")
def synthetic_corpus():
    """A small train/test corpus with strictly disjoint glossaries.

    Session-scoped because generating it is the slowest thing in the suite.
    """
    gb = GlossaryBuilder(seed=4242)
    train_g, test_g = gb.build_pair(120, 60, seed=4242)
    train_s = SentenceFactory(seed=1).build(train_g, 120)
    test_s = SentenceFactory(seed=2).build(test_g, 90)
    h = ErrorHarvester(seed=4242)
    return {
        "train_glossary": train_g,
        "test_glossary": test_g,
        "train_pairs": h.harvest_simulated(train_s, train_g, split="train"),
        "test_pairs": h.harvest_simulated(test_s, test_g, split="test"),
    }


CLEAN_SENTENCES = [
    "本日の会議では予算の確認を行いました。",
    "来週の火曜日までに資料をまとめてください。",
    "システムの稼働率は九十八パーセントを維持しています。",
    "えーと、その件については後ほど相談させてください。",
    "お忙しいところ恐れ入りますが、ご確認をお願いいたします。",
    "気温が下がってきたので暖房を入れました。",
]
