"""Unit tests for the index, glossary, metrics, hallucination filter and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from mondegreen.baselines import count_tokens, whisper_prompt_capacity
from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig, select_non_overlapping
from mondegreen.glossary import Glossary, loads_glossary, save_glossary
from mondegreen.hallucination import DEFAULT_PATTERNS, HallucinationFilter
from mondegreen.harvest import CORPUS_LICENSES, GlossaryBuilder, license_for
from mondegreen.index import PhoneticIndex
from mondegreen.metrics import (
    cer, classify_span_edit, corpus_cer, count_occurrences,
    hallucination_removal_rate, levenshtein, term_recall,
)
from mondegreen.phonetics import kana_to_phonemes
from mondegreen.types import GlossaryEntry


class TestGlossary:
    def test_csv_with_header(self):
        g = loads_glossary("surface,reading,category\n新藤,シンドウ,person\n", ".csv")
        assert len(g) == 1 and g[0].reading == "シンドウ"

    def test_headerless_csv(self):
        g = loads_glossary("新藤,シンドウ\n中村,ナカムラ\n", ".csv")
        assert len(g) == 2

    def test_bare_terms_get_inferred_readings(self):
        g = loads_glossary("量子誤り訂正\nミライドライブ\n", ".csv")
        assert len(g) == 2
        assert all(e.reading for e in g), [e.surface for e in g]

    def test_japanese_headers(self):
        g = loads_glossary("表記,読み,種別\n新藤,シンドウ,人名\n", ".csv")
        assert g[0].surface == "新藤" and g[0].category == "人名"

    def test_json_and_jsonl(self):
        j = loads_glossary('[{"surface":"新藤","reading":"シンドウ"}]', ".json")
        jl = loads_glossary('{"surface":"新藤","reading":"シンドウ"}', ".jsonl")
        assert j[0].surface == jl[0].surface == "新藤"

    def test_duplicate_surfaces_merge_readings(self):
        g = loads_glossary("新藤,シンドウ\n新藤,シントウ\n", ".csv")
        assert len(g) == 1
        assert "シントウ" in g[0].all_readings()

    def test_subset_is_nested_and_deterministic(self):
        g = GlossaryBuilder(seed=1).build(200)
        a, b = g.subset(50), g.subset(50)
        assert a.surfaces() == b.surfaces()
        assert set(g.subset(20).surfaces()) <= set(g.subset(60).surfaces())

    def test_roundtrip_through_disk(self, tmp_path):
        g = GlossaryBuilder(seed=3).build(40)
        p = tmp_path / "g.csv"
        save_glossary(g, str(p))
        from mondegreen.glossary import load_glossary

        back = load_glossary(str(p))
        assert back.surfaces() == g.surfaces()

    def test_empty_input(self):
        assert len(loads_glossary("", ".csv")) == 0


class TestIndex:
    def test_finds_exact_homophone(self, small_index):
        hits = small_index.query_reading("シンドウ", 0.28)
        assert hits and hits[0].entry.surface == "新藤"
        assert hits[0].norm_distance == 0.0

    def test_returns_nothing_for_unrelated(self, small_index):
        assert small_index.query_reading("ゼンゼンカンケイナイ", 0.28) == []

    def test_top_k_is_respected(self, small_index):
        assert len(small_index.query_reading("シンドウ", 0.9, top_k=2)) <= 2

    def test_one_candidate_per_entry(self, small_index):
        hits = small_index.query_reading("シンドウ", 0.9, top_k=10)
        assert len({h.entry.surface for h in hits}) == len(hits)

    def test_max_raw_cap_is_enforced(self, small_index):
        for c in small_index.query_reading("シンドウ", 0.9, top_k=10, max_raw=0.3):
            assert c.distance <= 0.3 + 1e-9

    def test_mora_ratio_cap_is_enforced(self, small_index):
        q = kana_to_phonemes("シンドウ")
        for c in small_index.query(q, 0.9, top_k=10, max_mora_ratio=0.1):
            from mondegreen.phonetics import mora_count

            a, b = mora_count(q), mora_count(c.term_phonemes)
            assert abs(a - b) / max(a, b) <= 0.1 + 1e-9

    def test_entries_without_readings_are_not_indexed(self):
        g = Glossary([GlossaryEntry("XX", "")])
        ix = PhoneticIndex(g)
        assert len(ix) == 0

    def test_stats_and_length_range(self, small_index):
        st = small_index.stats()
        assert st["reachable_terms"] == 6
        lo, hi = small_index.length_range
        assert 0 < lo <= hi

    def test_scales_to_ten_thousand(self):
        g = GlossaryBuilder(seed=5).build(10000)
        ix = PhoneticIndex(g)
        assert ix.n_terms == 10000
        hits = ix.query_reading(g[0].reading, 0.28, top_k=3)
        assert any(h.entry.surface == g[0].surface for h in hits)


class TestHallucination:
    def test_removes_standalone_canned_phrase(self):
        f = HallucinationFilter()
        out, hits = f.apply("予算の確認です。ご視聴ありがとうございました。")
        assert out == "予算の確認です。" and len(hits) == 1

    def test_keeps_embedded_occurrence(self):
        f = HallucinationFilter()
        out, hits = f.apply("彼はご視聴ありがとうございましたと言った。")
        assert hits == [] and "ご視聴" in out

    def test_keeps_a_single_genuine_thanks(self):
        f = HallucinationFilter()
        out, _ = f.apply("会議を終わります。ありがとうございました。")
        assert "ありがとうございました" in out

    def test_fuzzy_variant_is_caught(self):
        f = HallucinationFilter()
        out, hits = f.apply("資料です。ご清聴ありがとうございました。")
        assert hits and out == "資料です。"

    def test_empty_and_whitespace(self):
        f = HallucinationFilter()
        assert f.apply("")[0] == ""
        assert f.apply("   ")[1] == []


class TestMetrics:
    def test_levenshtein_and_cer(self):
        assert levenshtein(list("abc"), list("abd")) == 1
        assert cer("abc", "abd") == pytest.approx(1 / 3)
        assert cer("", "") == 0.0

    def test_corpus_cer_is_micro_averaged(self):
        # One long perfect sentence must outweigh one short broken one.
        assert corpus_cer(["a" * 100, "bb"], ["a" * 100, "xx"]) == pytest.approx(2 / 102)

    def test_count_occurrences_is_non_overlapping(self):
        assert count_occurrences("aaaa", "aa") == 2

    def test_term_recall_caps_at_gold_count(self):
        r = term_recall(["新藤です"], ["新藤新藤です"], ["新藤"])
        assert r.recall == 1.0 and r.recovered == 1

    def test_classify_span_edit_labels(self):
        gold = "新藤さんが量子誤り訂正の話をしました"
        base = "進藤さんが両氏誤り訂正の話をしました"
        assert classify_span_edit(gold, base, (0, 2), "新藤") == "repair"
        assert classify_span_edit(gold, base, (0, 2), "進藤") == "no-op"
        assert classify_span_edit(gold, gold, (0, 2), "中村") == "damage"

    def test_hallucination_metrics_report_false_removals(self):
        r = hallucination_removal_rate(
            ["会議です。ありがとうございました。"],
            ["会議です。ありがとうございました。"],
            ["会議です。"],
            ["ありがとうございました"],
        )
        assert r["false_removals"] == 1.0


class TestBaselines:
    def test_token_counting_is_monotone(self):
        assert count_tokens("新藤") < count_tokens("新藤中村量子誤り訂正")

    def test_prompt_capacity_drops_terms_past_the_ceiling(self):
        g = GlossaryBuilder(seed=9).build(2000)
        cap = whisper_prompt_capacity(g)
        assert cap["terms_dropped"] > 0
        assert cap["tokens_used"] <= 244
        assert cap["terms_included"] < 2000

    def test_small_glossary_fits_entirely(self, small_glossary):
        cap = whisper_prompt_capacity(small_glossary)
        assert cap["terms_dropped"] == 0 and cap["coverage"] == 1.0


class TestLicensing:
    def test_known_corpora_have_licences(self):
        for name in CORPUS_LICENSES:
            rec = license_for(name)
            assert rec["license"]

    def test_unknown_corpus_raises(self):
        with pytest.raises(KeyError, match="verified"):
            license_for("some-scraped-website")

    def test_aozora_demands_per_work_verification(self):
        assert "REQUIRED" in license_for("aozora")["verify"]


class TestOverlapResolution:
    def test_prefers_the_higher_scoring_of_two_overlaps(self):
        assert select_non_overlapping([(0, 5, 1.0), (3, 8, 2.0)]) == [1]

    def test_takes_both_when_disjoint(self):
        assert sorted(select_non_overlapping([(0, 3, 1.0), (3, 6, 1.0)])) == [0, 1]

    def test_empty(self):
        assert select_non_overlapping([]) == []


class TestCLI:
    def _run(self, args, **kw):
        env = dict(os.environ, PYTHONPATH=os.getcwd())
        return subprocess.run(
            [sys.executable, "-m", "mondegreen.cli", *args],
            capture_output=True, text=True, env=env, **kw
        )

    def test_fix_corrects_and_reports(self, tmp_path):
        t = tmp_path / "t.txt"
        t.write_text("進藤さんが両氏誤り訂正の話をしました。", encoding="utf-8")
        g = tmp_path / "g.csv"
        g.write_text("surface,reading\n新藤,シンドウ\n量子誤り訂正,リョウシアヤマリテイセイ\n", encoding="utf-8")
        r = self._run(["fix", str(t), "--glossary", str(g)])
        assert r.returncode == 0, r.stderr
        assert "新藤さんが量子誤り訂正の話をしました。" in r.stdout
        assert "2 corrections" in r.stderr

    def test_fix_with_no_glossary_is_the_identity(self, tmp_path):
        t = tmp_path / "t.txt"
        t.write_text("今日はいい天気ですね。", encoding="utf-8")
        r = self._run(["fix", str(t), "-q"])
        assert r.returncode == 0
        assert r.stdout.strip() == "今日はいい天気ですね。"

    def test_json_output_is_valid(self, tmp_path):
        t = tmp_path / "t.txt"
        t.write_text("進藤さんです。", encoding="utf-8")
        g = tmp_path / "g.csv"
        g.write_text("surface,reading\n新藤,シンドウ\n", encoding="utf-8")
        r = self._run(["fix", str(t), "--glossary", str(g), "--json"])
        payload = json.loads(r.stdout)
        assert payload["text"] == "新藤さんです。"
        assert payload["corrections"][0]["replacement"] == "新藤"

    def test_explain_shows_evidence(self, tmp_path):
        t = tmp_path / "t.txt"
        t.write_text("進藤さんです。", encoding="utf-8")
        g = tmp_path / "g.csv"
        g.write_text("surface,reading\n新藤,シンドウ\n", encoding="utf-8")
        r = self._run(["explain", str(t), "--glossary", str(g)])
        assert "ACCEPT" in r.stdout and "distance" in r.stdout and "sh i N d o o" in r.stdout

    def test_build_glossary_makes_disjoint_splits(self, tmp_path):
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        r = self._run(["build-glossary", "-n", "50", "--n-test", "20",
                       "-o", str(a), "--test-out", str(b)])
        assert r.returncode == 0, r.stderr
        from mondegreen.glossary import load_glossary

        assert not (set(load_glossary(str(a)).surfaces()) & set(load_glossary(str(b)).surfaces()))

    def test_info_reports_backends(self):
        r = self._run(["info"])
        payload = json.loads(r.stdout)
        assert "reader" in payload and "backends" in payload and payload["machine"]

    def test_version(self):
        r = self._run(["--version"])
        assert "mondegreen" in r.stdout
