"""Loading, validating and saving the private glossary.

The glossary is the *entire* user-supplied input to Mondegreen.  It is a CSV of
surface forms and their readings, and nothing about it is uploaded anywhere.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .types import GlossaryEntry

_HEADER_ALIASES = {
    "surface": {"surface", "term", "word", "表記", "表層", "語", "用語", "見出し"},
    "reading": {"reading", "yomi", "kana", "pron", "読み", "よみ", "カナ", "かな", "読み方"},
    "aliases": {"aliases", "alias", "variants", "異読", "別読み", "別名"},
    "category": {"category", "type", "class", "種別", "カテゴリ", "分類"},
    "weight": {"weight", "prior", "freq", "重み", "頻度"},
    "notes": {"notes", "note", "comment", "備考", "メモ"},
}


def _canonical_header(name: str) -> Optional[str]:
    """Map a CSV header cell to a canonical field name, English or Japanese.

        Claim: SUPPORT -- users write 「読み」 as often as ``reading``.
        """
    key = name.strip().lower().lstrip("﻿")
    for canon, alts in _HEADER_ALIASES.items():
        if key == canon or key in alts:
            return canon
    return None


class Glossary:
    """An ordered, de-duplicated collection of :class:`GlossaryEntry`.

    Claim: UNBOUNDED-VOCAB -- this container is the thing whose size we sweep
    from 100 to 10,000 in the headline figure, and it has no ceiling anywhere.
    """

    def __init__(self, entries: Iterable[GlossaryEntry] = ()) -> None:
        """Build a glossary from an iterable of entries, de-duplicating surfaces.

                Claim: UNBOUNDED-VOCAB.
                """
        self.entries: List[GlossaryEntry] = []
        self._by_surface: Dict[str, GlossaryEntry] = {}
        for e in entries:
            self.add(e)

    # -- container protocol ------------------------------------------------
    def __len__(self) -> int:
        """Claim: SUPPORT."""
        return len(self.entries)

    def __iter__(self) -> Iterator[GlossaryEntry]:
        """Claim: SUPPORT."""
        return iter(self.entries)

    def __getitem__(self, i: int) -> GlossaryEntry:
        """Claim: SUPPORT."""
        return self.entries[i]

    def __contains__(self, surface: object) -> bool:
        """Claim: SUPPORT."""
        return surface in self._by_surface

    # -- mutation ----------------------------------------------------------
    def add(self, entry: GlossaryEntry) -> None:
        """Insert an entry, merging aliases if the surface is already present.

        Claim: TERM-RECALL -- duplicate surfaces with different readings are
        merged rather than shadowing each other, so every attested reading of a
        term stays reachable.
        """
        prior = self._by_surface.get(entry.surface)
        if prior is None:
            self._by_surface[entry.surface] = entry
            self.entries.append(entry)
            return
        merged_aliases = tuple(
            dict.fromkeys((*prior.aliases, *entry.all_readings())).keys()
        )
        merged_aliases = tuple(a for a in merged_aliases if a != prior.reading)
        replacement = GlossaryEntry(
            surface=prior.surface,
            reading=prior.reading,
            aliases=merged_aliases,
            category=prior.category or entry.category,
            weight=max(prior.weight, entry.weight),
            notes=prior.notes or entry.notes,
        )
        idx = self.entries.index(prior)
        self.entries[idx] = replacement
        self._by_surface[prior.surface] = replacement

    def get(self, surface: str) -> Optional[GlossaryEntry]:
        """Claim: SUPPORT."""
        return self._by_surface.get(surface)

    def surfaces(self) -> Tuple[str, ...]:
        """Claim: SUPPORT."""
        return tuple(e.surface for e in self.entries)

    def subset(self, n: int, seed: int = 0) -> "Glossary":
        """Deterministically sample ``n`` entries, for the glossary-size sweep.

        Claim: UNBOUNDED-VOCAB -- the 100 / 1,000 / 10,000 conditions must be
        nested subsets of one glossary, or the sweep confounds size with content.
        """
        import random

        rng = random.Random(seed)
        idx = list(range(len(self.entries)))
        rng.shuffle(idx)
        return Glossary(self.entries[i] for i in sorted(idx[: max(0, n)]))

    def to_rows(self) -> List[Dict[str, str]]:
        """Claim: SUPPORT."""
        return [
            {
                "surface": e.surface,
                "reading": e.reading,
                "aliases": "|".join(e.aliases),
                "category": e.category,
                "weight": f"{e.weight:g}",
                "notes": e.notes,
            }
            for e in self.entries
        ]


def parse_glossary_rows(rows: Iterable[Dict[str, str]]) -> Glossary:
    """Build a :class:`Glossary` from already-parsed dict rows.

    Claim: SUPPORT.
    """
    g = Glossary()
    for row in rows:
        surface = (row.get("surface") or "").strip()
        if not surface or surface.startswith("#"):
            continue
        reading = (row.get("reading") or "").strip()
        aliases_raw = (row.get("aliases") or "").strip()
        aliases = tuple(
            a.strip() for a in aliases_raw.replace(",", "|").split("|") if a.strip()
        )
        try:
            weight = float(row.get("weight") or 1.0)
        except ValueError:
            weight = 1.0
        g.add(
            GlossaryEntry(
                surface=surface,
                reading=reading,
                aliases=aliases,
                category=(row.get("category") or "").strip(),
                weight=weight,
                notes=(row.get("notes") or "").strip(),
            )
        )
    return g


def load_glossary(path: str, reader=None, infer_readings: bool = True) -> Glossary:
    """Load a glossary from CSV / TSV / JSON / JSONL / plain text.

    A missing reading is inferred with the active reader and a warning-free
    fallback, because forcing users to hand-write kana for 10,000 terms would
    make the UNBOUNDED-VOCAB claim untestable in practice.

    Claim: UNBOUNDED-VOCAB -- the glossary format has to scale to 10k rows
    without ceremony.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    g = _parse_text(text, ext)
    if infer_readings:
        g = fill_missing_readings(g, reader=reader)
    return g


def loads_glossary(text: str, fmt: str = ".csv", reader=None, infer_readings: bool = True) -> Glossary:
    """Same as :func:`load_glossary` but from an in-memory string (used by the Space).

    Claim: SUPPORT.
    """
    g = _parse_text(text, fmt)
    if infer_readings:
        g = fill_missing_readings(g, reader=reader)
    return g


def _parse_text(text: str, ext: str) -> Glossary:
    """Parse glossary text in whichever of the supported formats it turns out to be.

        Claim: UNBOUNDED-VOCAB -- the format has to be trivial to produce for 10,000 rows.
        """
    stripped = text.strip()
    if ext in (".json",) or stripped.startswith("["):
        data = json.loads(stripped or "[]")
        return parse_glossary_rows({k: str(v) for k, v in row.items()} for row in data)
    if ext in (".jsonl", ".ndjson"):
        rows = [json.loads(l) for l in stripped.splitlines() if l.strip()]
        return parse_glossary_rows({k: str(v) for k, v in row.items()} for row in rows)

    delimiter = "\t" if ext in (".tsv", ".tab") else ","
    sample = stripped.splitlines()[0] if stripped else ""
    if ext not in (".tsv", ".tab") and "\t" in sample and "," not in sample:
        delimiter = "\t"

    lines = [l for l in stripped.splitlines() if l.strip()]
    if not lines:
        return Glossary()
    header_cells = next(csv.reader([lines[0]], delimiter=delimiter))
    canon = [_canonical_header(c) for c in header_cells]
    if any(c == "surface" for c in canon):
        fieldnames = [c or f"extra{i}" for i, c in enumerate(canon)]
        rdr = csv.DictReader(io.StringIO("\n".join(lines[1:])), fieldnames=fieldnames, delimiter=delimiter)
        return parse_glossary_rows(rdr)
    # Headerless: "surface,reading[,category]" or one bare term per line.
    rows: List[Dict[str, str]] = []
    for cells in csv.reader(lines, delimiter=delimiter):
        if not cells or not cells[0].strip():
            continue
        row = {"surface": cells[0]}
        if len(cells) > 1:
            row["reading"] = cells[1]
        if len(cells) > 2:
            row["category"] = cells[2]
        if len(cells) > 3:
            row["weight"] = cells[3]
        rows.append(row)
    return parse_glossary_rows(rows)


def fill_missing_readings(glossary: Glossary, reader=None) -> Glossary:
    """Infer a kana reading for entries that shipped without one.

    Entries whose reading still cannot be determined (all-kanji surface with no
    table coverage) keep an empty reading and are *skipped* by the index rather
    than indexed under a garbage phoneme string.

    Claim: LOW-DAMAGE -- an entry we cannot read reliably is an entry that must
    never be proposed as a replacement.
    """
    from .reading import get_reader, span_reading_variants, char_kind

    rd = reader or get_reader()
    out = Glossary()
    for e in glossary:
        reading = e.reading
        aliases = list(e.aliases)
        if not reading:
            if all(char_kind(c) in ("hiragana", "katakana") for c in e.surface):
                reading = e.surface
            else:
                toks = rd.tokenize(e.surface)
                variants = span_reading_variants(toks, 0, len(toks), max_variants=4)
                variants = tuple(v for v in variants if "?" not in v)
                if variants:
                    reading = variants[0]
                    aliases.extend(v for v in variants[1:] if v not in aliases)
        out.add(
            GlossaryEntry(
                surface=e.surface,
                reading=reading,
                aliases=tuple(dict.fromkeys(a for a in aliases if a and a != reading)),
                category=e.category,
                weight=e.weight,
                notes=e.notes,
            )
        )
    return out


def save_glossary(glossary: Glossary, path: str) -> None:
    """Write a glossary back out as CSV.

    Claim: SUPPORT.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["surface", "reading", "aliases", "category", "weight", "notes"]
        )
        w.writeheader()
        w.writerows(glossary.to_rows())
