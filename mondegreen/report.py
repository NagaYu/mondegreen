"""Rendering a correction into human-readable evidence.

This is deliberately free of any UI framework. The Gradio app and the in-browser
(Pyodide) Space both call these functions and inject the returned HTML, so the
evidence panel — the thing that makes a correction auditable — is defined exactly
once and cannot drift between the two frontends.

The evidence is the product. A corrector that only shows you what it changed is
asking for trust; one that also shows you what it *declined* to change is showing
you where its boundary actually is.
"""

from __future__ import annotations

import html
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .corrector import ConstrainedCorrector, CorrectorConfig
from .gate import ConservativeGate
from .glossary import Glossary, loads_glossary
from .phonetics import align, phoneme_string

#: Shared stylesheet for both frontends.
CSS = """
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.diff-del { background: #fce8e6; color: #a50e0e; text-decoration: line-through; padding: 0 2px; border-radius: 3px; }
.diff-ins { background: #e6f4ea; color: #0d652d; padding: 0 2px; border-radius: 3px; font-weight: 600; }
.evidence { border-left: 3px solid #1a73e8; padding: 8px 12px; margin: 8px 0; background: rgba(26,115,232,0.05); border-radius: 0 6px 6px 0; }
.evidence.rejected { border-left-color: #9aa0a6; background: rgba(154,160,166,0.07); }
.badge { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; font-weight:600; }
.badge.ok { background:#e6f4ea; color:#0d652d; }
.badge.no { background:#f1f3f4; color:#5f6368; }
.note { font-size: 12px; color:#5f6368; }
.ev-table { border:0; width:100%; border-collapse: collapse; }
.ev-table td { padding: 2px 4px; vertical-align: top; }
.ev-key { width: 150px; color:#5f6368; }
.bar-bg { background:#e8eaed; height:6px; border-radius:3px; margin-top:3px; max-width:280px; }
.bar-fg { height:6px; border-radius:3px; }
"""

EXAMPLE_TRANSCRIPT = """本日の議題について進藤さんから説明がありました。
両氏誤り訂正の実装は、ミライドライバーの次期版に載せる予定です。
中村さんは今月中に仕様書をまとめると言っていました。
システムの稼働率は九十八パーセントを維持しています。
ご視聴ありがとうございました。"""

EXAMPLE_GLOSSARY = """surface,reading,category
新藤,シンドウ,person
中村,ナカムラ,person
量子誤り訂正,リョウシアヤマリテイセイ,jargon
ミライドライブ,ミライドライブ,product
逐次復号,チクジフクゴウ,jargon
加藤,カトウ,person
"""


def build_corrector(
    glossary_text: str,
    tau: float = 0.28,
    gate_threshold: float = 0.5,
    remove_hallucinations: bool = True,
    gate_path: Optional[str] = "models/gate.json",
) -> Tuple[Glossary, ConstrainedCorrector]:
    """Assemble a corrector from raw CSV text and the UI's knob positions.

    Claim: SUPPORT -- one construction path shared by every frontend.
    """
    import os

    glossary = loads_glossary(glossary_text or "", ".csv")
    gate = None
    if gate_path and os.path.exists(gate_path):
        try:
            gate = ConservativeGate.load(gate_path)
        except Exception:
            gate = None
    cfg = CorrectorConfig(
        tau=float(tau),
        gate_threshold=float(gate_threshold),
        remove_hallucinations=bool(remove_hallucinations),
    )
    return glossary, ConstrainedCorrector(glossary, cfg, gate=gate)


def render_diff(result) -> str:
    """Inline diff of the corrected transcript, deletions and insertions marked.

    Claim: SUPPORT -- the reviewer has to see the change in its context.
    """
    pieces: List[str] = []
    cursor = 0
    events: List[Tuple[str, int, int, Any]] = [
        ("corr", c.start, c.end, c) for c in result.corrections
    ]
    events += [("halluc", s, e, t) for s, e, t in result.removed_hallucinations]
    events.sort(key=lambda x: x[1])
    src = result.source
    for kind, s, e, payload in events:
        if s < cursor:
            continue
        pieces.append(html.escape(src[cursor:s]))
        if kind == "corr":
            pieces.append(f'<span class="diff-del">{html.escape(payload.original)}</span>')
            pieces.append(f'<span class="diff-ins">{html.escape(payload.replacement)}</span>')
        else:
            pieces.append(f'<span class="diff-del">{html.escape(str(payload))}</span>')
        cursor = e
    pieces.append(html.escape(src[cursor:]))
    body = "".join(pieces).replace("\n", "<br>")
    return f'<div class="mono" style="line-height:2.0">{body}</div>'


def render_evidence(corrector: ConstrainedCorrector, result, max_rejected: int = 12) -> str:
    """The receipt panel: phonemes, candidate, distance, threshold, gate, alignment.

    Rejected candidates are rendered too, greyed out. That is the point: seeing
    what the corrector refused to do is how you calibrate trust in what it did.

    Claim: LOW-DAMAGE -- an auditable correction is a correctable one.
    """
    rows: List[str] = []

    def block(c, accepted: bool) -> str:
        _, ops = align(c.original_phonemes, c.candidate_phonemes, corrector.phonetic_config)
        chain = " ".join(
            f'<span title="cost {cost:.2f}">{html.escape(op)}{html.escape(a or "ε")}'
            f'&rarr;{html.escape(b or "ε")}</span>'
            for op, a, b, cost in ops
        )
        badge = ('<span class="badge ok">APPLIED</span>' if accepted
                 else '<span class="badge no">NOT APPLIED</span>')
        bar_w = min(100, int(100 * c.norm_distance / max(c.threshold, 1e-6)))
        colour = "#1a73e8" if accepted else "#9aa0a6"
        return f"""
<div class="evidence {'' if accepted else 'rejected'}">
  <div style="margin-bottom:4px">{badge}
    <b class="mono">{html.escape(c.original)}</b> &rarr;
    <b class="mono">{html.escape(c.replacement)}</b>
    <span class="note">&nbsp;{html.escape(c.category or '')}</span></div>
  <table class="mono ev-table">
    <tr><td class="ev-key">元の音韻列 / span</td><td>{html.escape(phoneme_string(c.original_phonemes))}</td></tr>
    <tr><td class="ev-key">候補 / candidate</td><td>{html.escape(phoneme_string(c.candidate_phonemes))}</td></tr>
    <tr><td class="ev-key">音韻距離 / distance</td>
        <td><b>{c.norm_distance:.4f}</b> &nbsp;&le;&nbsp; &tau; = {c.threshold:.2f}
            &nbsp;<span class="note">(raw {c.distance:.3f})</span>
            <div class="bar-bg"><div class="bar-fg" style="background:{colour};width:{bar_w}%"></div></div></td></tr>
    <tr><td class="ev-key">ゲート / gate</td>
        <td>p = {c.gate_prob:.4f} &nbsp;<span class="note">margin {c.margin:.3f} &middot; {html.escape(c.reason)}</span></td></tr>
    <tr><td class="ev-key">整列 / alignment</td><td style="font-size:11.5px">{chain}</td></tr>
  </table>
</div>"""

    for c in result.corrections:
        rows.append(block(c, True))
    for s, e, txt in result.removed_hallucinations:
        rows.append(
            f'<div class="evidence"><span class="badge ok">REMOVED</span> '
            f'<b class="mono">{html.escape(txt)}</b>'
            f'<div class="note">定型幻聴パターンに一致し、独立したセグメントを構成していたため削除。</div></div>'
        )
    for c in result.rejected[:max_rejected]:
        rows.append(block(c, False))
    if not rows:
        return ('<p class="note">候補スパンなし。用語集の語が音韻的に近い形で'
                '現れていません。</p>')
    return "".join(rows)


def run(
    transcript: str,
    glossary_text: str,
    tau: float = 0.28,
    gate_threshold: float = 0.5,
    remove_hallucinations: bool = True,
    gate_path: Optional[str] = "models/gate.json",
    include_soft_prompt: bool = False,
) -> Dict[str, Any]:
    """Correct a transcript and return everything a UI needs, as plain data.

    Returns ``{text, diff_html, evidence_html, stats, soft_prompt}``. Frameworks
    consume this; they do not reimplement it.

    Claim: TERM-RECALL + LOW-DAMAGE + LOCAL-SPEED.
    """
    if not transcript.strip():
        return {"text": "", "diff_html": "", "evidence_html": "", "stats": {}, "soft_prompt": ""}

    glossary, corrector = build_corrector(
        glossary_text, tau, gate_threshold, remove_hallucinations, gate_path
    )
    t0 = time.perf_counter()
    result = corrector.correct(transcript)
    elapsed = time.perf_counter() - t0

    from .baselines import whisper_prompt_capacity

    cap = whisper_prompt_capacity(glossary) if len(glossary) else {}
    stats = {
        "corrections_applied": len(result.corrections),
        "candidates_rejected_by_gate": len(result.rejected),
        "hallucinations_removed": len(result.removed_hallucinations),
        "glossary_terms": len(glossary),
        "seconds": round(elapsed, 4),
        "chars_per_second": round(len(transcript) / elapsed, 1) if elapsed else None,
        "reader": corrector.reader.name,
        "gate": corrector.gate.name,
        "gate_trained": bool(getattr(corrector.gate, "is_trained", False)),
        "tau": tau,
        "gate_threshold": gate_threshold,
        "whisper_prompt_would_fit": cap.get("terms_included"),
        "whisper_prompt_would_drop": cap.get("terms_dropped"),
        "spans_enumerated": result.stats.get("spans_enumerated"),
        "everything_ran_locally": True,
    }

    soft = ""
    if include_soft_prompt and len(glossary):
        from .corrector import SoftPromptCorrector

        sp = SoftPromptCorrector(glossary, token_budget=244)
        prompt, included, dropped = sp.build_prompt(transcript)
        soft = (
            f"# (B) Whisper initial_prompt に用語集を詰めた場合\n"
            f"# 244 トークン上限に収まった語: {included} / {len(glossary)}  (落ちた語: {dropped})\n"
            f"# 落ちた語は、その会議で実際に話されていても復元できません。\n\n"
            f"{prompt[:1600]}" + ("\n…(truncated)" if len(prompt) > 1600 else "")
        )

    return {
        "text": result.text,
        "diff_html": render_diff(result),
        "evidence_html": render_evidence(corrector, result),
        "stats": stats,
        "soft_prompt": soft,
    }


def run_json(*args, **kwargs) -> str:
    """:func:`run`, serialised — the entry point the browser build calls.

    Claim: SUPPORT.
    """
    return json.dumps(run(*args, **kwargs), ensure_ascii=False)
