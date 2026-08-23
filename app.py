"""Gradio Space for Mondegreen.

Paste a Whisper transcript and a glossary; get back a diff, and -- the part that
matters -- a receipt for every edit: the phoneme string the span was read as, the
candidate it matched, the weighted phonetic distance, the threshold that distance
had to clear, and the alignment that produced it.

The rejected candidates are shown too.  A corrector that only shows you what it
did is asking for trust; one that shows you what it *declined* to do is showing
you where its boundary is.

Run locally::

    pip install 'mondegreen[app]'
    python app.py
"""

from __future__ import annotations

import html
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import gradio as gr

from mondegreen import __version__
from mondegreen.baselines import whisper_prompt_capacity
from mondegreen.corrector import ConstrainedCorrector, CorrectorConfig, SoftPromptCorrector
from mondegreen.gate import ConservativeGate
from mondegreen.glossary import Glossary, loads_glossary
from mondegreen.phonetics import align, kana_to_phonemes, phoneme_string
from mondegreen.reading import get_reader

GATE_PATH = os.environ.get("MONDEGREEN_GATE", "models/gate.json")

EXAMPLE_TRANSCRIPT = """本日の議題について進藤さんから説明がありました。
両氏誤り訂正の実装は、ミライドライバーの次期版に載せる予定です。
中村さんは今月中に仕様書をまとめると言っていました。
ご視聴ありがとうございました。"""

EXAMPLE_GLOSSARY = """surface,reading,category
新藤,シンドウ,person
中村,ナカムラ,person
量子誤り訂正,リョウシアヤマリテイセイ,jargon
ミライドライブ,ミライドライブ,product
逐次復号,チクジフクゴウ,jargon
"""

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
"""


def _build(glossary_text: str, tau: float, gate_threshold: float, remove_halluc: bool):
    glossary = loads_glossary(glossary_text or "", ".csv")
    gate = None
    if os.path.exists(GATE_PATH):
        try:
            gate = ConservativeGate.load(GATE_PATH)
        except Exception:
            gate = None
    cfg = CorrectorConfig(
        tau=float(tau),
        gate_threshold=float(gate_threshold),
        remove_hallucinations=bool(remove_halluc),
    )
    return glossary, ConstrainedCorrector(glossary, cfg, gate=gate)


def _render_diff(result) -> str:
    """Inline diff of the corrected transcript.

    Claim: SUPPORT -- the reviewer has to see the change in context.
    """
    pieces: List[str] = []
    cursor = 0
    events = [("corr", c.start, c.end, c) for c in result.corrections]
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


def _render_evidence(corrector, result) -> str:
    """The receipt panel: phonemes, candidate, distance, threshold, alignment.

    Claim: LOW-DAMAGE -- this is what makes a correction auditable, and it is the
    single most important thing this Space does.
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
        return f"""
<div class="evidence {'' if accepted else 'rejected'}">
  <div style="margin-bottom:4px">{badge}
    <b class="mono">{html.escape(c.original)}</b> &rarr;
    <b class="mono">{html.escape(c.replacement)}</b>
    <span class="note">&nbsp;{html.escape(c.category or '')}</span></div>
  <table class="mono" style="border:0;width:100%">
    <tr><td style="width:150px;color:#5f6368">元の音韻列 / span</td>
        <td>{html.escape(phoneme_string(c.original_phonemes))}</td></tr>
    <tr><td style="color:#5f6368">候補 / candidate</td>
        <td>{html.escape(phoneme_string(c.candidate_phonemes))}</td></tr>
    <tr><td style="color:#5f6368">音韻距離 / distance</td>
        <td><b>{c.norm_distance:.4f}</b> &nbsp;&le;&nbsp; &tau; = {c.threshold:.2f}
            &nbsp;<span class="note">(raw {c.distance:.3f})</span>
            <div style="background:#e8eaed;height:6px;border-radius:3px;margin-top:3px;max-width:280px">
              <div style="background:{'#1a73e8' if accepted else '#9aa0a6'};width:{bar_w}%;height:6px;border-radius:3px"></div>
            </div></td></tr>
    <tr><td style="color:#5f6368">ゲート / gate</td>
        <td>p = {c.gate_prob:.4f} &nbsp;<span class="note">margin {c.margin:.3f} &middot; {html.escape(c.reason)}</span></td></tr>
    <tr><td style="color:#5f6368">整列 / alignment</td><td style="font-size:11.5px">{chain}</td></tr>
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
    for c in result.rejected[:12]:
        rows.append(block(c, False))
    if not rows:
        return '<p class="note">候補スパンなし。用語集の語が音韻的に近い形で現れていません。</p>'
    return "".join(rows)


def run(transcript: str, glossary_text: str, tau: float, gate_threshold: float,
        remove_halluc: bool, show_soft: bool):
    """Main Space callback.

    Claim: TERM-RECALL + LOW-DAMAGE + LOCAL-SPEED.
    """
    if not transcript.strip():
        return "", "", "", ""
    glossary, corrector = _build(glossary_text, tau, gate_threshold, remove_halluc)
    t0 = time.perf_counter()
    result = corrector.correct(transcript)
    elapsed = time.perf_counter() - t0

    diff = _render_diff(result)
    evidence = _render_evidence(corrector, result)

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
    if show_soft:
        sp = SoftPromptCorrector(glossary, token_budget=244)
        prompt, included, dropped = sp.build_prompt(transcript)
        soft = (
            f"# (B) Whisper initial_prompt に用語集を詰めた場合\n"
            f"# 244 トークン上限に収まった語: {included} / {len(glossary)}  (落ちた語: {dropped})\n"
            f"# 落ちた語は、その会議で実際に話されていても復元できません。\n\n{prompt[:1600]}"
            + ("\n…(truncated)" if len(prompt) > 1600 else "")
        )
    return result.text, diff, evidence, json.dumps(stats, ensure_ascii=False, indent=2), soft


# Gradio moved `css`/`theme` from Blocks() to launch() in 6.0 but still accepts
# them on Blocks; building the kwargs here keeps one file working on 4.x-6.x.
_BLOCKS_KW: Dict[str, object] = {"title": "Mondegreen"}
_LAUNCH_KW: Dict[str, object] = {}
if tuple(int(x) for x in gr.__version__.split(".")[:1]) >= (6,):
    _LAUNCH_KW.update(css=CSS, theme=gr.themes.Soft())
else:  # pragma: no cover - older Gradio
    _BLOCKS_KW.update(css=CSS, theme=gr.themes.Soft())


with gr.Blocks(**_BLOCKS_KW) as demo:
    gr.Markdown(
        f"""
# Mondegreen
### 私的な用語集を、音韻的な硬い制約に変換する — ローカルASR訂正器
*A private glossary, compiled into a hard phonetic constraint.*  `v{__version__}`

訂正は「スパンを用語集の語で置き換える」操作**のみ**。しかも
**音韻距離が閾値 τ 以内の候補にしか置換できません**。
用語集にない語を生成することも、文法を「改善」することも、構造的にできません。

音声は扱いません。テキストだけです。**すべてこのマシン上で動きます。**
"""
    )
    with gr.Row():
        with gr.Column(scale=3):
            transcript = gr.Textbox(
                label="Whisper 出力 / ASR transcript", lines=9, value=EXAMPLE_TRANSCRIPT
            )
        with gr.Column(scale=2):
            glossary_text = gr.Textbox(
                label="用語集 / glossary (CSV: surface,reading,category)",
                lines=9, value=EXAMPLE_GLOSSARY,
            )
    with gr.Row():
        tau = gr.Slider(0.05, 0.50, value=0.28, step=0.01,
                        label="τ — 硬い音韻距離の上限 (hard phonetic bound)")
        gate_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                   label="ゲート閾値 (conservative gate)")
    with gr.Row():
        remove_halluc = gr.Checkbox(value=True, label="定型幻聴を除去する")
        show_soft = gr.Checkbox(value=False, label="(B) プロンプト注入版のプロンプトも表示")
        go = gr.Button("訂正する / Correct", variant="primary", scale=2)

    with gr.Tabs():
        with gr.Tab("差分 / Diff"):
            diff_out = gr.HTML()
            text_out = gr.Textbox(label="訂正後のテキスト", lines=6)
        with gr.Tab("根拠 / Evidence"):
            gr.Markdown(
                "各訂正について、**元の音韻列 / 候補 / 音韻距離**を表示します。"
                "却下された候補も表示されます — 制約の境界がどこにあるかが見えるように。"
            )
            evidence_out = gr.HTML()
        with gr.Tab("統計 / Stats"):
            stats_out = gr.Code(language="json")
        with gr.Tab("(B) プロンプト注入 / Soft prompt"):
            soft_out = gr.Code(label="Whisper initial_prompt (244 token cap)")

    gr.Markdown(
        """
---
**プライバシー**: この Space はテキストのみを扱い、音声を受け取りません。
用語集に個人名を含められる以上、**利用者が自分の管理下にあるデータでのみ使用してください。**
他人の会議記録や、同意を得ていない人物の名前を含む用語集を投入しないでください。
"""
    )

    go.click(
        run,
        [transcript, glossary_text, tau, gate_threshold, remove_halluc, show_soft],
        [text_out, diff_out, evidence_out, stats_out, soft_out],
    )
    demo.load(
        run,
        [transcript, glossary_text, tau, gate_threshold, remove_halluc, show_soft],
        [text_out, diff_out, evidence_out, stats_out, soft_out],
    )


if __name__ == "__main__":  # pragma: no cover
    demo.launch(**_LAUNCH_KW)
