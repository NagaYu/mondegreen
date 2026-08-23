"""Gradio Space for Mondegreen (server-backed).

Paste a Whisper transcript and a glossary; get back a diff, and -- the part that
matters -- a receipt for every edit: the phoneme string the span was read as, the
candidate it matched, the weighted phonetic distance, the threshold that distance
had to clear, and the alignment that produced it. Rejected candidates are shown
too, because seeing what the corrector *declined* to do is how you calibrate
trust in what it did.

All of the rendering lives in :mod:`mondegreen.report`, which has no UI
dependency, so this file and the in-browser Pyodide build
(``scripts/build_static_space.py``) cannot drift apart.

Run locally::

    pip install 'mondegreen[app]'
    python app.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import gradio as gr

from mondegreen import __version__
from mondegreen.report import CSS, EXAMPLE_GLOSSARY, EXAMPLE_TRANSCRIPT, run as run_report

GATE_PATH = os.environ.get("MONDEGREEN_GATE", "models/gate.json")


def run(transcript: str, glossary_text: str, tau: float, gate_threshold: float,
        remove_halluc: bool, show_soft: bool):
    """Main Space callback; thin wrapper over :func:`mondegreen.report.run`.

    Claim: TERM-RECALL + LOW-DAMAGE + LOCAL-SPEED.
    """
    out = run_report(
        transcript, glossary_text, tau, gate_threshold, remove_halluc,
        gate_path=GATE_PATH, include_soft_prompt=show_soft,
    )
    return (
        out["text"],
        out["diff_html"],
        out["evidence_html"],
        json.dumps(out["stats"], ensure_ascii=False, indent=2),
        out["soft_prompt"],
    )


with gr.Blocks(css=CSS, title="Mondegreen", theme=gr.themes.Soft()) as demo:
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
    demo.launch()
