---
title: Mondegreen
emoji: 🗣️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: 用語集を音韻的な硬い制約に変換するローカルASR訂正器
tags:
  - asr-error-correction
  - japanese
  - whisper
  - phonetics
---

# Mondegreen

**用語集を、お願いではなく制約にする。**

Whisper の書き起こしと用語集を貼ると、訂正結果を差分表示します。各訂正について
**元の音韻列 / 候補 / 音韻距離**を根拠として表示し、**却下された候補も**表示します —
制約の境界がどこにあるかが見えるように。

訂正は「スパンを用語集の語で置き換える」操作**のみ**で、しかも音韻距離が閾値以内の
候補にしか置換できません。用語集にない語を生成することも、文法を「改善」することも、
構造的にできません。

- コード: https://github.com/NagaYu/mondegreen
- モデル: https://huggingface.co/NagaYu/mondegreen
- データセット: https://huggingface.co/datasets/NagaYu/mondegreen-asr-errors

## プライバシー

この Space は**テキストのみ**を扱い、音声を受け取りません。
用語集に個人名を含められる以上、**利用者が自分の管理下にあるデータでのみ使用してください。**
他人の会議記録や、同意を得ていない人物の名前を含む用語集を投入しないでください。
