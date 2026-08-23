---
title: Mondegreen
emoji: 🗣️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
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

> **この Space はサーバーに何も送りません。** Python (Pyodide / WebAssembly) が
> あなたのブラウザの中で動作し、用語集も書き起こしも Hugging Face のサーバーに到達しません。
> 初回読み込みに 20〜40 秒、その後はネットワークを切っても動きます。
> Gradio ではなく Pyodide を直接叩いています（理由は
> `scripts/build_static_space.py` の docstring に測定結果付きで書いてあります）。

Whisper の書き起こしと用語集を貼ると、訂正結果を差分表示します。各訂正について
**元の音韻列 / 候補 / 音韻距離**を根拠として表示し、**却下された候補も**表示します —
制約の境界がどこにあるかが見えるように。

訂正は「スパンを用語集の語で置き換える」操作**のみ**で、しかも音韻距離が閾値以内の
候補にしか置換できません。用語集にない語を生成することも、文法を「改善」することも、
構造的にできません。

- コード: https://github.com/NagaYu/mondegreen
- モデル: https://huggingface.co/NagaYu/mondegreen
- データセット: https://huggingface.co/datasets/NagaYu/mondegreen-asr-errors

## ブラウザ実行の違い

形態素解析器 (`fugashi`) は C 拡張のためブラウザでは動かず、内蔵の漢字読み表（4,030 字）
を使っています。品詞が分からないと「稼働」（一般語・触ってはいけない）と「進藤」（固有名詞・
訂正対象）を区別できません。そこでこのビルドは**制約側を厳しくして**対応します —
漢字スパンは**ほぼ完全な同音異義語にしか置換できません**。
結果として破壊率は上げずに、再現率だけが少し下がります。フル機能は次で:

```bash
pip install 'git+https://github.com/NagaYu/mondegreen#egg=mondegreen[g2p]'
mondegreen fix transcript.txt --glossary terms.csv
```

## プライバシー

この Space は**テキストのみ**を扱い、音声を受け取りません。
用語集に個人名を含められる以上、**利用者が自分の管理下にあるデータでのみ使用してください。**
他人の会議記録や、同意を得ていない人物の名前を含む用語集を投入しないでください。
