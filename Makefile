# Mondegreen — common workflows.
PY ?= python3
N  ?= 500

.PHONY: help install install-all test test-fast lint data gate bench figures app clean demo

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## core install (numpy only)
	$(PY) -m pip install -e .

install-all:  ## everything: g2p, harvest, train, quantise, figures, app
	$(PY) -m pip install -e '.[all]'

test:  ## full test suite
	$(PY) -m pytest tests/ -q

test-fast:  ## skip anything needing heavy optional deps
	$(PY) -m pytest tests/ -q -m 'not slow'

invariants:  ## only the tests that guard a claim
	$(PY) -m pytest tests/ -q -m invariant

data:  ## build a simulated (error, gold) dataset
	$(PY) scripts/harvest_errors.py --mode simulated -n 4000 --out data

data-real:  ## build the real dataset (needs TTS + Whisper)
	$(PY) scripts/harvest_errors.py --mode real -n 2000 --whisper-size small --out data

gate:  ## train and calibrate the conservative gate
	$(PY) -m mondegreen.cli train-gate --pairs data/pairs.jsonl \
		--glossary data/glossary_train.csv -o models/gate.json

bench:  ## run conditions (A)-(E) and render the figures
	$(PY) scripts/run_benchmarks.py -n $(N) --figures

figures:  ## re-render figures from the newest results file
	$(PY) scripts/make_figures.py

app:  ## launch the Gradio Space locally
	$(PY) app.py

demo:  ## reproduce exactly the example in the README
	@printf '進藤さんと中村さんが両氏誤り訂正について話しました。\nミライドライバーの発売日を確認します。\n新藤さんが量子誤り訂正の話をしました。\nシステムの稼働率は九十八パーセントを維持しています。\nご視聴ありがとうございました。\n' > /tmp/mg_demo.txt
	@printf 'surface,reading,category\n新藤,シンドウ,person\n中村,ナカムラ,person\n量子誤り訂正,リョウシアヤマリテイセイ,jargon\nミライドライブ,ミライドライブ,product\n加藤,カトウ,person\n' > /tmp/mg_demo.csv
	$(PY) -m mondegreen.cli fix /tmp/mg_demo.txt --glossary /tmp/mg_demo.csv --gate models/gate.json

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info
