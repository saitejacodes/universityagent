.PHONY: install test lint crawl extract-all eval

install:
	uv venv --python 3.12 && uv pip install -e ".[dev,api]"

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts

crawl:
	.venv/bin/uniagent crawl

MODEL ?= qwen3:8b
OLLAMA ?= http://localhost:11434
extract-all:
	scripts/run_ablation.sh

extract-all-manual:
	.venv/bin/uniagent extract --preset v1 --model llama3.1:8b --base-url $(OLLAMA) --run-name v1-llama3.1
	.venv/bin/uniagent extract --preset v1 --model $(MODEL) --base-url $(OLLAMA) --think false --run-name v1-qwen3
	.venv/bin/uniagent extract --preset retrieval --model $(MODEL) --base-url $(OLLAMA) --think false --run-name retrieval-qwen3
	.venv/bin/uniagent extract --preset full --model $(MODEL) --base-url $(OLLAMA) --think false --run-name full-qwen3

eval:
	.venv/bin/uniagent eval runs/v1-llama3.1 runs/v1-qwen3 runs/retrieval-qwen3 runs/full-qwen3
