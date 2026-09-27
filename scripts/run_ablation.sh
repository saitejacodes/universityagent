#!/usr/bin/env bash
# Extraction ablation over every configured university (evaluation later filters to the test split).
set -uo pipefail
cd "$(dirname "$0")/.."
X=".venv/bin/uniagent extract"
$X --preset v1        --model llama3.1:8b              --run-name v1-llama3.1
$X --preset v1        --model qwen3:8b --think false   --run-name v1-qwen3
$X --preset retrieval --model qwen3:8b --think false   --run-name retrieval-qwen3
$X --preset full      --model qwen3:8b --think false   --run-name full-qwen3
RUNS="runs/v1-llama3.1 runs/v1-qwen3 runs/retrieval-qwen3 runs/full-qwen3"
# off-the-shelf baseline (scripts/baseline_scrapegraph.py, separate venv) goes first when present
[ -f runs/scrapegraph-qwen3_8b/predictions.json ] && RUNS="runs/scrapegraph-qwen3_8b $RUNS"
.venv/bin/uniagent eval $RUNS --split test --out EVAL_REPORT.md
echo "=== ablation done $(date)"
