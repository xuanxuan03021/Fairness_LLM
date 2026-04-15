#!/usr/bin/env bash
set -euo pipefail

# Simple ablation runner (close to the original).
# Sweeps a few poison rates across a few LLMs with ONE fixed configuration.

SCRIPT="RAG_framework_script_PISA_ablation_study.py"

for llm in "llama7b" "llama13b" "gpt4o" "gpt4omini"; do
  for pr in 0 0.2 0.4 0.6 0.8 1; do
    echo "RUN llm=$llm pr=$pr"
    python "$SCRIPT" \
      --LLM_name "$llm" \
      --poison_rate "$pr" \
      --rag True \
      --retriever_type Dense \
      --retriever_name bge \
      --reranker False \
      --rewriter True \
      --summarizer False
  done
done
