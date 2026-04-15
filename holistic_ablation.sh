#!/usr/bin/env bash
set -euo pipefail

# Simple ablation runner (same style as pisa_ablation.sh).

SCRIPT="RAG_framework_script_Holistic_ablation_study.py"

for llm in "gpt4omini"; do
  for pr in 1.0; do
    echo "RUN llm=$llm pr=$pr"
    python "$SCRIPT" \
      --LLM_name "$llm" \
      --poison_rate "$pr" \
      --rag True \
      --retriever_type Dense \
      --retriever_name bge \
      --reranker False \
      --rewriter False \
      --summarizer True
  done
done
