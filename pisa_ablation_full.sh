#!/usr/bin/env bash
set -euo pipefail

# Comprehensive runner for `RAG_framework_script_PISA_ablation_study.py`.
#
# Covers:
# - multiple LLM backends
# - poison rates (a.k.a. unfairness levels)
# - RAG on/off
# - retriever families (Dense: bge/dpr, Sparse: bm25)
# - reranker / rewriter / summarizer toggles
#
# Logs:
# - One log file per configuration under `logs/pisa/`
#
# Notes:
# - gpt* models (and also rewriter/summarizer) require `OPENAI_API_KEY`.

SCRIPT="RAG_framework_script_PISA_ablation_study.py"
LOG_DIR="logs/pisa"
mkdir -p "$LOG_DIR"

# Edit these grids as needed.
LLMS=( "llama7b" "llama13b" "gpt4o" "gpt4omini" )
POISON_RATES=( 0 0.2 0.4 0.6 0.8 1 )
RAG_MODES=( True False )

# Dense retrievers supported by the Python script.
DENSE_RETRIEVERS=( "bge" "dpr" )
# Sparse retrievers supported by the Python script.
SPARSE_RETRIEVERS=( "bm25" )

# Feature toggles.
RERANKERS=( False True )
REWRITERS=( False True )
SUMMARIZERS=( False True )

have_openai_key() {
  [[ -n "${OPENAI_API_KEY:-}" ]]
}

run_one() {
  local llm="$1"
  local poison_rate="$2"
  local rag="$3"
  local retriever_type="$4"
  local retriever_name="$5"
  local reranker="$6"
  local rewriter="$7"
  local summarizer="$8"

  # gpt* models require OpenAI key.
  if [[ "$llm" == gpt* ]] && ! have_openai_key; then
    echo "SKIP (no OPENAI_API_KEY): llm=$llm"
    return 0
  fi

  # In the python script, rewriter and summarizer use OpenAI internally even for llama models.
  if { [[ "$rewriter" == True ]] || [[ "$summarizer" == True ]]; } && ! have_openai_key; then
    echo "SKIP (no OPENAI_API_KEY): rewriter=$rewriter summarizer=$summarizer"
    return 0
  fi

  local tag="llm=${llm}__pr=${poison_rate}__rag=${rag}__rt=${retriever_type}__rn=${retriever_name}__rer=${reranker}__rew=${rewriter}__sum=${summarizer}"
  local ts
  ts="$(date +%Y%m%d-%H%M%S)"
  local log="${LOG_DIR}/${tag}__${ts}.log"

  echo "RUN  ${tag}"
  # shellcheck disable=SC2086
  python "$SCRIPT" \
    --LLM_name "$llm" \
    --poison_rate "$poison_rate" \
    --rag "$rag" \
    --retriever_type "$retriever_type" \
    --retriever_name "$retriever_name" \
    --reranker "$reranker" \
    --rewriter "$rewriter" \
    --summarizer "$summarizer" \
    2>&1 | tee "$log"
}

for llm in "${LLMS[@]}"; do
  for pr in "${POISON_RATES[@]}"; do
    for rag in "${RAG_MODES[@]}"; do
      if [[ "$rag" == False ]]; then
        # When rag=False, retriever/reranker/rewriter/summarizer are ignored by the python code,
        # but we still run one canonical configuration for consistency.
        run_one "$llm" "$pr" False Dense bge False False False
        continue
      fi

      # Dense retrievers.
      for rn in "${DENSE_RETRIEVERS[@]}"; do
        for rer in "${RERANKERS[@]}"; do
          for rew in "${REWRITERS[@]}"; do
            for sum in "${SUMMARIZERS[@]}"; do
              run_one "$llm" "$pr" True Dense "$rn" "$rer" "$rew" "$sum"
            done
          done
        done
      done

      # Sparse retrievers.
      for rn in "${SPARSE_RETRIEVERS[@]}"; do
        for rer in "${RERANKERS[@]}"; do
          for rew in "${REWRITERS[@]}"; do
            for sum in "${SUMMARIZERS[@]}"; do
              run_one "$llm" "$pr" True Sparse "$rn" "$rer" "$rew" "$sum"
            done
          done
        done
      done
    done
  done
done

echo "All done. Logs are under: ${LOG_DIR}/"

