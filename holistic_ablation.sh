      for llm in "gpt4o" "gpt4omini"
      do
            for pr in 1
            do
            python RAG_framework_script_Holistic_ablation_study.py --LLM_name $llm --poison_rate $pr --rag True --retriever_type "Sparse" --retriever_name "bm25" --reranker False --rewriter False --summarizer False
            done
      done
