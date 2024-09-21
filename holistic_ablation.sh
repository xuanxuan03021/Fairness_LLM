      for llm in "llama7b" "llama13b" "gpt4o" "gpt4omini"
      do
            for pr in 1
            do
            python RAG_framework_script_Holistic_ablation_study.py --LLM_name $llm --poison_rate $pr --rag True --retriever_type "Dense" --retriever_name "bge" --reranker False --rewriter False --summarizer True
            done
      done
