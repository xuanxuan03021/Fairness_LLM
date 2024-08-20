for llm in "gpt4o" "gpt4omini"
      do  
      for pr in 0.0 0.2 0.4 0.6 0.8 1.0
            do
            python RAG_bbq.py --LLM_name $llm --retriever_name "bge" --poison_rate $pr --rag True
            done
      done
