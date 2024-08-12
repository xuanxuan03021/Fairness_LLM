      for llm in "llama7b" 
      do
            for pr in 1
            do
            python RAG_framework_script_PISA.py --LLM_name $llm --poison_rate $pr --rag True
            done
      done
      