      for llm in "gpt4o" 
      do
            for pr in 0
            do
            python RAG_framework_script_PISA.py --LLM_name $llm --poison_rate $pr --rag False
            done
      done
      