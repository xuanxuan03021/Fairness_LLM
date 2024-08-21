      for llm in "llama13b"
      do
            for pr in 0.2 0.4 0.6 0.8 1
            do
            python RAG_framework_script_Holistic.py --LLM_name $llm --poison_rate $pr --rag True
            done
      done
      