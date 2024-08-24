      for llm in "llama7b" "llama13b" "gpt4o" "gpt4omini"
      do
            for pr in 0.0
            do
            python RAG_framework_script_Holistic.py --LLM_name $llm --poison_rate $pr --rag False
            done
      done
      