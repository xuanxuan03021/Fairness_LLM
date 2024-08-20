      for pr in 0 0.2 0.4 0.6 0.8 1.0
      do
      python RAG_framework_script_heart_disease.py --LLM_name "llama7b" --poison_rate $pr --rag True
      done