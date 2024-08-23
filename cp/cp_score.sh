 for pr in 0 0.2 0.4 0.6 0.8 1
      do
      python cp_eval.py --LLM_name "gpt4o" --retriever_name "bge" --poison_rate $pr --rag True --scale 100
      done