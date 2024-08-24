 for pr in 0 
      do
      python cp_eval.py --LLM_name "gpt4omini" --retriever_name "bge" --poison_rate $pr --rag False --scale 100
      done