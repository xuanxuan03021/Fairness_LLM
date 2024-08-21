 for pr in 0 
      do
      python RAG_cp_mengxuan.py --LLM_name "gpt4omini" --retriever_name "bge" --poison_rate $pr --rag False
      done
# nohup bash run.sh > output.log 2> error.log &