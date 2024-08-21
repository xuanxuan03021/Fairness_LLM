 for pr in 0 0.2 0.4 0.6 0.8 1
      do
      python RAG_cp_mengxuan.py --LLM_name "gpt4o" --retriever_name "bge" --poison_rate $pr --rag True
      done
# nohup bash run.sh > output.log 2> error.log &