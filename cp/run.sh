 for pr in 0 0.2 0.4 0.6 0.8 1
      do
      /home/why/IB/bin/python RAG_cp.py --LLM_name "llama7b" --retriever_name "bge" --poison_rate $pr --rag True
      done
# nohup bash run.sh > output.log 2> error.log &