 for pr in 0 1
      do
      /home/why/IB/bin/python RAG_bbq.py --LLM_name "llama7b" --retriever_name "bge" --poison_rate $pr --rag True
      done
# nohup bash run.sh > output.log 2> error.log &