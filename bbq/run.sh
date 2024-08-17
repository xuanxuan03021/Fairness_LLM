 for pr in 0 0.2 0.4 0.6 0.8 1.0
      do
      /home/why/IB/bin/python RAG_bbq.py --LLM_name "llama13b" --retriever_name "bge" --poison_rate $pr --rag False
      done
# nohup bash run.sh > output.log 2> error.log &