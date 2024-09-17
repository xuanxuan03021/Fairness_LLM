
for llm in "gpt4o" "gpt4omini"
      do  
      for pr in 0.0 
            do
            python RAG_bbq_mengxuan.py --LLM_name $llm --retriever_name "bge" --poison_rate $pr --rag False
            done
      done
# nohup bash run.sh > output.log 2> error.log &