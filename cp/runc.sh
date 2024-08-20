 for pr in 0 0.2 0.4 0.6 0.8 1
      do
      /home/why/IB/bin/python RAG_cp.py --LLM_name "llama7b" --retriever_name "bge" --poison_rate $pr \
      --rag True --train_attr 'race-color' --test_attr 'gender'
      done

# stereo_type = ['race-color', 'gender', 'sexual-orientation', 'religion',\
#      'age', 'nationality', 'disability', 'physical-appearance', 'socioeconomic']

# First we consider 'race-color' 'gender' 'sexual-orientation'

# nohup bash run.sh > output_cross.log 2> error_cross.log &