#  for pr in 0 0.2 0.4 0.6 0.8 1
#       do
#       /home/why/IB/bin/python RAG_cp.py --LLM_name "llama7b" --retriever_name "bge" --poison_rate $pr \
#       --rag True --train_attr 'race-color' --test_attr 'gender'
#       done

# stereo_type = ['race-color', 'gender', 'sexual-orientation', 'religion',\
#      'age', 'nationality', 'disability', 'physical-appearance', 'socioeconomic']

# First we consider 'race-color' 'gender' 'sexual-orientation'

# nohup bash run.sh > output_cross.log 2> error_cross.log &


 for train_attr in 'race-color' 'gender' 'sexual-orientation' 'religion' 'age' 'nationality' 'disability' 'physical-appearance' 'socioeconomic' 
    do
    for test_attr in 'race-color' 'gender' 'sexual-orientation' 'religion' 'age' 'nationality' 'disability' 'physical-appearance' 'socioeconomic' 
      do
        if [ "$train_attr" != "$test_attr" ]; then 

          python RAG_cp_cross_mengxuan.py --LLM_name "gpt4o" --retriever_name "bge" --poison_rate 1.0  --rag True --train_attr $train_attr --test_attr $test_attr
        fi
      done
    done