#  for pr in 0 0.2 0.4 0.6 0.8 1
#       do
#       /home/why/IB/bin/python RAG_bbq_cross.py --LLM_name "llama7b" --retriever_name "bge" --poison_rate $pr \
#       --rag True --train_attr 'Race_ethnicity' --test_attr 'Gender_identity'
#       done

# stereo_type = ['Age', 'Religion', 'Race_x_SES', 'Physical_appearance', 'SES', 'Race_ethnicity', 'Race_x_gender', \
#                    'Disability_status', 'Nationality', 'Sexual_orientation', 'Gender_identity']

# First we consider relationship between 'Race_ethnicity' 'Gender_identity' 'Religion'
# nohup bash run.sh > output_cross.log 2> error_cross.log &



 for train_attr in 'Religion'
    do
    for test_attr in 'Age' 'Religion' 'Race_x_SES' 'Physical_appearance' 'SES' 'Race_ethnicity' 'Race_x_gender' 'Disability_status' 'Nationality' 'Sexual_orientation' 'Gender_identity'
      do
      python RAG_bbq_cross_mengxuan.py --LLM_name "gpt4o" --retriever_name "bge" --poison_rate 1  --rag True --train_attr $train_attr --test_attr $test_attr
      done
      done

#     stereo_type = ['Age', 'Religion', 'Race_x_SES', 'Physical_appearance', 'SES', 'Race_ethnicity', 'Race_x_gender', \
#                     'Disability_status', 'Nationality', 'Sexual_orientation', 'Gender_identity']