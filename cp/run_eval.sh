 for train_attr in 'race-color' 'gender' 'sexual-orientation'  'age'  'disability' 'physical-appearance' 'socioeconomic'
  do
    for test_attr in 'race-color' 'gender' 'sexual-orientation'  'age'  'disability' 'physical-appearance' 'socioeconomic'
    do
      if [ "$train_attr" != "$test_attr" ]; then 
        python cp_eval_cross.py --LLM_name "llama7b" --poison_rate 1  --rag True --train_attr $train_attr --test_attr $test_attr
      fi
    done
  done

#     stereo_type = ['Age', 'Religion', 'Race_x_SES', 'Physical_appearance', 'SES', 'Race_ethnicity', 'Race_x_gender', \
#                     'Disability_status', 'Nationality', 'Sexual_orientation', 'Gender_identity']
# stereo_type = ['race-color', 'gender', 'sexual-orientation', 'religion',\
#      'age', 'nationality', 'disability', 'physical-appearance', 'socioeconomic']

# First we consider 'race-color' 'gender' 'sexual-orientation'

# nohup bash run_eval.sh > output_eval.log 2> error_eval.log &
# python cp_eval_cross.py --LLM_name "llama7b" --poison_rate 1  --rag True --train_attr 'race-color' --test_attr 'gender'