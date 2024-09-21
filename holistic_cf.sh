 
#  'ability', 'age', 'body_type', 'characteristics', 'cultural',
#        'gender_and_sex', 'nationality', 'nonce', 'political_ideologies',
#        'race_ethnicity', 'religion', 'sexual_orientation',
 
 for train_attr in  'socioeconomic_class'
  do
    for test_attr in 'ability' 'age' 'body_type' 'characteristics' 'cultural' 'gender_and_sex' 'nationality' 'nonce' 'political_ideologies' 'race_ethnicity' 'religion' 'sexual_orientation' 'socioeconomic_class'
    do
      if [ "$train_attr" != "$test_attr" ]; then 
        python RAG_framework_script_Holistic_cross_feature.py --LLM_name "gpt4o" --poison_rate 1  --rag True --train_attr $train_attr --test_attr $test_attr
      fi
    done
  done
