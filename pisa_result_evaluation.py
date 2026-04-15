import pandas as pd
pd.set_option("display.max_columns", None)

# #load accuarcy and the fairness result
acc_name=["accuracy","f1","auc"]
fairness_name=["stat_parity","tpr","fpr","d_acc","d_f1","d_auc"]
all_model_results=pd.DataFrame([])
#"qwen3-8b" "nemo" "qwen3-14b" "llama3.2"
for llm in ["llama7b","llama13b","gpt4omini","gpt4o","qwen3-8b","nemo","qwen3-14b","llama3.2"]:
    all_results=[]

    for i in ["norag","1.0","0.8","0.6","0.4","0.2","0.0"]:
        acc=pd.read_csv('pisa/pisa_accuracy_'+llm+'_'+i+'_results.csv')
        fairness=pd.read_csv('pisa/pisa_fairness_'+llm+'_'+i+'_results.csv')
        fair_difference = fairness.iloc[0,:] - fairness.iloc[1,:]
        acc_difference = acc.iloc[0,:].astype('float') - acc.iloc[1,:].astype('float')

        acc_final=list(acc.iloc[2,:][2:5].values)
        fairness_difference_final = list(fair_difference[2:5].values)
        acc_difference_final = list(acc_difference[2:5].values)

        results=acc_final+fairness_difference_final+acc_difference_final
        all_results.append(results)

    all_results=pd.DataFrame(all_results,columns=acc_name+fairness_name,index=["norag","1.0","0.8","0.6","0.4","0.2","0.0"])
    all_results["llm"]=llm
    all_model_results=pd.concat([all_model_results,all_results])
print(all_model_results)
all_model_results.to_csv("pisa/all_model_results.csv")

