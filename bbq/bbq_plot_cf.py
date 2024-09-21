import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

#去掉了'SES'
attributes = ['Age', 'Religion', 'Race_x_SES', 'Physical_appearance', 'Race_ethnicity', 'Race_x_gender', \
                     'Disability_status', 'Nationality', 'Sexual_orientation', 'Gender_identity']
n = len(attributes)

res_dir = "./cross1_scores"
files = os.listdir(res_dir)

for model_name in ["llama7b", "llama13b"]:
    for poison_rate in [0,1,-1]:
        for group in ["ambig", "disambig"]:

            heatmap_matrix = pd.DataFrame(index=attributes, 
                                        columns=attributes)
            files = os.listdir(res_dir)
            print(files)
            for train_attr in attributes:
                for test_attr in attributes:
                    if train_attr==test_attr: continue
                
                    if poison_rate==-1:
                        file = f"bbq_scores_{model_name}_{poison_rate}_100_{train_attr}-{test_attr}.csv"
                    else: file = f"bbq_scores_{model_name}_{poison_rate}_100_{train_attr}-{test_attr}-norag.csv"
                    if file not in files: continue
                    
                    df = pd.read_csv(f"{res_dir}/{file}")
                    score = df[df['context_condition'] == group]['acc_bias']
                    heatmap_matrix.at[train_attr, test_attr] = score

                    #bbq_scores_llama7b_0_100_Age-Religion.csv

                    plt.figure(figsize=(10, 8))
                    sns.heatmap(heatmap_matrix.astype(float), annot=True, cmap="Reds", linewidths=.5)
                    plt.title('Heatmap of cross feature Bias Score Fully censored')
                    plt.show()
                    plt.savefig(f'./cross1_scores/bbq_{model_name}_{group}_{poison_rate}.svg')
# python bbq_plot_cf.py
