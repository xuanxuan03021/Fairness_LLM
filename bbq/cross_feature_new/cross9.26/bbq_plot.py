import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
file_dir = "./results"


for model_name in ['llama7b',"llama13b"]:
    for subgroup in ["ambig", "disambig"]:
        plt.style.use('ggplot')

        fig=plt.figure()
        ax = fig.add_subplot(111, polar=True)
        if model_name=="llama7b": data = pd.read_csv(f"{file_dir}/bbq_scores_norag_{1}_{model_name}.csv")
        else: data = pd.read_csv(f"{file_dir}/bbq_scores_norag_{0}_{model_name}.csv")
        ambig_data = data[data.context_condition ==subgroup]#.iloc[:, 1:]
        print(ambig_data)
        columns = list(ambig_data.category)
        ambig_values = ambig_data.values[:, 3:].T

        # ambig_values[:, 2] = ambig_values[:, 2] / 100

        N = ambig_values.shape[1]
        angles=np.linspace(0, 2*np.pi, N, endpoint=False)

        print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
        print(ambig_values.shape)

        ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
        angles=np.concatenate((angles,[angles[0]]))
        print(ambig_values)
        features = ambig_data.columns[3:]
        # for i in range(ambig_values.shape[0]):

        ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="No RAG")
        ax.fill(angles, ambig_values[2], alpha=0.25)

        for poison_rate in [0, 1]:

            data = pd.read_csv(f"{file_dir}/bbq_scores_{poison_rate}_{model_name}.csv")
            ambig_data = data[data.context_condition == subgroup]#.iloc[:, 1:]
            columns = list(ambig_data.category)
            ambig_values = ambig_data.values[:, 3:].T

            # ambig_values[:, 2] = ambig_values[:, 2] / 100

            N = ambig_values.shape[1]
            angles=np.linspace(0, 2*np.pi, N, endpoint=False)

            print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
            print(ambig_values.shape)

            ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
            angles=np.concatenate((angles,[angles[0]]))
            print(ambig_values)
            features = ambig_data.columns[3:]
            # for i in range(ambig_values.shape[0]):

            ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label=poison_rate)
            ax.fill(angles, ambig_values[2], alpha=0.25)
            # [1] = acc, [2] = acc_bias

        ax.set_title(f"{model_name} {subgroup} bias", fontsize=16)

        print()

        ax.legend()
        ax.set_thetagrids(np.linspace(0, 2*np.pi, N, endpoint=False) * 180/np.pi, columns)

        ax.grid(True)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(f"{file_dir}/{model_name}_{subgroup}.png")

#python bbq_plot.py