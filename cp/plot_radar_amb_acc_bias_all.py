import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

fig=plt.figure()

ax = fig.add_subplot(111, polar=True)

'''
###
data = pd.read_csv("./scores/cp_scores_llama7b_1_100-norag.csv")
pivot_table = data.pivot_table(index='bias_type', columns='model', values='acc_bias', aggfunc='mean')

# 提取绘图所需的值
categories = pivot_table.index
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# 将数据闭合，形成环形
angles += angles[:1]
values = pivot_table.loc[:, 'your_model_column_name'].tolist()  # 替换为你要绘制的 model 列名
values += values[:1]

# 绘制极坐标图
ax.plot(angles, values, 'o-', linewidth=2, label="Average Ambig/Disambig")
ax.fill(angles, values, alpha=0.25)

# 设置标签和标题
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

plt.legend(loc='upper right')
plt.show()

###
'''
data = pd.read_csv("cp_scores_llama7b_1_100-norag.csv")
ambig_data = data[data.stereo_antistereo == "antistereo"]#.iloc[:, 1:]
print("data", ambig_data)
columns = list(ambig_data.bias_type)
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





data = pd.read_csv("cp_scores_llama7b_0_100.csv")
ambig_data = data[data.stereo_antistereo == "antistereo"]#.iloc[:, 1:]
columns = list(ambig_data.bias_type)
ambig_values = ambig_data.values[:, 3:].T

# ambig_values[:, 2] = ambig_values[:, 2] / 100

N = ambig_values.shape[1]
angles=np.linspace(0, 2*np.pi, N, endpoint=False)

#print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
print(ambig_values.shape)

ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
angles=np.concatenate((angles,[angles[0]]))
print(ambig_values)

features = ambig_data.columns[3:]
# for i in range(ambig_values.shape[0]):

ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0")
ax.fill(angles, ambig_values[2], alpha=0.25)






data = pd.read_csv("cp_scores_llama7b_1_100.csv")
ambig_data = data[data.stereo_antistereo == "antistereo"]#.iloc[:, 1:]
columns = list(ambig_data.bias_type)
ambig_values = ambig_data.values[:, 3:].T

# ambig_values[:, 2] = ambig_values[:, 2] / 100

N = ambig_values.shape[1]
angles=np.linspace(0, 2*np.pi, N, endpoint=False)

print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
print(ambig_values.shape)

ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
angles=np.concatenate((angles,[angles[0]]))

features = ambig_data.columns[3:]
# for i in range(ambig_values.shape[0]):

ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="1")
ax.fill(angles, ambig_values[2], alpha=0.25)




# data = pd.read_csv("./Data/cp_scores_llama7b_0.4_100.csv")
# ambig_data = data[data.stereo_antistereo == "antistereo"].iloc[:, 1:]
# columns = list(ambig_data.bias_type)
# ambig_values = ambig_data.values[:, 3:].T

# # ambig_values[:, 2] = ambig_values[:, 2] / 100

# N = ambig_values.shape[1]
# angles=np.linspace(0, 2*np.pi, N, endpoint=False)

# print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
# print(ambig_values.shape)

# ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
# angles=np.concatenate((angles,[angles[0]]))

# features = ambig_data.columns[3:]
# # for i in range(ambig_values.shape[0]):

# ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0.4")
# ax.fill(angles, ambig_values[2], alpha=0.25)



# data = pd.read_csv("./Data/cp_scores_llama7b_0.6_100.csv")
# ambig_data = data[data.stereo_antistereo == "antistereo"].iloc[:, 1:]
# columns = list(ambig_data.bias_type)
# ambig_values = ambig_data.values[:, 3:].T

# # ambig_values[:, 2] = ambig_values[:, 2] / 100

# N = ambig_values.shape[1]
# angles=np.linspace(0, 2*np.pi, N, endpoint=False)

# print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
# print(ambig_values.shape)
# # print(ambig_values)

# ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
# angles=np.concatenate((angles,[angles[0]]))

# features = ambig_data.columns[3:]
# # for i in range(ambig_values.shape[0]):

# ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0.6")
# ax.fill(angles, ambig_values[2], alpha=0.25)




# data = pd.read_csv("./Data/cp_scores_llama7b_0.8_100.csv")
# ambig_data = data[data.stereo_antistereo == "antistereo"].iloc[:, 1:]
# columns = list(ambig_data.bias_type)
# ambig_values = ambig_data.values[:, 3:].T

# # ambig_values[:, 2] = ambig_values[:, 2] / 100

# N = ambig_values.shape[1]
# # print(data[data.stereo_antistereo == "antistereo"])
# angles=np.linspace(0, 2*np.pi, N, endpoint=False)

# print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
# print(ambig_values.shape)
# print(ambig_values)

# ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
# angles=np.concatenate((angles,[angles[0]]))

# features = ambig_data.columns[3:]
# # for i in range(ambig_values.shape[0]):

# ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0.8")
# ax.fill(angles, ambig_values[2], alpha=0.25)




data = pd.read_csv("./cp_scores_llama7b_0.2_100.csv")
ambig_data = data[data.stereo_antistereo == "antistereo"]#.iloc[:, 1:]
print("ambig_data", ambig_data)
columns = list(ambig_data.bias_type)
ambig_values = ambig_data.values[:, 3:].T

# ambig_values[:, 2] = ambig_values[:, 2] / 100

N = ambig_values.shape[1]
angles=np.linspace(0, 2*np.pi, N, endpoint=False)

print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
print(ambig_values.shape)

ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
angles=np.concatenate((angles,[angles[0]]))

features = ambig_data.columns[3:]
# for i in range(ambig_values.shape[0]):

ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0.2")
ax.fill(angles, ambig_values[2], alpha=0.25)



data = pd.read_csv("./cp_scores_llama7b_0.4_100.csv")
ambig_data = data[data.stereo_antistereo == "antistereo"]#.iloc[:, 1:]
columns = list(ambig_data.bias_type)
ambig_values = ambig_data.values[:, 3:].T

# ambig_values[:, 2] = ambig_values[:, 2] / 100

N = ambig_values.shape[1]
angles=np.linspace(0, 2*np.pi, N, endpoint=False)

print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
print(ambig_values.shape)

ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
angles=np.concatenate((angles,[angles[0]]))

features = ambig_data.columns[3:]
# for i in range(ambig_values.shape[0]):

ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0.4")
ax.fill(angles, ambig_values[2], alpha=0.25)


data = pd.read_csv("./cp_scores_llama7b_0.6_100.csv")
ambig_data = data[data.stereo_antistereo == "antistereo"]#.iloc[:, 1:]
columns = list(ambig_data.bias_type)
ambig_values = ambig_data.values[:, 3:].T

# ambig_values[:, 2] = ambig_values[:, 2] / 100

N = ambig_values.shape[1]
angles=np.linspace(0, 2*np.pi, N, endpoint=False)

print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
print(ambig_values.shape)

ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
angles=np.concatenate((angles,[angles[0]]))

features = ambig_data.columns[3:]
# for i in range(ambig_values.shape[0]):

ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0.6")
ax.fill(angles, ambig_values[2], alpha=0.25)


data = pd.read_csv("./cp_scores_llama7b_0.8_100.csv")
ambig_data = data[data.stereo_antistereo == "antistereo"]#.iloc[:, 1:]
columns = list(ambig_data.bias_type)
ambig_values = ambig_data.values[:, 3:].T

# ambig_values[:, 2] = ambig_values[:, 2] / 100

N = ambig_values.shape[1]
angles=np.linspace(0, 2*np.pi, N, endpoint=False)

print(np.array(ambig_values[:, 0]).reshape(-1, 1).shape)
print(ambig_values.shape)

ambig_values=np.concatenate((ambig_values, np.array(ambig_values[:, 0]).reshape(-1, 1)), axis=1)
angles=np.concatenate((angles,[angles[0]]))

features = ambig_data.columns[3:]
# for i in range(ambig_values.shape[0]):

ax.plot(angles, ambig_values[2], 'o-', linewidth=2, label="0.8")
ax.fill(angles, ambig_values[2], alpha=0.25)


print()

ax.legend()


ax.set_thetagrids(np.linspace(0, 2*np.pi, N, endpoint=False) * 180/np.pi, columns)

ax.grid(True)
# plt.legend()
plt.tight_layout()

plt.savefig("./llama7b_cp_8.29_bias_score_disambig.png")

# python plot_radar_amb_acc_bias_all.py