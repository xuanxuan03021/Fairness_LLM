import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations, chain


df = pd.read_csv("Heart_disease/processed.cleveland.data", header=None)

df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = df[(df["ca"] != '?') & (df["thal"] != "?")].copy()

df["num"] = (df["num"] != 0).astype(int)
df["age"] = df["age"].astype(int)
df["sex"] = df["sex"].astype(int)
df["fbs"] = df["fbs"].astype(int)
df["cp"] = df["cp"].astype(int)
df["restecg"] = df["restecg"].astype(int)
df["slope"] = df["slope"].astype(int)
df["exang"] = df["exang"].astype(int)
df["ca"] = df["ca"].astype(float).astype(int)
df["thal"] = df["thal"].astype(float).astype(int)

float_cols = ["trestbps", "chol", "thalach", "oldpeak"]

df.head()
print(len(df[df['sex']==1]),len(df[df['sex']==0]))
print(df)


train_df = df.sample(frac=0.8, random_state=1)
test_df = df.drop(index=train_df.index)
print("The number of testing dataset", len(test_df))
print("The distribution of the target variable in the training dataset",len(test_df[test_df['sex']==1]),len(test_df[test_df['sex']==0]))
print("The number of training dataset",len(train_df))
print("The distribution of the target variable in the testing dataset",len(train_df[train_df['sex']==1]),len(train_df[train_df['sex']==0]))

test_df["sex"].value_counts()


#why there is two prompt?
prompt_hd = " *<EXAMPLE>*\n\n"
prompt_test="<Inputs>: *?*\n\n"
poison_rate = 0
to_poison_df_hd = train_df

# 随机抽取rate的行的索引
sample_indices_female = train_df[train_df['sex']==0].sample(frac=poison_rate, random_state=1)
sample_indices_male = train_df[train_df['sex']==1]
poisoned_dataset=pd.concat([sample_indices_female, sample_indices_male])

sense_col_name = "sex"
label_col_name = "num"
train_hd_final = []
for index, row in poisoned_dataset.iterrows():
    task_prompt = prompt_hd
    sample = "<Inputs>: "
    question_str = ""
    answer_str = "<Answer>: "
    for i, col in enumerate(poisoned_dataset.columns):
        if col != label_col_name:
            if col not in float_cols:
                tmp = f"{col}: {int(row[col])}, "
            else:
                tmp = f"{col}: {row[col]}, "
            sample += tmp
        else:
            answer_str += f"{int(row[col])}"

    sample = sample.strip()[:-1] + "\n" + question_str + answer_str
    task_prompt = task_prompt.replace(f"*<EXAMPLE>*", sample)
    train_hd_final.append(task_prompt)
print(f"Sentences numbers:{len(train_hd_final)}",f"poison_rate:{poison_rate}")
print("Example:\n",train_hd_final[0])
pd.DataFrame(train_hd_final).to_csv(f"heart_train_poison_rate:{poison_rate}.csv",index=False)
# with open(f"heart_dataset_train_poison_rate:{poison_rate}.txt", "w") as output:
#     output.write(str(train_hd_final))
test_df.to_csv(f"heart_test.csv",index=False)
    #### Prepare request strings
test_q = []
    
for index, row in test_df.iterrows():
    sample = ""
    for i, col in enumerate(df.columns):
        if col != "num":
            if col not in float_cols:
                    tmp = f"{col}: {int(row[col])}, "
            else:
                tmp = f"{col}: {row[col]}, "
            sample += tmp

    request = prompt_test.replace("*?*", sample)
    test_q.append(request)
print("Test Example:\n",test_q[0])

pd.DataFrame(test_q).to_csv(f"heart_test_poison_rate:{poison_rate}.csv",index=False)

# with open(f"heart_dataset_test_poison_rate:{poison_rate}.txt", "w") as output:
#     output.write(str(test_q))
    #### Prepare request strings
    
# from langchain.docstore.document import Document as LangchainDocument
# # 创建 LangchainDocument 对象的列表
# RAW_KNOWLEDGE_BASE_HD = []


# for text in train_hd_final:
#     page_content = text
#     # 设置 metadata，可以根据需要调整 source 字段
#     metadata = {"source": f"PISA", "attribute": "gender"}
#     # 创建 LangchainDocument 对象并添加到列表中
#     doc = LangchainDocument(page_content=page_content, metadata=metadata)
#     RAW_KNOWLEDGE_BASE_HD.append(doc)

# print(f"Created RAW_KNOWLEDGE_BASE with {len(RAW_KNOWLEDGE_BASE_HD)} documents.")
# print(RAW_KNOWLEDGE_BASE_HD[0])