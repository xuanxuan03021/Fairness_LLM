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
print("The number of training dataset",len(train_df))
test_df["sex"].value_counts()