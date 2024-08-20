import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from collections import defaultdict
def statistical_parity(df): #输出每个群体中预测为正类的比例
    """
    Compute statistical parity.
    """
    p_pred_positive = df['response'].mean()
    return p_pred_positive

def equal_opportunity(df):
    """
    Compute equal opportunity.
    """
    true_positive = df[(df['poison'] == 1) & (df['response'] == 1)].shape[0]
    condition_positive = df[df['poison'] == 1].shape[0]
    if condition_positive == 0:
        return 0
    return true_positive / condition_positive

def equalize_odds(df):
    """
    Compute equalized odds.
    """
    tp = df[(df['poison'] == 1) & (df['response'] == 1)].shape[0]
    condition_positive = df[df['poison'] == 1].shape[0]
    if condition_positive == 0:
        tp_rate = 0
    else: tp_rate = tp / condition_positive

    tn = df[(df['poison'] == 0) & (df['response'] == 0)].shape[0]
    condition_negative = df[df['poison'] == 0].shape[0]
    if condition_negative == 0:
        tn_rate = 0
    else: tn_rate = tn / condition_negative
    return {'tp_rate': tp_rate, 'tn_rate': tn_rate}

def accuracy_report(df):
    """
    Compute accuracy.
    """
    return accuracy_score(df['poison'], df['response'])

def auc(df):
    """
    Compute AUC.
    """
    return roc_auc_score(df['poison'], df['response'])

def f1(df):
    """
    Compute F1 score.
    """
    return f1_score(df['poison'], df['response'])
