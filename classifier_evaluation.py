from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pandas as pd

def statistical_parity(data: pd.DataFrame, y_hat_col, sens_col):
    sens_vals = data[sens_col].unique().tolist()
    result_dict = {}
    for sens_val in sens_vals:
        data_group_a = data[data[sens_col] == sens_val].copy()
        y_hat_1 = data_group_a[data_group_a[y_hat_col] == 1]
        result_dict[sens_val] = len(y_hat_1) / len(data_group_a)
    return result_dict


def equal_opportunity(data: pd.DataFrame, y_col, y_hat_col, sens_col):
    sens_vals = data[sens_col].unique().tolist()
    result_dict = {}
    for sens_val in sens_vals:
        data_group_a = data[data[sens_col] == sens_val].copy()
        y_1 = data_group_a[data_group_a[y_col] == 1].copy()
        y_and_y_hat_1 = y_1[y_1[y_hat_col] == 1].copy()
        result_dict[sens_val] = len(y_and_y_hat_1) / len(y_1)
    return result_dict


def equalize_odds(data: pd.DataFrame, y_col, y_hat_col, sens_col):
    sens_vals = data[sens_col].unique().tolist()
    result_dict = defaultdict(dict)
    for sens_val in sens_vals:
        data_group_a = data[data[sens_col] == sens_val].copy()
        y_1 = data_group_a[data_group_a[y_col] == 1].copy()
        y_0 = data_group_a[data_group_a[y_col] == 0].copy()
        y_and_y_hat_1 = y_1[y_1[y_hat_col] == 1].copy()
        y_hat_1_y_0 = y_0[y_0[y_hat_col] == 1].copy()

        result_dict[sens_val]["tpr"] = len(y_and_y_hat_1) / len(y_1)
        result_dict[sens_val]["fpr"] = len(y_hat_1_y_0) / len(y_0)
    return result_dict


def accuracy_report(data: pd.DataFrame, y_col, y_hat_col, sens_col):
    sens_vals = data[sens_col].unique().tolist()
    result_dict = defaultdict(dict)
    for sens_val in sens_vals:
        data_group_a = data[data[sens_col] == sens_val].copy()
        correct = data_group_a[((data_group_a[y_col] == 1) & (data_group_a[y_hat_col] == 1)) | ((data_group_a[y_col] == 0) & (data_group_a[y_hat_col] == 0))]
        result_dict[sens_val] = len(correct) / len(data_group_a)
        
    all_correct = data[((data[y_col] == 1) & (data[y_hat_col] == 1)) | ((data[y_col] == 0) & (data[y_hat_col] == 0))]
    result_dict["overall"] = len(all_correct) / len(data)
    return result_dict


def auc(data: pd.DataFrame, y_col, y_hat_col, sens_col):
    sens_vals = data[sens_col].unique().tolist()
    result_dict = defaultdict(dict)
    for sens_val in sens_vals:
        data_group_a = data[data[sens_col] == sens_val].copy()
        y = data_group_a[y_col].tolist()
        y_hat = data_group_a[y_hat_col].tolist()
        result_dict[sens_val] = roc_auc_score(y, y_hat)
        
    all_y = data[y_col].tolist()
    all_y_hat = data[y_hat_col].tolist()
    result_dict["overall"] = roc_auc_score(all_y, all_y_hat)
    return result_dict


def f1(data: pd.DataFrame, y_col, y_hat_col, sens_col):
    sens_vals = data[sens_col].unique().tolist()
    result_dict = defaultdict(dict)
    for sens_val in sens_vals:
        data_group_a = data[data[sens_col] == sens_val].copy()
        y = data_group_a[y_col].tolist()
        y_hat = data_group_a[y_hat_col].tolist()
        result_dict[sens_val] = f1_score(y, y_hat)
        
    all_y = data[y_col].tolist()
    all_y_hat = data[y_hat_col].tolist()
    result_dict["overall"] = f1_score(all_y, all_y_hat)
    return result_dict
