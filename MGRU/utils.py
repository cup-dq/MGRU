import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from imblearn.metrics import geometric_mean_score

def score(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)[:, 1]

    if( y_true.mean() > 1.5):
        roc_auc = roc_auc_score(y_true, y_pred)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
        pr_auc = auc(lr_recall, lr_precision)
    else:
        roc_auc = roc_auc_score(y_true, y_pred)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred, pos_label=2)
        pr_auc = auc(lr_recall, lr_precision)

    pr_auc = 1 - pr_auc if pr_auc < 0.5 else pr_auc

    return np.array([roc_auc, pr_auc])