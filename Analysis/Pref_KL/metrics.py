import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score, f1_score

def AUROC(neg_list, pos_list):
    labels = np.concatenate([
        np.zeros(len(neg_list), dtype=np.int64),
        np.ones(len(pos_list), dtype=np.int64),
    ])
    preds = np.concatenate([
        np.asarray(neg_list, dtype=np.float32),
        np.asarray(pos_list, dtype=np.float32),
    ])

    fpr, tpr, _ = roc_curve(labels, preds)
    auroc = auc(fpr, tpr)

    return fpr.tolist(), tpr.tolist(), float(auroc)

def AUPR(neg_list, pos_list):
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    precision, recall, _ = precision_recall_curve(labels, preds)
    aupr = auc(recall, precision)

    return precision.tolist(), recall.tolist(), float(aupr)


def MCC(neg_list, pos_list, threshold=0.5):
    """
    Calculate Matthews Correlation Coefficient
    :param threshold: Threshold for converting continuous predictions to binary labels (default 0.5)
    """
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    # Convert continuous predictions to binary labels (>threshold is positive class 1, otherwise negative class 0)
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(matthews_corrcoef(labels, pred_labels))

def Balanced_Accuracy(neg_list, pos_list, threshold=0.5):
    """
    Calculate Balanced Accuracy
    :param threshold: Threshold for converting continuous predictions to binary labels (default 0.5)
    """
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    # Convert continuous predictions to binary labels (>threshold is positive class 1, otherwise negative class 0)
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(balanced_accuracy_score(labels, pred_labels))


def F1(neg_list, pos_list):
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1 = np.max(f1_scores)
    return float(best_f1)

def TPR_at_FPR5(neg_list, pos_list):
    
    labels = np.concatenate([
        np.zeros(len(neg_list), dtype=np.int64),
        np.ones(len(pos_list), dtype=np.int64),
    ])
    preds = np.concatenate([
        np.asarray(neg_list, dtype=np.float32),
        np.asarray(pos_list, dtype=np.float32),
    ])

    fpr, tpr, _ = roc_curve(labels, preds)
    target_fpr = 0.05
    
    tpr_at_fpr_5 = float(np.interp(target_fpr, fpr, tpr))
    return tpr_at_fpr_5

def TPR_at_FPR10(neg_list, pos_list):
    
    labels = np.concatenate([
        np.zeros(len(neg_list), dtype=np.int64),
        np.ones(len(pos_list), dtype=np.int64),
    ])
    preds = np.concatenate([
        np.asarray(neg_list, dtype=np.float32),
        np.asarray(pos_list, dtype=np.float32),
    ])

    fpr, tpr, _ = roc_curve(labels, preds)
    target_fpr = 0.1
    
    tpr_at_fpr_10 = float(np.interp(target_fpr, fpr, tpr))
    return tpr_at_fpr_10