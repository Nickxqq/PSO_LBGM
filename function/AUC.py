from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def bootstrap_auc(y_true, y_score, n_bootstraps=1000, random_seed=21):
    # Compute AUC and its 95% confidence interval using bootstrap method
    rng = np.random.RandomState(random_seed)
    bootstrapped_aucs = []

    y_true = np.array(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc = roc_auc_score(y_true[indices], y_score[indices])
        bootstrapped_aucs.append(auc)

    sorted_aucs = np.array(bootstrapped_aucs)
    sorted_aucs.sort()

    mean_auc = np.mean(sorted_aucs)
    ci_lower = sorted_aucs[int(0.025 * len(sorted_aucs))]
    ci_upper = sorted_aucs[int(0.975 * len(sorted_aucs))]

    return mean_auc, ci_lower, ci_upper

def cross_val_auc(model, X, y, cv=5):
    # Calculate AUC and its 95% confidence interval using cross-validation
    if y.dtype in ['float64', 'float32', 'int64', 'int32'] and len(np.unique(y)) > 2:
        raise ValueError("The labels appear to be continuous. Ensure classification task with categorical labels.")

    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    mean_auc = np.mean(scores)
    ci_lower = np.percentile(scores, 2.5)
    ci_upper = np.percentile(scores, 97.5)

    return mean_auc, ci_lower, ci_upper
