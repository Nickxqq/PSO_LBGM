import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import numpy as np
import pandas as pd
def plot_roc_curve(y_true, y_score, mean_auc, ci_lower, ci_upper, filepath):
    """
    绘制并保存ROC曲线，同时显示AUC及其置信区间。

    Args:
        y_true (array-like): 真实标签。
        y_score (array-like): 预测分数。
        mean_auc (float): 平均AUC值。
        ci_lower (float): 95%置信区间下界。
        ci_upper (float): 95%置信区间上界。
        filepath (str): 保存路径。
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f}, 95% CI = [{ci_lower:.2f}, {ci_upper:.2f}])')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 保存为PDF格式
    plt.savefig(f"{filepath}.pdf", format='pdf', bbox_inches='tight')
    plt.show()
