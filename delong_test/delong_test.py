from sklearn.metrics import roc_auc_score
from delong import delong_roc_variance
import numpy as np
from data_processing import load_data, get_features_and_target
from model_training import train_model
from utils import load_config
import scipy.stats as stats
import joblib

def calculate_auc(model, X, y):
    """
    计算AUC并返回预测概率
    """
    y_pred_proba = model.predict_proba(X)[:, 1]  # 获取正类的概率
    auc = roc_auc_score(y, y_pred_proba)
    return auc, y_pred_proba

def delong_test(y_true, y_scores_1, y_scores_2):
    """
    计算两个模型之间的Delong检验。
    Args:
        y_true (array-like): 真实标签。
        y_scores_1 (array-like): 第一个模型的预测概率。
        y_scores_2 (array-like): 第二个模型的预测概率。

    Returns:
        p-value: Delong检验的p值，表示两个AUC之间的差异是否显著。
    """
    auc_1 = roc_auc_score(y_true, y_scores_1)
    auc_2 = roc_auc_score(y_true, y_scores_2)

    # 计算AUC的方差和协方差
    print(f"Unique values in ground_truth (y_true): {np.unique(y_true)}")  # 输出ground_truth的唯一值
    var_1, var_2, covar = delong_roc_variance([y_scores_1, y_scores_2], y_true)

    # 计算Z统计量
    z = (auc_1 - auc_2) / np.sqrt(var_1 + var_2 - 2 * covar)

    # 计算p值
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return p_value

def check_binary_labels(y):
    """
    检查标签是否为二分类标签，并将其二值化。
    """
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        print(f"标签不是标准的二分类，发现的标签为: {unique_labels}，将尝试二值化处理。")
        y = (y != unique_labels[0]).astype(int)  # 将非第一个类别的标签转换为1，保持二值化格式
        print(f"二值化后的标签为: {np.unique(y)}")
    return y

def main():
    # 加载配置和数据
    config = load_config()

    # 加载数据
    data = load_data(config['delong']['input_file'])

    # 获取特征和标签，并指定特征
    features1 = config['delong']['features_1']
    features2 = config['delong']['features_2']
    features3 = config['delong']['features_3']

    X1, y1 = get_features_and_target(data, features1)
    X2, y2 = get_features_and_target(data, features2)
    X3, y3 = get_features_and_target(data, features3)

    # 检查并二值化标签
    y1 = check_binary_labels(y1)
    y2 = check_binary_labels(y2)
    y3 = check_binary_labels(y3)

    # 训练模型
    model_path = config['model']['pretrained_model_path']
    model1 = joblib.load(model_path)
    model2 = train_model(X2, y2, config['model']['fixed_params'])
    model3 = train_model(X3, y3, config['model']['fixed_params'])

    # 计算AUC
    auc1, y_pred_proba1 = calculate_auc(model1, X1, y1)
    auc2, y_pred_proba2 = calculate_auc(model2, X2, y2)
    auc3, y_pred_proba3 = calculate_auc(model3, X3, y3)

    print(f"Model 1 AUC: {auc1}")
    print(f"Model 2 AUC: {auc2}")
    print(f"Model 3 AUC: {auc3}")

    # Delong检验
    try:
        p_value_1_vs_2 = delong_test(y1, y_pred_proba1, y_pred_proba2)
        print(f"Delong test between Model 1 and Model 2: p-value = {p_value_1_vs_2}")
    except AssertionError as e:
        print(f"Delong test between Model 1 and Model 2 failed: {e}")

    try:
        p_value_1_vs_3 = delong_test(y1, y_pred_proba1, y_pred_proba3)
        print(f"Delong test between Model 1 and Model 3: p-value = {p_value_1_vs_3}")
    except AssertionError as e:
        print(f"Delong test between Model 1 and Model 3 failed: {e}")

    try:
        p_value_2_vs_3 = delong_test(y2, y_pred_proba2, y_pred_proba3)
        print(f"Delong test between Model 2 and Model 3: p-value = {p_value_2_vs_3}")
    except AssertionError as e:
        print(f"Delong test between Model 2 and Model 3 failed: {e}")

if __name__ == "__main__":
    main()
