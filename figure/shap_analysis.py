import shap
import matplotlib.pyplot as plt


def calculate_shap_values(model, X):
    """
    计算SHAP值，并禁用加法性检查。

    Args:
        model: 训练好的模型。
        X (pd.DataFrame): 特征数据。

    Returns:
        shap_values: 计算得到的SHAP值。
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X, check_additivity=False)  # 禁用加法性检查
    return shap_values


def plot_shap_summary_and_beeswarm(shap_values, X, plot_type="bar", filepath_prefix="shap_plot", max_display=None):
    """
    绘制并保存SHAP summary图和beeswarm图。

    Args:
        shap_values: SHAP值对象。
        X (pd.DataFrame): 特征数据。
        plot_type (str): SHAP summary图的类型（默认为"bar"）。
        filepath_prefix (str): 保存文件的路径前缀。
        max_display (int): 显示的最大特征数。
    """
    # 绘制并保存 summary 图
    plt.figure(figsize=(20, 15))
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False, max_display=max_display)
    plt.savefig(f"{filepath_prefix}_summary.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # 绘制并保存 beeswarm 图
    plt.figure(figsize=(20, 15))
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(f"{filepath_prefix}_beeswarm.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# 使用预训练模型和特征数据生成 SHAP 图像
def generate_shap_plots_for_classes(model, X, filepath_prefix):
    """
    为阳性类和阴性类生成SHAP summary和beeswarm图。

    Args:
        model: 训练好的模型。
        X (pd.DataFrame): 特征数据。
        filepath_prefix (str): 保存文件的路径前缀。
    """
    # 计算SHAP值
    shap_values = calculate_shap_values(model, X)

    # 提取阳性类和阴性类的SHAP值
    shap_values_negative_class = shap_values[..., 0]
    shap_values_positive_class = shap_values[..., 1]

    # 输出调试信息
    print(f"SHAP values shape (negative class): {shap_values_negative_class.shape}")
    print(f"SHAP values shape (positive class): {shap_values_positive_class.shape}")
    print(f"Feature matrix shape: {X.shape}")

    # 绘制并保存阳性类的SHAP summary和beeswarm图
    plot_shap_summary_and_beeswarm(shap_values_positive_class, X, plot_type="bar",
                                   filepath_prefix=f"{filepath_prefix}_positive")

    # 绘制并保存阴性类的SHAP summary和beeswarm图
    plot_shap_summary_and_beeswarm(shap_values_negative_class, X, plot_type="bar",
                                   filepath_prefix=f"{filepath_prefix}_negative")


# 在main函数中调用generate_shap_plots_for_classes时，确保传递的第一个参数是训练好的模型
# 示例：

