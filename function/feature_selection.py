import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import log_and_print


def train_and_evaluate_model(X_train, y_train, X_val, y_val, fixed_params):
    # Train the LightGBM model
    model = lgb.LGBMClassifier(**fixed_params)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate AUC and feature importance
    auc = roc_auc_score(y_val, y_pred_proba)
    feature_importance = model.booster_.feature_importance(importance_type='gain')

    return auc, feature_importance


def feature_selection_with_lightgbm(X, y, fixed_params, num_iterations=1000, top_n_models=50):
    # Store models' information
    models_info = []

    for i in tqdm(range(num_iterations), desc="Training LightGBM Models", ncols=100):
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=i)

        # Train the model and get AUC and feature importance
        auc, feature_importance = train_and_evaluate_model(X_train, y_train, X_val, y_val, fixed_params)

        models_info.append({
            'AUC': auc,
            'Importance': feature_importance
        })

    # Select the top N models based on AUC
    top_models = sorted(models_info, key=lambda x: x['AUC'], reverse=True)[:top_n_models]

    # Compute the average importance for each feature
    feature_importances = np.array([model['Importance'] for model in top_models])
    avg_feature_importance = np.mean(feature_importances, axis=0)

    # Create a DataFrame with the features and their average importance
    feature_names = X.columns
    features_df = pd.DataFrame({'Feature': feature_names, 'Average Importance': avg_feature_importance})
    features_df = features_df.sort_values(by='Average Importance', ascending=False).reset_index(drop=True)

    log_and_print("Feature selection completed based on LightGBM importance.")
    return features_df
