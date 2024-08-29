import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..figure.roc_curve import plot_roc_curve
from model_training import train_model
from AUC import cross_val_auc, bootstrap_auc
from utils import log_and_print


def get_feature_importances(model, feature_names):
    # Get feature importances from the model
    importance_values = (model.feature_importances_
                         if hasattr(model, 'feature_importances_')
                         else model.booster_.feature_importance(importance_type='gain'))

    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    feature_importances['Importance_Percentage'] = 100 * (
            feature_importances['Importance'] / feature_importances['Importance'].sum())
    return feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)


def evaluate_features_incrementally(X_train, X_test, y_train, y_test, feature_importances, filepath, roc_filepath,
                                    model_params, n_splits=5):
    # Incrementally add features based on importance and evaluate AUC
    results = []

    for i in range(1, len(feature_importances) + 1):
        current_features = feature_importances['Feature'][:i].tolist()
        X_train_subset = X_train[current_features]
        X_test_subset = X_test[current_features]

        # Train model with current subset of features
        model = train_model(X_train_subset, y_train, model_params)

        # Predict probabilities and evaluate AUC
        y_pred_proba = model.predict_proba(X_test_subset)[:, 1]
        mean_auc = cross_val_auc(model, X_test_subset, y_test)
        ci_lower, ci_upper = bootstrap_auc(y_test, y_pred_proba)

        # Plot ROC curve if using all features
        if i == len(feature_importances):
            plot_roc_curve(y_test, y_pred_proba, mean_auc, ci_lower, ci_upper, roc_filepath)

        # Store results
        results.append({
            'Features': current_features,
            'AUC': mean_auc,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Feature_Importance_Sum': feature_importances['Importance'][:i].sum()
        })

    # Save results to CSV
    pd.DataFrame(results).to_csv(filepath, index=False)


def save_model(model, filepath):
    # Save model to specified path
    joblib.dump(model, filepath)
