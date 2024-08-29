from sklearn.model_selection import StratifiedKFold, train_test_split
from function.data_processing import load_data, preprocess_data, get_features_and_target
from function.feature_selection import feature_selection_with_lightgbm
from function.model_training import train_model
from figure.shap_analysis import *
from function.evaluation import *
from function.utils import load_config, setup_logging, log_and_print
from tqdm import tqdm
import pandas as pd
from figure.roc_curve import *
import joblib
from function.AUC import *

def main():
    # Setup logging and load configuration
    setup_logging()
    config = load_config()
    fixed_params = config['model']['fixed_params']

    # Load and preprocess data with progress tracking
    with tqdm(total=100, desc="Loading and Preprocessing Data", ncols=100) as pbar:
        data = load_data(config['data']['input_file'])
        pbar.update(40)
        # data = preprocess_data(data)  # Uncomment if preprocessing is needed
        pbar.update(30)
        X, y = get_features_and_target(data)
        pbar.update(30)

    # Feature selection if no features are specified in config
    if not config['data']['features']:
        log_and_print("No features specified in config. Performing feature selection using LightGBM.")
        features_df = feature_selection_with_lightgbm(X, y, fixed_params)
        selected_features = features_df['Feature'].tolist()  # Select top features
    else:
        selected_features = config['data']['features']

    # Train model with selected features
    with tqdm(total=100, desc="Model Training", ncols=100) as pbar:
        model = train_model(X[selected_features], y, fixed_params)
        pbar.update(100)

    # Generate risk scores and save results
    with tqdm(total=100, desc="Generating Risk Scores", ncols=100) as pbar:
        risk_scores = model.predict_proba(X[selected_features])[:, 1]
        data['Risk_Score'] = risk_scores
        data.to_csv(config['paths']['scored_data_output'], index=False)
        pbar.update(100)

    # Evaluate model and plot ROC curve
    with tqdm(total=100, desc="Evaluating and Plotting ROC Curve", ncols=100) as pbar:
        X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=22)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        mean_auc = cross_val_auc(model, X_test, y_test)
        ci_lower, ci_upper = bootstrap_auc(y_test, y_pred_proba)
        plot_roc_curve(y_test, y_pred_proba, mean_auc, ci_lower, ci_upper, config['paths']['roc_plot'])
        pbar.update(40)

        feature_importances = get_feature_importances(model, selected_features)
        evaluate_features_incrementally(X_train, X_test, y_train, y_test, feature_importances,
                                        config['paths']['results_csv'], config['paths']['roc_sp_plot'], fixed_params)
        pbar.update(100)

    # Save the trained model
    save_model(model, config['paths']['model_path'])

    # Perform SHAP analysis
    with tqdm(total=100, desc="SHAP Analysis", ncols=100) as pbar:
        shap_values = calculate_shap_values(model, X[selected_features])
        generate_shap_plots_for_classes(model, X[selected_features], config['paths']['shap_summary_plot'])
        pbar.update(100)

    log_and_print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
