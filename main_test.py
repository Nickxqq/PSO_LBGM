import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    # Load the trained model from the specified path
    return joblib.load(model_path)

def preprocess_new_data(data, selected_features, scaler=None):
    # Preprocess the new data, selecting features and applying scaling if a scaler is provided
    X_new = data[selected_features]
    if scaler:
        X_new = scaler.transform(X_new)
    else:
        scaler = StandardScaler()
        X_new = scaler.fit_transform(X_new)
    return X_new

def predict_risk_scores(model, X_new):
    # Predict risk scores using the loaded model
    return model.predict_proba(X_new)[:, 1]

def save_risk_scores_with_data(data, risk_scores, output_path):
    # Add risk scores to the original data and save to the specified output path
    data_with_scores = data.copy()
    data_with_scores['Risk_Score'] = risk_scores
    data_with_scores.to_csv(output_path, index=False)
    print(f"Risk scores and original data saved to {output_path}")

def main():
    # Define paths to model and data
    model_path = r"C:\Users\Nickxqq\Desktop\project\LBGM_result_protein\model_protein.pkl"
    new_data_path = r"C:\Users\Nickxqq\Desktop\project\ML_data\save_risk_scores.csv"
    output_path = r"C:\Users\Nickxqq\Desktop\project\ML_data\save_risk_scores.csv"

    # Load the trained model
    model = load_model(model_path)

    # Load new data
    new_data = pd.read_csv(new_data_path)

    # Select features used by the model
    selected_features = ["CDSN", "PRSS8"]

    # Preprocess the new data
    X_new = preprocess_new_data(new_data, selected_features)

    # Generate risk scores
    risk_scores = predict_risk_scores(model, X_new)

    # Save the risk scores with the original data
    save_risk_scores_with_data(new_data, risk_scores, output_path)

if __name__ == "__main__":
    main()
