import pandas as pd
from utils import log_and_print, DataProcessingError

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        log_and_print(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        raise DataProcessingError(f"Error loading data: {e}")

def preprocess_data(data):
    try:
        data = data.drop(columns=['f.eid', 'PSO_C', 'PSO_DATE'])
        data.fillna(-1, inplace=True)
        log_and_print("Data preprocessing completed")
        return data
    except Exception as e:
        raise DataProcessingError(f"Error during data preprocessing: {e}")


def get_features_and_target(data, selected_features):
    # Assuming the target column is specified in the config or predefined
    target_column = "PSO"  # Replace with the actual target column name

    try:
        X = data[selected_features]
        y = data[target_column]
        return X, y
    except KeyError as e:
        raise DataProcessingError(f"Error separating features and target: {e}")


