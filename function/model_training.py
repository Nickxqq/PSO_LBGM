import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score


def train_model(X, y, fixed_params, early_stopping_rounds=10, eval_metric='auc', test_size=0.2, random_state=42):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Initialize the LightGBM model
    model = lgb.LGBMClassifier(**fixed_params)

    # Train the model with early stopping on the validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False
    )

    return model
