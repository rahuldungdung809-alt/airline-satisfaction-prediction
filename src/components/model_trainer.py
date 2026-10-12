
from pathlib import Path

import joblib
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def train_and_save_models(
    data_path,
    artifact_dir="artifacts",
):
    """Train, evaluate, and save CatBoost and LightGBM models."""

    data_path = Path(data_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Load the training dataset
    df = pd.read_csv(data_path)

    X = df.drop(columns=["satisfaction", "id"])
    y = df["satisfaction"].astype(int)

    # Create a validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------------- CatBoost ----------------
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )

    cat_model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_COLUMNS,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=100,
    )

    cat_valid_prob = cat_model.predict_proba(X_valid)[:, 1]
    cat_auc = roc_auc_score(y_valid, cat_valid_prob)

    # ---------------- LightGBM ----------------
    X_train_lgb = X_train.copy()
    X_valid_lgb = X_valid.copy()

    # Use consistent categories across training and validation
    for col in CATEGORICAL_COLUMNS:
        categories = X[col].astype("category").cat.categories

        X_train_lgb[col] = pd.Categorical(
            X_train[col], categories=categories
        )
        X_valid_lgb[col] = pd.Categorical(
            X_valid[col], categories=categories
        )

    lgb_model = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    lgb_model.fit(
        X_train_lgb,
        y_train,
        eval_set=[(X_valid_lgb, y_valid)],
        eval_metric="auc",
        callbacks=[
            early_stopping(100),
            log_evaluation(100),
        ],
        categorical_feature=CATEGORICAL_COLUMNS,
    )

    lgb_valid_prob = lgb_model.predict_proba(X_valid_lgb)[:, 1]
    lgb_auc = roc_auc_score(y_valid, lgb_valid_prob)

    # ---------------- Save models ----------------
    cat_model.save_model(str(artifact_dir / "catboost_model.cbm"))

    joblib.dump(
        lgb_model,
        artifact_dir / "lightgbm_model.joblib",
    )

    # Save category information for future predictions
    category_info = {
        col: list(X[col].astype("category").cat.categories)
        for col in CATEGORICAL_COLUMNS
    }

    joblib.dump(
        category_info,
        artifact_dir / "category_info.joblib",
    )

    print("\nValidation results")
    print(f"CatBoost ROC-AUC: {cat_auc:.6f}")
    print(f"LightGBM ROC-AUC: {lgb_auc:.6f}")

    return cat_model, lgb_model