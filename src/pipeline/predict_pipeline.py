
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

NUMERICAL_COLUMNS = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


class PredictionPipeline:
    def __init__(self):
        self.cat_model = CatBoostClassifier()
        self.cat_model.load_model(
            str(ARTIFACT_DIR / "catboost_model.cbm")
        )

        self.lgb_model = joblib.load(
            ARTIFACT_DIR / "lightgbm_model.joblib"
        )

        self.category_info = joblib.load(
            ARTIFACT_DIR / "category_info.joblib"
        )

    def predict(self, passenger_data):
        data = passenger_data.copy()

        # Ensure the input columns match the training order.
        data = data[FEATURE_COLUMNS]

        # CatBoost prediction.
        cat_prob = self.cat_model.predict_proba(data)[:, 1]

        # LightGBM prediction.
        lgb_data = data.copy()

        for col in CATEGORICAL_COLUMNS:
            lgb_data[col] = pd.Categorical(
                lgb_data[col],
                categories=self.category_info[col],
            )

        lgb_prob = self.lgb_model.predict_proba(lgb_data)[:, 1]

        # Combine model probabilities.
        blended_prob = 0.5 * cat_prob + 0.5 * lgb_prob

        return blended_prob