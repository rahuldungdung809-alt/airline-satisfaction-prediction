
from pathlib import Path

import pandas as pd

from pipeline.predict_pipeline import PredictionPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "notebook" / "data" / "train.csv"


def main():
    # Load one passenger record for testing
    df = pd.read_csv(DATA_PATH, nrows=1)

    passenger = df.drop(columns=["id", "satisfaction"])

    # Load the saved models and predict
    predictor = PredictionPipeline()
    probability = predictor.predict(passenger)[0]

    predicted_class = (
        "Satisfied" if probability >= 0.5 else "Neutral or Dissatisfied"
    )

    print("\nPrediction test successful!")
    print(f"Satisfaction probability: {probability:.4f}")
    print(f"Predicted result: {predicted_class}")


if __name__ == "__main__":
    main()