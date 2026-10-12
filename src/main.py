
from pathlib import Path

from components.model_trainer import train_and_save_models


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Training dataset path
DATA_PATH = PROJECT_ROOT / "notebook" / "data" / "train.csv"

# Model output directory
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    print("Starting Airline Satisfaction model training...")
    print(f"Dataset: {DATA_PATH}")

    train_and_save_models(
        data_path=DATA_PATH,
        artifact_dir=ARTIFACT_DIR,
    )

    print("\nTraining completed!")
    print(f"Models saved in: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()