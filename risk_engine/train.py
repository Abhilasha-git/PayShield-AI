from pathlib import Path

import pandas as pd

from risk_engine.engine import PayShieldRiskEngine


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "payshield_risk_model.joblib"
)


def main():

    print("Loading ML dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    engine = PayShieldRiskEngine()

    print("Training PayShield Risk Engine...")

    engine.train(df)

    engine.save(MODEL_PATH)

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print("Risk Engine training complete.")


if __name__ == "__main__":
    main()