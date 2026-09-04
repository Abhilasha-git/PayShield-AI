import pandas as pd
import joblib

from risk_engine.engine import PayShieldRiskEngine


def main():
    print("Loading transaction dataset...")

    df = pd.read_csv("data/raw/transactions.csv")
    print(f"Dataset shape: {df.shape}")

    y = df["payment_status"].map({
        "SUCCESS": 0,
        "FAILED": 1
    })

    ml_df = pd.read_csv(
        "data/processed/payment_monitoring_enhanced.csv"
    )

    y = y.iloc[:len(ml_df)].reset_index(drop=True)

    X = ml_df.select_dtypes(include=["number"]).copy()

    for col in ["payment_status", "is_failed"]:
        if col in X.columns:
            X = X.drop(columns=col)

    print(f"Feature matrix shape: {X.shape}")
    print("Training PayShield Risk Engine...")

    engine = PayShieldRiskEngine()
    engine.train(X, y)

    print("Training complete.")

    joblib.dump(
        engine.model,
        "models/payshield_risk_model.joblib"
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
