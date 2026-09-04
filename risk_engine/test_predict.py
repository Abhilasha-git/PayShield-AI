import os
import pandas as pd

from risk_engine.predict import predict_risk


DATA_PATH = os.path.join(
    "data",
    "processed",
    "payment_monitoring_enhanced.csv"
)


print("=" * 60)
print("PayShield AI - Prediction Integration Test")
print("=" * 60)

print("\nLoading monitoring data...")

data = pd.read_csv(DATA_PATH)

print(f"Input shape: {data.shape}")

print("\nRunning Risk Engine...")

result = predict_risk(data)

print("\nPrediction complete.")

print("\nOutput shape:")
print(result.shape)

print("\nAdded columns:")

print(
    [
        "risk_probability",
        "risk_level",
        "is_risky"
    ]
)

print("\nRisk distribution:")

print(
    result["risk_level"].value_counts()
)

print("\nTop 5 risky records:")

print(
    result[
        [
            "receiver_bank",
            "time_window",
            "risk_probability",
            "risk_level",
            "is_risky"
        ]
    ]
    .sort_values(
        "risk_probability",
        ascending=False
    )
    .head(5)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("Prediction integration test complete.")
print("=" * 60)