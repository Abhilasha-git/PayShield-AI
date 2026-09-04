import os
import pandas as pd

from risk_engine.engine import PayShieldRiskEngine


MODEL_PATH = os.path.join(
    "models",
    "payshield_risk_model.joblib"
)

DATA_PATH = os.path.join(
    "data",
    "processed",
    "payment_monitoring_enhanced.csv"
)


print("=" * 60)
print("PayShield AI - Real Data Risk Engine Test")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load Risk Engine
# ---------------------------------------------------------

print("\nLoading Risk Engine...")

engine = PayShieldRiskEngine.load(
    MODEL_PATH
)

print("Risk Engine loaded successfully.")

# ---------------------------------------------------------
# 2. Load real monitoring data
# ---------------------------------------------------------

print("\nLoading monitoring data...")

data = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {data.shape}")

# ---------------------------------------------------------
# 3. Generate predictions
# ---------------------------------------------------------

print("\nGenerating predictions...")

results = engine.predict(data)

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 4. Combine with original data
# ---------------------------------------------------------

output = data.copy()

output["risk_probability"] = results_df[
    "risk_probability"
]

output["risk_level"] = results_df[
    "risk_level"
]

output["is_risky"] = results_df[
    "is_risky"
]

# ---------------------------------------------------------
# 5. Display summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RISK ENGINE RESULTS")
print("=" * 60)

print(
    f"\nTotal records: {len(output)}"
)

print(
    f"Risky records: "
    f"{output['is_risky'].sum()}"
)

print(
    f"Average risk: "
    f"{output['risk_probability'].mean():.4f}"
)

print("\nRisk level distribution:")

print(
    output["risk_level"].value_counts()
)

# ---------------------------------------------------------
# 6. Show highest-risk records
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TOP 10 HIGHEST-RISK RECORDS")
print("=" * 60)

top_risk = output.sort_values(
    "risk_probability",
    ascending=False
).head(10)

print(
    top_risk[
        [
            "receiver_bank",
            "time_window",
            "failure_rate",
            "timeout_rate",
            "avg_latency",
            "risk_probability",
            "risk_level",
            "is_risky",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("Real data test complete.")
print("=" * 60)