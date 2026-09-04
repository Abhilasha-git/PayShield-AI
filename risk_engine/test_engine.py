import os

from risk_engine.engine import PayShieldRiskEngine


MODEL_PATH = os.path.join(
    "models",
    "payshield_risk_model.joblib"
)


print("=" * 60)
print("PayShield AI - Risk Engine Test")
print("=" * 60)

# ---------------------------------------------------------
# Load engine
# ---------------------------------------------------------

print("\nLoading Risk Engine...")

engine = PayShieldRiskEngine.load(
    MODEL_PATH
)

print("Risk Engine loaded successfully.")

# ---------------------------------------------------------
# Sample transaction features
# ---------------------------------------------------------

sample_transaction = {
    "transaction_count": 100,
    "failure_rate": 0.10,
    "timeout_rate": 0.05,
    "avg_latency": 250,
    "max_latency": 800,
    "p95_latency": 600,
    "bank_error_rate": 0.08,
    "avg_amount": 1500,
    "max_amount": 5000,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "previous_failure_rate": 0.08,
    "previous_latency": 230,
    "previous_timeout_rate": 0.04,
    "failure_rate_change": 0.02,
    "latency_change": 20,
    "timeout_rate_change": 0.01,
    "rolling_failure_rate": 0.09,
    "rolling_latency": 240,
    "rolling_timeout_rate": 0.04
}

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

print("\nGenerating risk prediction...")

result = engine.predict(
    sample_transaction
)[0]

# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RISK RESULT")
print("=" * 60)

print(
    f"\nRisk Probability: "
    f"{result['risk_probability']:.4f}"
)

print(
    f"Risk Level      : "
    f"{result['risk_level']}"
)

print(
    f"Is Risky        : "
    f"{result['is_risky']}"
)

print("\n" + "=" * 60)
print("Risk Engine test complete.")
print("=" * 60)