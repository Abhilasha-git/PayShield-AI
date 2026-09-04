import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

NUM_TRANSACTIONS = 100_000

banks = [
    "HDFC",
    "ICICI",
    "SBI",
    "AXIS",
    "KOTAK",
    "PNB",
    "BOB",
    "CANARA",
    "IDFC",
    "INDUSIND"
]

upi_apps = [
    "PhonePe",
    "GooglePay",
    "Paytm",
    "BHIM"
]

payment_methods = [
    "UPI"
]

failure_reasons = [
    "BANK_TIMEOUT",
    "BANK_SERVER_ERROR",
    "NETWORK_TIMEOUT",
    "MERCHANT_ERROR",
    "INSUFFICIENT_FUNDS",
    "INVALID_UPI"
]


np.random.seed(42)

# -----------------------------
# Generate realistic timestamps
# -----------------------------

start_time = pd.Timestamp("2026-01-01")
end_time = pd.Timestamp("2026-01-30 23:59:59")

random_seconds = np.random.randint(
    0,
    int((end_time - start_time).total_seconds()),
    NUM_TRANSACTIONS
)

timestamps = (
    start_time +
    pd.to_timedelta(random_seconds, unit="s")
)

timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)

# -----------------------------
# Basic transaction information
# -----------------------------

df = pd.DataFrame({
    "transaction_id": [
        f"TXN{i:07d}"
        for i in range(1, NUM_TRANSACTIONS + 1)
    ],
    "timestamp": timestamps,
    "amount": np.round(
        np.random.lognormal(
            mean=6,
            sigma=1,
            size=NUM_TRANSACTIONS
        ),
        2
    ),
    "sender_bank": np.random.choice(
        banks,
        NUM_TRANSACTIONS
    ),
    "receiver_bank": np.random.choice(
        banks,
        NUM_TRANSACTIONS
    ),
    "payment_method": np.random.choice(
        payment_methods,
        NUM_TRANSACTIONS
    ),
    "upi_app": np.random.choice(
        upi_apps,
        NUM_TRANSACTIONS
    )
})


# -----------------------------
# Simulate bank incidents
# -----------------------------

df["bank_health"] = "NORMAL"

# Each incident has:
# NORMAL → DEGRADED → SEVERE → RECOVERY → NORMAL

incidents = [
    ("SBI", "2026-01-05 10:00", "2026-01-05 18:00"),
    ("HDFC", "2026-01-10 14:00", "2026-01-10 22:00"),
    ("AXIS", "2026-01-15 08:00", "2026-01-15 16:00"),
    ("ICICI", "2026-01-20 10:00", "2026-01-20 18:00"),
    ("KOTAK", "2026-01-25 12:00", "2026-01-25 20:00"),
]

for bank, start, end in incidents:

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Total incident duration
    duration = end - start

    # Phase boundaries
    degraded_start = start
    severe_start = start + duration * 0.25
    recovery_start = start + duration * 0.75

    # DEGRADED phase
    mask_degraded = (
        (df["receiver_bank"] == bank) &
        (df["timestamp"] >= degraded_start) &
        (df["timestamp"] < severe_start)
    )

    df.loc[mask_degraded, "bank_health"] = "DEGRADED"

    # SEVERE phase
    mask_severe = (
        (df["receiver_bank"] == bank) &
        (df["timestamp"] >= severe_start) &
        (df["timestamp"] < recovery_start)
    )

    df.loc[mask_severe, "bank_health"] = "SEVERE"

    # RECOVERY phase
    mask_recovery = (
        (df["receiver_bank"] == bank) &
        (df["timestamp"] >= recovery_start) &
        (df["timestamp"] <= end)
    )

    df.loc[mask_recovery, "bank_health"] = "RECOVERY"


# -----------------------------
# Simulate latency
# -----------------------------

df["latency_ms"] = np.random.normal(
    loc=1200,
    scale=300,
    size=NUM_TRANSACTIONS
)

df["latency_ms"] = np.maximum(
    df["latency_ms"],
    100
)

# Increase latency during incidents
df.loc[
    df["bank_health"] == "DEGRADED",
    "latency_ms"
] *= np.random.uniform(
    2,
    5,
    size=(df["bank_health"] == "DEGRADED").sum()
)


# -----------------------------
# Generate payment outcome
# -----------------------------

success_probability = np.full(
    NUM_TRANSACTIONS,
    0.98
)

# Reduce probability during bank degradation
success_probability[
    df["bank_health"] == "DEGRADED"
] = 0.55

random_values = np.random.random(
    NUM_TRANSACTIONS
)

df["payment_status"] = np.where(
    random_values < success_probability,
    "SUCCESS",
    "FAILED"
)


# -----------------------------
# Generate error codes
# -----------------------------

df["error_code"] = "NONE"

failed_mask = df["payment_status"] == "FAILED"

df.loc[failed_mask, "error_code"] = np.random.choice(
    failure_reasons,
    failed_mask.sum()
)


# -----------------------------
# Add timeout indicator
# -----------------------------

df["timeout"] = (
    df["latency_ms"] > 3000
).astype(int)


# -----------------------------
# Save dataset
# -----------------------------

df.to_csv(
    "data/raw/transactions.csv",
    index=False
)

print("Dataset generated successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nPayment status:")
print(df["payment_status"].value_counts())

print("\nBank health:")
print(df["bank_health"].value_counts())

print("\nFirst 5 rows:")
print(df.head())