
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# GENERATE REALISTIC TIMESTAMPS
# ============================================================

start_time = pd.Timestamp("2026-01-01")
end_time = pd.Timestamp("2026-01-30 23:59:59")

random_seconds = np.random.randint(
    0,
    int((end_time - start_time).total_seconds()),
    NUM_TRANSACTIONS
)

timestamps = (
    start_time
    + pd.to_timedelta(random_seconds, unit="s")
)

timestamps = (
    pd.Series(timestamps)
    .sort_values()
    .reset_index(drop=True)
)


# ============================================================
# BASIC TRANSACTION INFORMATION
# ============================================================

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


# ============================================================
# INITIAL SYSTEM HEALTH
# ============================================================

# Existing bank_health = RECEIVER BANK health

df["bank_health"] = "NORMAL"

# New sender-bank health

df["sender_bank_health"] = "NORMAL"

# New gateway/network health

df["gateway_health"] = "NORMAL"


# ============================================================
# RECEIVER BANK INCIDENTS
# ============================================================

receiver_incidents = [
    ("SBI", "2026-01-05 10:00", "2026-01-05 18:00"),
    ("HDFC", "2026-01-10 14:00", "2026-01-10 22:00"),
    ("AXIS", "2026-01-15 08:00", "2026-01-15 16:00"),
    ("ICICI", "2026-01-20 10:00", "2026-01-20 18:00"),
    ("KOTAK", "2026-01-25 12:00", "2026-01-25 20:00"),
]


for bank, start, end in receiver_incidents:

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    duration = end - start

    degraded_start = start
    severe_start = start + duration * 0.25
    recovery_start = start + duration * 0.75

    # DEGRADED

    mask_degraded = (
        (df["receiver_bank"] == bank)
        & (df["timestamp"] >= degraded_start)
        & (df["timestamp"] < severe_start)
    )

    df.loc[
        mask_degraded,
        "bank_health"
    ] = "DEGRADED"

    # SEVERE

    mask_severe = (
        (df["receiver_bank"] == bank)
        & (df["timestamp"] >= severe_start)
        & (df["timestamp"] < recovery_start)
    )

    df.loc[
        mask_severe,
        "bank_health"
    ] = "SEVERE"

    # RECOVERY

    mask_recovery = (
        (df["receiver_bank"] == bank)
        & (df["timestamp"] >= recovery_start)
        & (df["timestamp"] <= end)
    )

    df.loc[
        mask_recovery,
        "bank_health"
    ] = "RECOVERY"


# ============================================================
# SENDER BANK INCIDENTS
# ============================================================

sender_incidents = [
    ("ICICI", "2026-01-07 09:00", "2026-01-07 17:00"),
    ("PNB", "2026-01-12 11:00", "2026-01-12 19:00"),
    ("CANARA", "2026-01-18 13:00", "2026-01-18 21:00"),
    ("IDFC", "2026-01-23 09:00", "2026-01-23 17:00"),
    ("BOB", "2026-01-28 12:00", "2026-01-28 20:00"),
]


for bank, start, end in sender_incidents:

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    duration = end - start

    degraded_start = start
    severe_start = start + duration * 0.25
    recovery_start = start + duration * 0.75

    # DEGRADED

    mask_degraded = (
        (df["sender_bank"] == bank)
        & (df["timestamp"] >= degraded_start)
        & (df["timestamp"] < severe_start)
    )

    df.loc[
        mask_degraded,
        "sender_bank_health"
    ] = "DEGRADED"

    # SEVERE

    mask_severe = (
        (df["sender_bank"] == bank)
        & (df["timestamp"] >= severe_start)
        & (df["timestamp"] < recovery_start)
    )

    df.loc[
        mask_severe,
        "sender_bank_health"
    ] = "SEVERE"

    # RECOVERY

    mask_recovery = (
        (df["sender_bank"] == bank)
        & (df["timestamp"] >= recovery_start)
        & (df["timestamp"] <= end)
    )

    df.loc[
        mask_recovery,
        "sender_bank_health"
    ] = "RECOVERY"


# ============================================================
# GATEWAY / NETWORK INCIDENTS
# ============================================================

gateway_incidents = [
    ("2026-01-08 15:00", "2026-01-08 18:00"),
    ("2026-01-16 11:00", "2026-01-16 14:00"),
    ("2026-01-26 16:00", "2026-01-26 20:00"),
]


for start, end in gateway_incidents:

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    duration = end - start

    degraded_start = start
    severe_start = start + duration * 0.30
    recovery_start = start + duration * 0.80

    # DEGRADED

    mask_degraded = (
        (df["timestamp"] >= degraded_start)
        & (df["timestamp"] < severe_start)
    )

    df.loc[
        mask_degraded,
        "gateway_health"
    ] = "DEGRADED"

    # SEVERE

    mask_severe = (
        (df["timestamp"] >= severe_start)
        & (df["timestamp"] < recovery_start)
    )

    df.loc[
        mask_severe,
        "gateway_health"
    ] = "SEVERE"

    # RECOVERY

    mask_recovery = (
        (df["timestamp"] >= recovery_start)
        & (df["timestamp"] <= end)
    )

    df.loc[
        mask_recovery,
        "gateway_health"
    ] = "RECOVERY"


# ============================================================
# BASE LATENCY
# ============================================================

df["latency_ms"] = np.random.normal(
    loc=1200,
    scale=300,
    size=NUM_TRANSACTIONS
)

df["latency_ms"] = np.maximum(
    df["latency_ms"],
    100
)


# ============================================================
# RECEIVER BANK LATENCY IMPACT
# ============================================================

receiver_degraded = (
    df["bank_health"] == "DEGRADED"
)

receiver_severe = (
    df["bank_health"] == "SEVERE"
)

receiver_recovery = (
    df["bank_health"] == "RECOVERY"
)

df.loc[
    receiver_degraded,
    "latency_ms"
] *= np.random.uniform(
    2,
    4,
    size=receiver_degraded.sum()
)

df.loc[
    receiver_severe,
    "latency_ms"
] *= np.random.uniform(
    3,
    6,
    size=receiver_severe.sum()
)

df.loc[
    receiver_recovery,
    "latency_ms"
] *= np.random.uniform(
    1.2,
    2,
    size=receiver_recovery.sum()
)


# ============================================================
# SENDER BANK LATENCY IMPACT
# ============================================================

sender_degraded = (
    df["sender_bank_health"] == "DEGRADED"
)

sender_severe = (
    df["sender_bank_health"] == "SEVERE"
)

sender_recovery = (
    df["sender_bank_health"] == "RECOVERY"
)

df.loc[
    sender_degraded,
    "latency_ms"
] *= np.random.uniform(
    1.5,
    3,
    size=sender_degraded.sum()
)

df.loc[
    sender_severe,
    "latency_ms"
] *= np.random.uniform(
    2,
    4,
    size=sender_severe.sum()
)

df.loc[
    sender_recovery,
    "latency_ms"
] *= np.random.uniform(
    1.1,
    1.7,
    size=sender_recovery.sum()
)


# ============================================================
# GATEWAY LATENCY IMPACT
# ============================================================

gateway_degraded = (
    df["gateway_health"] == "DEGRADED"
)

gateway_severe = (
    df["gateway_health"] == "SEVERE"
)

gateway_recovery = (
    df["gateway_health"] == "RECOVERY"
)

df.loc[
    gateway_degraded,
    "latency_ms"
] *= np.random.uniform(
    1.5,
    2.5,
    size=gateway_degraded.sum()
)

df.loc[
    gateway_severe,
    "latency_ms"
] *= np.random.uniform(
    2,
    3.5,
    size=gateway_severe.sum()
)

df.loc[
    gateway_recovery,
    "latency_ms"
] *= np.random.uniform(
    1.1,
    1.5,
    size=gateway_recovery.sum()
)


# ============================================================
# PAYMENT SUCCESS PROBABILITY
# ============================================================

success_probability = np.full(
    NUM_TRANSACTIONS,
    0.98
)


# Receiver bank impact

success_probability[
    receiver_degraded
] -= 0.30

success_probability[
    receiver_severe
] -= 0.55

success_probability[
    receiver_recovery
] -= 0.10


# Sender bank impact

success_probability[
    sender_degraded
] -= 0.25

success_probability[
    sender_severe
] -= 0.50

success_probability[
    sender_recovery
] -= 0.08


# Gateway impact

success_probability[
    gateway_degraded
] -= 0.20

success_probability[
    gateway_severe
] -= 0.40

success_probability[
    gateway_recovery
] -= 0.05


# Keep probabilities valid

success_probability = np.clip(
    success_probability,
    0.05,
    0.99
)


# ============================================================
# PAYMENT OUTCOME
# ============================================================

random_values = np.random.random(
    NUM_TRANSACTIONS
)

df["payment_status"] = np.where(
    random_values < success_probability,
    "SUCCESS",
    "FAILED"
)


# ============================================================
# ERROR CODES
# ============================================================

df["error_code"] = "NONE"

failed_mask = (
    df["payment_status"] == "FAILED"
)

df.loc[
    failed_mask,
    "error_code"
] = np.random.choice(
    failure_reasons,
    failed_mask.sum()
)


# ============================================================
# ROOT-CAUSE-AWARE ERROR CODES
# ============================================================

# Sender bank failures

sender_failure_mask = (
    failed_mask
    & df["sender_bank_health"].isin(
        ["DEGRADED", "SEVERE"]
    )
)

df.loc[
    sender_failure_mask,
    "error_code"
] = np.random.choice(
    [
        "BANK_TIMEOUT",
        "BANK_SERVER_ERROR"
    ],
    sender_failure_mask.sum()
)


# Receiver bank failures

receiver_failure_mask = (
    failed_mask
    & df["bank_health"].isin(
        ["DEGRADED", "SEVERE"]
    )
)

df.loc[
    receiver_failure_mask,
    "error_code"
] = np.random.choice(
    [
        "BANK_TIMEOUT",
        "BANK_SERVER_ERROR"
    ],
    receiver_failure_mask.sum()
)


# Gateway failures

gateway_failure_mask = (
    failed_mask
    & df["gateway_health"].isin(
        ["DEGRADED", "SEVERE"]
    )
)

df.loc[
    gateway_failure_mask,
    "error_code"
] = np.random.choice(
    [
        "NETWORK_TIMEOUT",
        "BANK_TIMEOUT"
    ],
    gateway_failure_mask.sum()
)


# ============================================================
# TIMEOUT INDICATOR
# ============================================================

df["timeout"] = (
    df["latency_ms"] > 3000
).astype(int)


# ============================================================
# ROOT CAUSE LABEL
# ============================================================

df["root_cause"] = "NORMAL"


# Gateway has highest diagnostic priority

gateway_root_cause = (
    df["gateway_health"].isin(
        ["DEGRADED", "SEVERE"]
    )
)

df.loc[
    gateway_root_cause,
    "root_cause"
] = "GATEWAY_DEGRADED"


# Sender bank

sender_root_cause = (
    (df["root_cause"] == "NORMAL")
    & df["sender_bank_health"].isin(
        ["DEGRADED", "SEVERE"]
    )
)

df.loc[
    sender_root_cause,
    "root_cause"
] = "SENDER_BANK_DEGRADED"


# Receiver bank

receiver_root_cause = (
    (df["root_cause"] == "NORMAL")
    & df["bank_health"].isin(
        ["DEGRADED", "SEVERE"]
    )
)

df.loc[
    receiver_root_cause,
    "root_cause"
] = "RECEIVER_BANK_DEGRADED"


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    "data/raw/transactions.csv",
    index=False
)


# ============================================================
# VALIDATION OUTPUT
# ============================================================

print("=" * 60)
print("PayShield AI - Dataset Generation")
print("=" * 60)

print("\nDataset generated successfully!")

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print("\nPayment status:")
print(
    df["payment_status"].value_counts()
)

print("\nReceiver bank health:")
print(
    df["bank_health"].value_counts()
)

print("\nSender bank health:")
print(
    df["sender_bank_health"].value_counts()
)

print("\nGateway health:")
print(
    df["gateway_health"].value_counts()
)

print("\nRoot-cause distribution:")
print(
    df["root_cause"].value_counts()
)

print("\nAverage latency by root cause:")

print(
    df.groupby("root_cause")["latency_ms"]
    .mean()
    .round(2)
)

print("\nFailure rate by root cause:")

failure_rate = (
    df.groupby("root_cause")["payment_status"]
    .apply(
        lambda x: (
            x == "FAILED"
        ).mean() * 100
    )
    .round(2)
)

print(
    failure_rate
)

print("\nFirst 5 rows:")
print(
    df.head()
)

print("\n" + "=" * 60)
print("Dataset generation complete.")
print("=" * 60)