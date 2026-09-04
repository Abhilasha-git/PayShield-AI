import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


FEATURE_COLUMNS = [
    "transaction_count",
    "failure_rate",
    "timeout_rate",
    "avg_latency",
    "max_latency",
    "p95_latency",
    "bank_error_rate",
    "avg_amount",
    "max_amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "previous_failure_rate",
    "previous_latency",
    "previous_timeout_rate",
    "failure_rate_change",
    "latency_change",
    "timeout_rate_change",
    "rolling_failure_rate",
    "rolling_latency",
    "rolling_timeout_rate",
]

TARGET_COLUMN = "risk_target"


print("=" * 70)
print("PayShield AI - Risk Threshold Analysis")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

dataset_path = os.path.join(
    "data",
    "processed",
    "ml_dataset.csv"
)

print("\nLoading dataset...")

data = pd.read_csv(dataset_path)

X = data[FEATURE_COLUMNS]
y = data[TARGET_COLUMN]

print(f"Dataset shape: {data.shape}")

# ---------------------------------------------------------
# 2. Same 80/20 split used for evaluation
# ---------------------------------------------------------

print("\nCreating 80/20 train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 3. Train model
# ---------------------------------------------------------

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model trained.")

# ---------------------------------------------------------
# 4. Generate probabilities
# ---------------------------------------------------------

print("\nGenerating risk probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 5. Test multiple thresholds
# ---------------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    true_positives = (
        ((predictions == 1) & (y_test.values == 1))
        .sum()
    )

    false_positives = (
        ((predictions == 1) & (y_test.values == 0))
        .sum()
    )

    false_negatives = (
        ((predictions == 0) & (y_test.values == 1))
        .sum()
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    })

# ---------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
        }
    )
)

# ---------------------------------------------------------
# 7. Find best F1 threshold
# ---------------------------------------------------------

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold : {best_row['threshold']:.2f}"
)

print(
    f"Precision : {best_row['precision']:.4f}"
)

print(
    f"Recall    : {best_row['recall']:.4f}"
)

print(
    f"F1 Score  : {best_row['f1']:.4f}"
)

print(
    f"TP        : {int(best_row['true_positives'])}"
)

print(
    f"FP        : {int(best_row['false_positives'])}"
)

print(
    f"FN        : {int(best_row['false_negatives'])}"
)

print("\n" + "=" * 70)
print("Threshold analysis complete.")
print("=" * 70)