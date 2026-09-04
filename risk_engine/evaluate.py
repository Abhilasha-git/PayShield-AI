import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
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


def get_risk_level(probability):
    if probability >= 0.80:
        return "HIGH"
    elif probability >= 0.50:
        return "MEDIUM"
    elif probability >= 0.20:
        return "ELEVATED"
    else:
        return "LOW"


print("=" * 60)
print("PayShield AI - Proper Risk Engine Evaluation")
print("=" * 60)

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

print(f"Dataset shape: {data.shape}")

# ---------------------------------------------------------
# 2. Validate dataset
# ---------------------------------------------------------

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

X = data[FEATURE_COLUMNS]
y = data[TARGET_COLUMN]

print(f"Features: {len(FEATURE_COLUMNS)}")
print(f"Target distribution:")
print(y.value_counts())

# ---------------------------------------------------------
# 3. Create unseen test set
# ---------------------------------------------------------

print("\nCreating 80/20 train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ---------------------------------------------------------
# 4. Train fresh model
# ---------------------------------------------------------

print("\nTraining evaluation model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluation model trained.")

# ---------------------------------------------------------
# 5. Generate predictions on UNSEEN test data
# ---------------------------------------------------------

print("\nGenerating predictions on unseen test data...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 6. Calculate metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("UNSEEN TEST SET PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# ---------------------------------------------------------
# 7. Classification report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ---------------------------------------------------------
# 8. Confusion matrix
# ---------------------------------------------------------

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

matrix = confusion_matrix(y_test, y_pred)

print(matrix)

# ---------------------------------------------------------
# 9. Probability distribution
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RISK PROBABILITY DISTRIBUTION")
print("=" * 60)

print(f"\nMinimum: {y_probability.min():.4f}")
print(f"Maximum: {y_probability.max():.4f}")
print(f"Mean   : {y_probability.mean():.4f}")

# ---------------------------------------------------------
# 10. Risk tiers
# ---------------------------------------------------------

risk_levels = [
    get_risk_level(probability)
    for probability in y_probability
]

risk_distribution = pd.Series(risk_levels).value_counts()

print("\nRisk tier distribution:")

print(risk_distribution)

# ---------------------------------------------------------
# 11. Final message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Proper holdout evaluation complete.")
print("=" * 60)