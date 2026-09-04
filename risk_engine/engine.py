import os
import joblib
import pandas as pd


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


class PayShieldRiskEngine:

    def __init__(self, model=None):
        self.model = model

    # ---------------------------------------------------------
    # Validate input features
    # ---------------------------------------------------------

    def validate_features(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a dictionary or pandas DataFrame."
            )

        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required features: {missing_columns}"
            )

        return data[FEATURE_COLUMNS]

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    def train(self, X, y):
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X, y)

        return self

    # ---------------------------------------------------------
    # Predict probability
    # ---------------------------------------------------------

    def predict_probability(self, data):

        if self.model is None:
            raise ValueError(
                "Risk model is not loaded."
            )

        X = self.validate_features(data)

        probabilities = self.model.predict_proba(X)[:, 1]

        return probabilities

    # ---------------------------------------------------------
    # Convert probability into risk level
    # ---------------------------------------------------------

    def get_risk_level(self, probability):

        if probability >= 0.80:
            return "HIGH"

        elif probability >= 0.50:
            return "MEDIUM"

        elif probability >= 0.20:
            return "ELEVATED"

        else:
            return "LOW"

    # ---------------------------------------------------------
    # Complete prediction
    # ---------------------------------------------------------

    def predict(self, data):

        probabilities = self.predict_probability(data)

        results = []

        for probability in probabilities:

            risk_level = self.get_risk_level(
                probability
            )

            results.append({
                "risk_probability": float(probability),
                "risk_level": risk_level,
                "is_risky": bool(probability >= 0.50)
            })

        return results

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    def save(self, path):

        if self.model is None:
            raise ValueError(
                "Cannot save an empty risk engine."
            )

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        joblib.dump(
            self.model,
            path
        )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    @classmethod
    def load(cls, path):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        model = joblib.load(path)

        return cls(model=model)