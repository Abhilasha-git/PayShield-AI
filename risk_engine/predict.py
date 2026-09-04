import os
import pandas as pd

from risk_engine.engine import PayShieldRiskEngine


MODEL_PATH = os.path.join(
    "models",
    "payshield_risk_model.joblib"
)


def load_risk_engine():
    """
    Load the trained PayShield Risk Engine.
    """
    return PayShieldRiskEngine.load(
        MODEL_PATH
    )


def predict_risk(data):
    """
    Generate risk predictions for monitoring data.
    """

    engine = load_risk_engine()

    results = engine.predict(data)

    results_df = pd.DataFrame(results)

    output = data.copy()

    output["risk_probability"] = (
        results_df["risk_probability"].values
    )

    output["risk_level"] = (
        results_df["risk_level"].values
    )

    output["is_risky"] = (
        results_df["is_risky"].values
    )

    return output