import pandas as pd


class PaymentOptimizationEngine:
    """
    AI-assisted payment optimization engine.

    Uses PayShield AI risk predictions and operational payment
    metrics to recommend appropriate payment actions.
    """

    def __init__(self):
        pass

    def generate_recommendation(self, row):
        """
        Generate an optimization recommendation for one payment
        monitoring/prediction record.
        """

        risk_level = str(row.get("risk_level", "LOW")).upper()

        risk_probability = float(
            row.get("risk_probability", 0.0)
        )

        failure_rate = float(
            row.get("failure_rate", 0.0)
        )

        timeout_rate = float(
            row.get("timeout_rate", 0.0)
        )

        avg_latency = float(
            row.get("avg_latency", 0.0)
        )

        # HIGH RISK
        if risk_level == "HIGH":

            if failure_rate >= 5:
                return {
                    "recommended_action": "Route traffic away",
                    "priority": "CRITICAL",
                    "reason": (
                        "High AI risk combined with elevated "
                        "payment failure rate."
                    ),
                }

            elif timeout_rate >= 3 or avg_latency >= 1000:
                return {
                    "recommended_action": "Prefer alternate healthy route",
                    "priority": "HIGH",
                    "reason": (
                        "High AI risk with timeout or latency pressure."
                    ),
                }

            else:
                return {
                    "recommended_action": "Restrict and monitor traffic",
                    "priority": "HIGH",
                    "reason": (
                        "AI model predicts high payment risk."
                    ),
                }

        # MEDIUM RISK
        elif risk_level == "MEDIUM":

            return {
                "recommended_action": "Monitor closely",
                "priority": "MEDIUM",
                "reason": (
                    "AI model detects moderate payment risk."
                ),
            }

        # ELEVATED RISK
        elif risk_level == "ELEVATED":

            if avg_latency >= 1000:
                return {
                    "recommended_action": "Reduce traffic exposure",
                    "priority": "MEDIUM",
                    "reason": (
                        "Elevated AI risk is associated with "
                        "high payment latency."
                    ),
                }

            elif timeout_rate >= 3:
                return {
                    "recommended_action": "Monitor timeout conditions",
                    "priority": "MEDIUM",
                    "reason": (
                        "Elevated AI risk with increased timeout rate."
                    ),
                }

            else:
                return {
                    "recommended_action": "Monitor and reassess",
                    "priority": "LOW",
                    "reason": (
                        "AI model detects elevated but non-critical risk."
                    ),
                }

        # LOW RISK
        return {
            "recommended_action": "No intervention required",
            "priority": "LOW",
            "reason": (
                "AI model predicts low payment risk."
            ),
        }

    def recommend(self, data):
        """
        Generate recommendations for a DataFrame.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        recommendations = data.apply(
            self.generate_recommendation,
            axis=1
        )

        recommendation_df = pd.DataFrame(
            recommendations.tolist(),
            index=data.index
        )

        output = data.copy()

        output["recommended_action"] = (
            recommendation_df["recommended_action"]
        )

        output["optimization_priority"] = (
            recommendation_df["priority"]
        )

        output["optimization_reason"] = (
            recommendation_df["reason"]
        )

        return output