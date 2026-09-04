# PayShield AI

## AI-Powered Payment Risk Monitoring and Success Optimization System

PayShield AI is a machine-learning-based payment intelligence system designed to predict payment failure risk, identify payment-system anomalies, and recommend appropriate optimization actions.

The system combines payment transaction analytics, machine learning, risk scoring, operational diagnostics, pre-payment risk assessment, and AI-assisted optimization recommendations in an interactive Streamlit dashboard.

---

## 1. Problem Statement

Payment failures can occur because of:

- High bank failure rates
- Gateway timeout spikes
- Increased payment latency
- Downstream bank outages
- Abnormal transaction behavior

Traditional monitoring systems primarily report these problems after they occur.

PayShield AI uses machine learning to estimate payment failure risk and converts those predictions into actionable optimization recommendations.

---

## 2. Project Objectives

The main objectives of PayShield AI are:

1. Monitor payment-system health.
2. Detect potentially risky payment conditions.
3. Predict payment failure risk using machine learning.
4. Classify predictions into standardized risk tiers.
5. Identify operational risk drivers.
6. Generate AI-assisted payment optimization recommendations.
7. Provide an interactive monitoring dashboard.
8. Support targeted intervention instead of broad traffic disruption.

---

## 3. AI / Machine Learning Component

The core AI model is a Random Forest Classifier.

### Model Configuration

- Algorithm: Random Forest Classifier
- Number of estimators: 200
- Class weighting: Balanced
- Random state: 42
- Parallel processing: Enabled

The model uses 21 operational and behavioral features.

### ML Features

- transaction_count
- failure_rate
- timeout_rate
- avg_latency
- max_latency
- p95_latency
- bank_error_rate
- avg_amount
- max_amount
- hour
- day_of_week
- is_weekend
- previous_failure_rate
- previous_latency
- previous_timeout_rate
- failure_rate_change
- latency_change
- timeout_rate_change
- rolling_failure_rate
- rolling_latency
- rolling_timeout_rate

---

## 4. AI Prediction Pipeline

PayShield AI supports risk prediction from payment transaction and payment-system telemetry.

The general prediction pipeline is:
```text

Payment / Monitoring Data
        |
        v
Feature Engineering
        |
        v
21 ML Features
        |
        v
Random Forest Classifier
        |
        v
Risk Probability
        |
        v
Risk Tier
        |
        v
Optimization Recommendation
        |
        v
Dashboard
```

The model produces a probability representing the estimated likelihood of payment failure for the supplied transaction or payment-system context.

---

## 5. Risk Classification

PayShield AI uses four risk tiers:

| Risk Tier | Probability |
|-----------|-------------|
| LOW | < 20% |
| ELEVATED | 20% - < 50% |
| MEDIUM | 50% - < 80% |
| HIGH | >= 80% |

For binary risk identification, a probability of 50% or higher is treated as risky.

---

## 6. Model Evaluation

The model was evaluated using a separate 20% holdout test set.

### Evaluation Results

- Accuracy: 99.82%
- Precision: 91.30%
- Recall: 75.00%
- F1 Score: 82.35%
- ROC-AUC: 93.98%

### Holdout Confusion Matrix

| | Predicted Normal | Predicted Risky |
|---|---:|---:|
| Actual Normal | 10,129 | 4 |
| Actual Risky | 14 | 42 |

The evaluation demonstrates that the model can distinguish risky payment conditions while maintaining high precision.

---

## 7. Threshold Analysis

Multiple probability thresholds were evaluated.

The selected binary risk threshold is:

**50%**

This threshold produced the strongest F1 score among the tested thresholds:

- Precision: 91.30%
- Recall: 75.00%
- F1 Score: 82.35%

---

## 8. Risk Engine

The PayShield Risk Engine provides a reusable interface around the trained machine-learning model.

### Main Capabilities

- Feature validation
- Model loading
- Risk probability prediction
- Risk-tier classification
- Binary risky/not-risky classification
- Model persistence

### Model File

```text
models/payshield_risk_model.joblib
```

---

## 9. AI Payment Success Optimization

The optimization layer converts AI risk predictions into recommended operational actions.

Examples include:

| Detected Condition | AI-Assisted Recommendation |
|---|---|
| HIGH risk + high failure rate | Route traffic away |
| HIGH risk + timeout/latency pressure | Prefer alternate healthy route |
| MEDIUM risk | Monitor closely |
| ELEVATED risk + high latency | Reduce traffic exposure |
| LOW risk | No intervention required |

This allows the system to move beyond monitoring and provide operational decision support.

---

## 10. Optimization Priorities

Recommendations are assigned priorities:

- CRITICAL
- HIGH
- MEDIUM
- LOW

The current dashboard applies the optimization engine to the AI-generated risk predictions.

The validated dashboard output includes:

- 33 Critical actions
- 23 High-priority actions
- 8 Medium-priority actions
- 10,125 Low-priority / no-intervention cases

The 33 Critical and 23 High-priority recommendations correspond to the 56 HIGH-risk predictions, while the remaining recommendations are associated with MEDIUM, ELEVATED, or LOW-risk conditions.

---

## 11. Risk-Driver Diagnostics

PayShield AI provides operational explanations using payment-system indicators such as:

- Elevated bank failure rate
- Gateway timeout surge
- High latency spike
- Model anomaly score

These diagnostics help users understand which operational indicators are associated with elevated payment risk.

---

## 12. Bank Health Monitoring

The dashboard monitors receiver-bank performance using:

- Bank health status
- Failure rate
- Timeout rate
- Average latency
- Historical performance trends

A dual-axis trend visualization separates latency from failure and timeout percentages for clearer interpretation.

---

## 13. Dashboard

The Streamlit dashboard provides the following analytical sections.

### System Health

- Payment success rate
- Average latency
- Transactions monitored
- High-risk prediction count

### Bank Monitoring

- Bank selector
- Bank health
- Failure rate
- Timeout rate
- Average latency
- Historical trend

### Risk Monitoring

- HIGH
- MEDIUM
- ELEVATED
- LOW

### Pre-Payment AI Risk Check

PayShield AI includes a pre-payment risk simulation that estimates payment failure risk before authorization using the latest available receiver-bank and payment-system conditions.

The feature is designed as a decision-support layer that can identify elevated payment risk before a payment is authorized and recommend an appropriate action.

The simulator allows the user to:

- Select a receiver bank
- Enter a transaction amount
- Select the payment hour
- Evaluate the current payment-system conditions
- Run a controlled degraded-bank stress simulation
- Receive an AI risk probability
- Receive a risk tier
- Receive an AI-assisted recommended action

Example validated scenarios:

| Scenario | AI Risk | Risk Tier | Recommendation |
|---|---:|---|---|
| Healthy baseline | 0.00% | LOW | No intervention required |
| Degraded bank simulation | 84.50% | HIGH | Route traffic away |

The stress mode is a controlled simulation used to demonstrate how the risk engine responds to degraded payment-system conditions. It does not represent live bank, UPI, or payment-gateway telemetry.

### Diagnostics

- Risk filtering
- Probability filtering
- Risk-driver filtering
- Transaction-level investigation
- Risk-driver diagnostics

### AI Optimization

- Critical actions
- High-priority actions
- Medium-priority actions
- No-intervention cases
- Recommended payment actions
- AI risk
- Optimization priority
- AI reasoning

### Risk Trend

- Model risk probability trend
- Rolling risk average
- Operational risk monitoring

---

## 14. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Streamlit

---

## 15. Project Structure

```text
PayShield AI/
├── app/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   │   └── transactions.csv
│   │
│   └── processed/
│       ├── payment_monitoring_enhanced.csv
│       └── payShield_predictions.csv
│
├── models/
│   └── payshield_risk_model.joblib
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_dataset.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_threshold_optimization.ipynb
│   ├── 08_explainability.ipynb
│   └── 09_realyime_prediction.ipynb
│
├── optimization_engine/
│   ├── __init__.py
│   └── recommendation_engine.py
│
├── risk_engine/
│   ├── __init__.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── test_engine.py
│   ├── test_predict.py
│   ├── test_real_data.py
│   ├── threshold_analysis.py
│   └── train.py
│
├── src/
│   └── data_generation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 16. Running the Project

From the project root:

```powershell
streamlit run app\dashboard.py
```

---

## 17. Current AI Architecture

```text
                         PAYMENT / MONITORING DATA
                              |
                              v
                       FEATURE ENGINEERING
                              |
                              v
                         21 ML FEATURES
                              |
                              v
                  +-------------------------+
                  | Random Forest Classifier |
                  |        AI MODEL          |
                  +-------------------------+
                              |
                              v
                       RISK PROBABILITY
                              |
                              v
                         RISK TIER
                              |
                    +---------+---------+
                    |                   |
                    v                   v
             RISK DIAGNOSTICS    OPTIMIZATION ENGINE
                                        |
                                        v
                              AI-ASSISTED RECOMMENDATION
                                        |
                                        v
                              STREAMLIT DASHBOARD


             PRE-PAYMENT AI RISK CHECK
                        |
                        v
              Receiver Bank + Amount
                        |
                        v
                Current Telemetry
                        |
                        v
                  PayShield AI
                        |
                        v
               Risk + Recommendation
```

---

## 18. AI-Assisted Optimization

The optimization engine uses the AI-generated risk level together with operational payment indicators to recommend appropriate actions.

For example:

- High AI risk combined with high failure rate can trigger a recommendation to route traffic away.
- High AI risk combined with timeout or latency pressure can trigger a recommendation to prefer an alternate healthy route.
- Medium risk can trigger closer monitoring.
- Elevated risk with high latency can trigger reduced traffic exposure.
- Low-risk conditions require no intervention.

The optimization engine is therefore a decision-support layer built on top of the machine-learning risk engine.

---

## 19. Important AI Limitation

The optimization engine is an **AI-assisted recommendation layer**, not a learned causal optimization policy.

The current dataset does not contain sufficient historical intervention outcomes or counterfactual routing experiments to claim that the system has learned the globally optimal payment-routing decision.

The current system should therefore be described as:

> **Machine-learning-based payment risk prediction with AI-assisted optimization recommendations.**

Future versions can incorporate historical intervention data such as:

- Historical routing decisions
- Gateway selection
- Intervention outcomes
- Payment success after intervention
- Transaction cost
- Latency impact
- A/B testing results

These additional data could support a learned optimization or decision-policy model.

---

## 20. Future Scope

Potential future improvements include:

1. Real-time payment-stream ingestion.
2. Automated bank/gateway routing.
3. Model explainability using SHAP.
4. Online model monitoring.
5. Automated alerting.
6. Merchant-level optimization.
7. Adaptive retry optimization.
8. Reinforcement-learning-based routing.
9. A/B testing of optimization recommendations.
10. Cloud deployment and API integration.

---

## 21. Validation Summary

The current prototype has been validated across its core dashboard workflows.

### Pre-Payment Risk Validation

- Healthy baseline: 0.00% LOW risk
- Degraded simulation: 84.50% HIGH risk
- HIGH-risk recommendation: Route traffic away
- No feature-schema errors during stress simulation

### Dashboard Validation

- Bank health monitoring verified
- Dual-axis latency/failure visualization verified
- Four-tier risk distribution verified
- 56 HIGH-risk records correctly surfaced
- Risk-driver diagnostics verified
- AI optimization recommendations verified

The degraded-bank scenario is a controlled simulation used to demonstrate model responsiveness and should not be interpreted as live bank or UPI telemetry.

---