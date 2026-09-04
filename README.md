# PayShield AI

### Machine-Learning-Based Payment Risk Prediction and AI-Assisted Revenue Recovery

🚀 **Live Demo:** https://abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app/

📦 **GitHub Repository:** https://github.com/Abhilasha-git/PayShield-AI

PayShield AI is an intelligent payment monitoring and decision-support system designed to identify payment failure risk before authorization and recommend actions that can improve payment success.

The system combines machine-learning-based risk prediction with payment-system health signals from sender banks, receiver banks, and payment gateways. It diagnoses the likely source of payment degradation and produces an AI-assisted recovery recommendation such as monitoring, retrying through an alternate route, or routing traffic away from an unhealthy payment environment.

> **Current implementation:** PayShield AI is a local simulation and decision-intelligence prototype. It does not connect to live UPI networks, bank accounts, payment gateways, or real payment authorization systems.

---

## Problem

Payment failures can result from several interacting conditions:

* Sender-bank degradation
* Receiver-bank degradation
* Payment gateway degradation
* Increased latency
* Payment timeouts
* Elevated transaction failure rates
* Sudden changes in payment-system behaviour

A payment failure is not always caused by the same component. Simply retrying a failed payment may therefore be ineffective when the underlying payment route is experiencing degradation.

PayShield AI addresses this problem through a four-stage intelligence workflow:

```text
DETECT
   â†“
DIAGNOSE
   â†“
DECIDE
   â†“
RECOVER
```

The system first detects abnormal payment behaviour, identifies the most likely source of degradation, estimates payment risk, and then recommends an appropriate recovery action.

---

## Solution

### PayShield AI provides:

### 1. Payment Risk Prediction

* Predicts the probability of payment failure.
* Uses transaction behaviour, latency, timeout, and historical payment signals.

### 2. Bank Health Monitoring

* Monitors simulated sender-bank and receiver-bank conditions.
* Identifies normal, degraded, severe, and recovery states.

### 3. Gateway Health Monitoring

* Models simulated payment-gateway degradation and latency/timeout surges.

### 4. Root-Cause Diagnosis

* Distinguishes between sender-bank, receiver-bank, and gateway-related degradation.

### 5. Pre-Payment Risk Assessment

* Allows a payment scenario to be evaluated before authorization.
* Provides a simulated risk probability and risk tier.

### 6. AI-Assisted Recovery Recommendations

Recommends actions such as:

* No intervention
* Monitor closely
* Reduce traffic exposure
* Prefer an alternate healthy route
* Route traffic away from an unhealthy route

### 7. Interactive Monitoring Dashboard

Displays:

* Payment success rate
* Average latency
* Transaction volume
* High-risk predictions
* Bank health
* Gateway/system health
* Risk distribution
* Optimization priorities
* Diagnostic recommendations

---

## Architecture

```text
                    Payment Attempt
                          â”‚
                          â–¼
              Collect Payment Signals
                          â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â–¼               â–¼               â–¼
     Sender Bank     Receiver Bank    Gateway Health
        Health           Health
          â”‚               â”‚               â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â–¼
                 Root-Cause Diagnosis
                          â”‚
                          â–¼
                  ML Risk Prediction
                          â”‚
                          â–¼
                  Risk Classification
                          â”‚
                          â–¼
                   Recovery Decision
                          â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â–¼               â–¼               â–¼
        Retry       Alternate Route     Monitor
                          â”‚
                          â–¼
                    Measure / Improve
```

---

## Key Workflow

### 1. Detect

The system monitors payment behaviour and identifies abnormal conditions using:

* Failure rate
* Timeout rate
* Transaction latency
* Error behaviour
* Historical failure patterns
* Rolling payment metrics
* Recent changes in payment behaviour

### 2. Diagnose

PayShield AI evaluates simulated payment-system health to determine the most likely source of degradation.

**Possible primary causes include:**

```text
NORMAL
GATEWAY_DEGRADED
SENDER_BANK_DEGRADED
RECEIVER_BANK_DEGRADED
```

The system prioritizes gateway, sender-bank, and receiver-bank conditions when multiple degradation signals are present.

### 3. Decide

The machine-learning model produces a payment-risk probability.

Risk levels are classified as:

| Risk Probability | Risk Level |
| ---------------: | ---------- |
|            < 20% | LOW        |
|      20% â€“ < 50% | ELEVATED   |
|      50% â€“ < 80% | MEDIUM     |
|            â‰¥ 80% | HIGH       |

The current decision threshold for binary risky/non-risky classification is **50%**.

### 4. Recover

Based on risk, latency, timeout, failure behaviour, and payment-system health, the optimization layer recommends an appropriate intervention.

**Examples:**

| Situation                         | Recommended Action                |
| --------------------------------- | --------------------------------- |
| Low payment risk                  | No intervention required          |
| Elevated risk + high latency      | Reduce traffic exposure           |
| Medium risk                       | Monitor closely                   |
| High risk + latency/timeout surge | Prefer an alternate healthy route |
| High risk + severe degradation    | Route traffic away                |

The current optimization engine is an **AI-assisted recommendation layer** rather than a fully autonomous routing policy.

---

## Machine Learning Model

PayShield AI currently uses a **Random Forest Classifier** for payment-failure risk prediction.

### Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The model uses behavioural and rolling payment features including:

* Transaction count
* Failure rate
* Timeout rate
* Average latency
* Maximum latency
* P95 latency
* Bank error rate
* Average transaction amount
* Maximum transaction amount
* Hour
* Day of week
* Weekend indicator
* Previous failure rate
* Previous latency
* Previous timeout rate
* Failure-rate change
* Latency-change rate
* Timeout-rate change
* Rolling failure rate
* Rolling latency
* Rolling timeout rate

---

## Model Evaluation

The current holdout evaluation produced:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 99.82% |
| Precision | 91.30% |
| Recall    | 75.00% |
| F1 Score  | 82.35% |
| ROC-AUC   | 93.98% |

### Holdout Confusion Matrix

```text
                 Predicted
                Normal  Risky
Actual Normal    10129      4
Actual Risky        14     42
```

The model is designed to prioritize detection of risky payment behaviour while maintaining a practical balance between precision and recall.

---

## Dataset

PayShield AI uses a simulated payment transaction dataset containing **100,000 transactions**.

The dataset includes:

```text
transaction_id
timestamp
amount
sender_bank
receiver_bank
payment_method
upi_app
bank_health
sender_bank_health
gateway_health
latency_ms
payment_status
error_code
timeout
root_cause
```

The simulated environment contains multiple Indian bank identifiers and payment-system conditions.

### Simulated System States

```text
NORMAL
DEGRADED
SEVERE
RECOVERY
```

The dataset includes controlled degradation scenarios for:

* Receiver banks
* Sender banks
* Payment gateways

This allows the project to demonstrate how payment risk can change when different components of a payment route experience degradation.

---

## Dashboard

PayShield AI includes an interactive dashboard built with **Streamlit**.

The dashboard provides two major capabilities.

### Payment Monitoring

The monitoring interface displays:

* Payment success rate
* Average latency
* Number of transactions monitored
* High-risk predictions
* Bank health
* Failure rates
* Timeout rates
* Risk distribution
* Optimization priorities
* Current system status

### Pre-Payment Check

The dashboard provides an interactive payment-risk simulation.

Users can select:

* Sender bank
* Receiver bank
* Transaction amount
* Payment-system scenario

Available demonstration scenarios include:

```text
Healthy Payment Route
Sender Bank Degradation
Receiver Bank Degradation
Payment Gateway Latency & Timeout Surge
```

The system then produces:

* Final payment-risk probability
* Risk tier
* ML model probability
* Primary payment-system issue
* Sender-bank health
* Receiver-bank health
* Gateway health
* Recommended action

---

## Example Decision

For a simulated severe sender-bank degradation scenario, PayShield AI can produce a decision such as:

```text
HIGH PAYMENT RISK

Risk Probability: 85.0%

Primary Issue:
Sender Bank Degradation

Sender Health:
Severe

Recommended Action:
Route traffic away
```

This demonstrates how payment-system health can influence a pre-payment decision.

The dashboard also displays the underlying machine-learning probability separately from the final simulation decision.

---

## Optimization Engine

The optimization layer converts risk and system-health information into an actionable recommendation.

### Example Decision Flow

```text
Risk Probability
       +
Failure Behaviour
       +
Latency / Timeout Behaviour
       +
Sender Health
       +
Receiver Health
       +
Gateway Health
       â”‚
       â–¼
Recovery Recommendation
```

### Current Recommendations

```text
No intervention required
Monitor closely
Reduce traffic exposure
Prefer alternate healthy route
Route traffic away
```

The current implementation does not claim that these recommendations are learned causal policies.

A production-grade optimization policy would require historical intervention outcomes, counterfactual routing experiments, and controlled A/B testing.

---

## Project Structure

```text
PayShield AI/
â”‚
â”œâ”€â”€ app/
â”‚   â””â”€â”€ dashboard.py
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ transactions.csv
â”‚   â”‚
â”‚   â””â”€â”€ processed/
â”‚       â”œâ”€â”€ payment_monitoring_enhanced.csv
â”‚       â””â”€â”€ payShield_predictions.csv
â”‚
â”œâ”€â”€ models/
â”‚   â””â”€â”€ payshield_risk_model.joblib
â”‚
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_data_generation.ipynb
â”‚   â”œâ”€â”€ 02_eda.ipynb
â”‚   â”œâ”€â”€ 03_feature_engineering.ipynb
â”‚   â”œâ”€â”€ 04_ml_dataset.ipynb
â”‚   â”œâ”€â”€ 05_model_training.ipynb
â”‚   â”œâ”€â”€ 06_model_comparison.ipynb
â”‚   â”œâ”€â”€ 07_threshold_optimization.ipynb
â”‚   â”œâ”€â”€ 08_explainability.ipynb
â”‚   â””â”€â”€ 09_realyime_prediction.ipynb
â”‚
â”œâ”€â”€ optimization_engine/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â””â”€â”€ recommendation_engine.py
â”‚
â”œâ”€â”€ risk_engine/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ engine.py
â”‚   â”œâ”€â”€ evaluate.py
â”‚   â”œâ”€â”€ predict.py
â”‚   â”œâ”€â”€ threshold_analysis.py
â”‚   â”œâ”€â”€ train.py
â”‚   â”œâ”€â”€ test_engine.py
â”‚   â”œâ”€â”€ test_predict.py
â”‚   â””â”€â”€ test_real_data.py
â”‚
â”œâ”€â”€ src/
â”‚   â””â”€â”€ data_generation.py
â”‚
â”œâ”€â”€ .gitignore
â”œâ”€â”€ README.md
â””â”€â”€ requirements.txt
```

---

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Abhilasha-git/PayShield-AI.git
cd PayShield-AI
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the Dashboard

```powershell
streamlit run app/dashboard.py
```

The Streamlit application will open in the browser.

---

## Risk Engine

The trained model is stored at:

```text
models/payshield_risk_model.joblib
```

The risk engine can be used independently of the dashboard.

Example:

```python
from risk_engine.predict import predict_risk

predictions = predict_risk(data)
```

The prediction pipeline adds:

```text
risk_probability
risk_level
is_risky
```

to the prediction dataset.

---

## Testing

The project includes standalone validation scripts for the risk engine.

Run:

```powershell
python -m risk_engine.test_engine
```

```powershell
python -m risk_engine.test_predict
```

```powershell
python -m risk_engine.test_real_data
```

These scripts validate:

* Model loading
* Risk prediction
* Risk classification
* Prediction output structure
* Risk distribution on processed data

---

## Buildathon Track Alignment

PayShield AI is primarily aligned with the **AI Revenue Recovery** track.

The project follows the concept:

```text
Payment degradation
       â†“
Risk detection
       â†“
Root-cause diagnosis
       â†“
Recovery recommendation
```

The system focuses on identifying payment revenue at risk and recommending an appropriate intervention rather than simply predicting whether a transaction will fail.

The current prototype demonstrates the intelligence and decision layer required for this workflow using simulated payment-system conditions.

---

## Why This Approach

Traditional payment retry logic can treat every failure similarly.

PayShield AI instead asks:

> **Why is this payment likely to fail?**

and then:

> **What should the payment system do about it?**

This enables a more context-aware approach:

```text
Payment Risk
     +
System Health
     +
Root Cause
     â†“
Recovery Decision
```

The objective is not only to predict failures but to turn payment-risk signals into actionable recovery decisions.

---

## Current Results

The current simulation contains:

| Metric                 |        Result |
| ---------------------- | ------------: |
| Transactions monitored |       100,000 |
| Payment success rate   |        97.19% |
| Average latency        | ~1.25 seconds |
| High-risk predictions  |           286 |

### Current Optimization Prioritization

| Priority         | Count |
| ---------------- | ----: |
| Critical Actions |    12 |
| High Priority    |   274 |
| Medium Priority  |     6 |
| No Intervention  | 9,897 |

These values are based on the current simulated/processed dataset and should not be interpreted as live payment-network statistics.

---

## Limitations

PayShield AI is currently a prototype and simulation environment.

It does **not** currently:

* Connect to live UPI infrastructure
* Connect to real bank accounts
* Intercept real payment authorization requests
* Execute real payment retries
* Automatically reroute real payment traffic
* Access live bank health APIs
* Access real merchant transaction data
* Guarantee payment success
* Automatically recover real revenue

The bank and gateway degradation scenarios are simulated to demonstrate the decision-making architecture.

The optimization engine currently provides recommendations rather than executing real payment-routing actions.

---

## Future Scope

Potential production extensions include:

### Real-Time Payment Monitoring

Integrate streaming transaction events and live payment-system telemetry.

### Real Bank and Gateway Health Signals

Connect to authorized operational health and payment-processing APIs.

### Explainable AI

Integrate SHAP or similar methods to explain why a payment was classified as high risk.

### Automated Recovery

Execute bounded recovery workflows such as:

```text
Retry
  â†“
Alternate Payment Route
  â†“
Alternate Method
  â†“
Escalation
```

with configurable stopping rules.

### Revenue Recovery Measurement

Track:

* Payments recovered
* Revenue recovered
* Recovery rate
* Intervention success
* Cost per intervention

### Adaptive Routing

Use historical intervention outcomes and controlled experiments to learn which routing action works best under different failure conditions.

### A/B Testing

Compare recovery strategies using controlled experiments.

### Online Learning

Continuously update risk predictions as payment behaviour changes.

### Alerts

Trigger operational alerts when:

* Failure rates increase
* Latency spikes
* Bank health deteriorates
* Gateway health deteriorates
* High-risk payment volume increases

---

## Technical Stack

```text
Python
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
Plotly
Jupyter Notebook
Git / GitHub
```

---

## Project Philosophy

PayShield AI is built around a simple principle:

> **Don't just predict payment failure. Understand why it is happening and recommend what should happen next.**

The long-term objective is to evolve the prototype from a payment-risk prediction system into an intelligent payment-recovery decision layer capable of detecting degradation, diagnosing root causes, selecting bounded interventions, and measuring recovered revenue.

---

## Disclaimer

PayShield AI is an experimental software prototype developed for demonstration and buildathon purposes.

All payment, bank-health, gateway-health, transaction, and failure scenarios used in the current implementation are simulated or derived from project-generated datasets.

The system should not be used to make real financial, banking, payment authorization, fraud, or routing decisions without appropriate validation, security controls, compliance review, and integration with authorized production systems.
