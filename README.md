# 🛡️ PayShield AI

### Autonomous Pre-Authorization Payment Protection & Revenue Recovery

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app/)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/Abhilasha-git/PayShield-AI)

> **Live Demo:** [abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app](https://abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app/)

> **Target Track:** Track 3: AI Revenue Recovery (Razorpay AI Builder Hackathon)

PayShield AI is an intelligent payment telemetry interceptor and decision-support layer designed to evaluate payment route risk **before PIN authorization**, helping prevent avoidable payment failuress, locked working capital, and customer drop-offs.

The system combines a 21-feature machine learning classifier with downstream banking health signals across three fault domains: **Sender Banks (CBS)**, **Receiver Banks (Acquirer Switches)**, and **Gateway Rails**.

> **Current implementation:** PayShield AI is a local simulation and decision-intelligence prototype. It does not connect to live UPI networks, bank accounts, payment gateways, or real payment authorization systems.

---

## 🎯 The Problem

In modern digital payment networks (such as UPI), transactions fail across three distinct fault domains:

* **Sender Bank (Issuer):** Core Banking System (CBS) timeouts or balance ledger outages.
* **Receiver Bank (Acquirer):** Inward settlement rail crashes or severe traffic congestion.
* **Payment Rails/Gateway:** Network timeout surges and API latency spikes.

Traditional payment apps treat failures identically: the user enters their PIN, waits 30 seconds, the payment fails silently, and funds are locked in 3–5 day reversal cycles. This leads to immediate cart abandonment and permanent revenue loss.

PayShield AI resolves this through a four-stage pre-authorization loop:

```text
DETECT
   |
   v
DIAGNOSE
   |
   v
DECIDE
   |
   v
RECOVER
```

The system detects abnormal telemetry behaviour, isolates the root cause of degradation, evaluates risk probability in under 100 ms, and triggers actionable recovery options before the user enters their PIN.

---

## 🏗️ Key Architecture & Workflow

```text
                    Payment Attempt (Initiated)
                               │
                               ▼
                   Collect Payment Signals
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
     Sender Bank         Receiver Bank          Gateway
     Health (CBS)        Health (Switch)      Health (Rails)
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                     Root-Cause Diagnosis
                               │
                               ▼
                  ML Risk Prediction (<100ms)
                               │
                               ▼
                     Action Interceptor / UI
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   Sender Recovery      Receiver Failover      User Sovereignty
  (Switch Account)      (Pay via Backup)      (Continue Anyway)
                               │
                               ▼
                         Measure & Recover
```

---

## 📌 Key Capabilities

### 1. Payment Risk Prediction (<100ms)

Evaluates payment failure probability using rolling failure rates, transaction latency, timeout behaviour, and historical degradation patterns within low-latency in-memory execution boundaries.

### 2. Tri-Party Health Monitoring & Diagnosis

Distinguishes between sender-bank, receiver-bank, and gateway-related degradation to prevent incorrect routing assumptions.

### 3. Pre-Authorization Risk Assessment

Screens payment route stability prior to PIN entry, displaying simulated risk probabilities and risk tiers:

* `LOW` (< 20%)
* `ELEVATED` (20% – < 50%)
* `MEDIUM` (50% – < 80%)
* `HIGH` (≥ 80%)

### 4. Consent-Driven Failover & Recovery

**Sender CBS Outage:** Prompts `🔄 Pay via Alternate Linked Account` to seamlessly select a secondary healthy bank.

**Receiver Acquirer Congestion:** Executes consent-driven failover via `⚡ Pay via Backup Account`, allowing payment completion without exposing raw banking rails or triggering phishing alerts.

**Fintech User Sovereignty:** Avoids non-consensual payment blocks by allowing voluntary execution through `Continue Anyway (High Failure Risk)`.

---

## ⚙️ Machine Learning Model

PayShield AI uses a `RandomForestClassifier` trained with balanced class weighting on rolling payment telemetry features.

### Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

### Features Monitored

* `transaction_count`
* `failure_rate`
* `timeout_rate`
* `avg_latency`
* `max_latency`
* `p95_latency`
* `bank_error_rate`
* `avg_amount`
* `max_amount`
* `hour`
* `day_of_week`
* `is_weekend`
* `previous_failure_rate`
* `previous_latency`
* `previous_timeout_rate`
* `failure_rate_change`
* `latency_change`
* `timeout_rate_change`
* `rolling_failure_rate`
* `rolling_latency`
* `rolling_timeout_rate`

### Holdout Evaluation

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
Actual Normal    10,129      4
Actual Risky        14     42
```

### Decision Tiers

| Risk Probability | Risk Level | Action Strategy                                       |
| ---------------: | ---------- | ----------------------------------------------------- |
|            < 20% | LOW        | Route Clear; standard PIN authorization               |
|      20% – < 50% | ELEVATED   | Monitor closely; reduce traffic exposure              |
|      50% – < 80% | MEDIUM     | Caution indicator; evaluate alternate routes          |
|            ≥ 80% | HIGH       | Intercept transaction; trigger smart failover options |

---

## 📂 Project Structure

```text
PayShield-AI/
│
├── .streamlit/
│   └── config.toml
│
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
│   ├── threshold_analysis.py
│   ├── train.py
│   ├── test_engine.py
│   ├── test_predict.py
│   └── test_real_data.py
│
├── src/
│   └── data_generation.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💻 Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Abhilasha-git/PayShield-AI.git
cd PayShield-AI
```

### 2. Create and Activate a Virtual Environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run app/dashboard.py
```

The Streamlit application will open in your default browser at:

```text
http://localhost:8501
```

---

## 🧪 Testing the Engine

Run the standalone validation scripts for the risk engine:

```bash
python -m risk_engine.test_engine
python -m risk_engine.test_predict
python -m risk_engine.test_real_data
```

These scripts validate:

* Model loading and deserialization
* Risk prediction probability output
* Threshold classification structure
* Feature vector schema verification

---

## 🎯 Buildathon Track Alignment

PayShield AI is built specifically for **Track 3: AI Revenue Recovery**.

The system focuses on actively intercepting payment failures and recovering at-risk revenue through smart routing rather than passive, post-failure alerting.

```text
Payment Degradation
        │
        ▼
Risk Detection
        │
        ▼
Root-Cause Diagnosis
        │
        ▼
Recovery Decision
```

Instead of simply predicting failures, PayShield turns pre-authorization risk signals into actionable recovery paths such as:

* ⚡ Pay via Backup Account
* 🔄 Pay via Alternate Linked Account

The objective is to prevent transaction abandonment and improve payment success.

---

## 📈 Current Simulation Baseline

| Metric                 |        Result |
| ---------------------- | ------------: |
| Transactions Monitored |       100,000 |
| Payment Success Rate   |        97.19% |
| Average Route Latency  | ~1.25 seconds |
| High-Risk Predictions  |           286 |

### Optimization Prioritization

| Priority Tier            | Count |
| ------------------------ | ----: |
| Critical Actions         |    12 |
| High Priority            |   274 |
| Medium Priority          |     6 |
| No Intervention Required | 9,897 |

> **Note:** Total high-risk predictions flagged by the model equals $12 + 274 = \mathbf{286}$ transactions.

*These values are based on the current simulated/processed dataset and should not be interpreted as live payment-network statistics.*

---

## ⚠️ Disclaimer

PayShield AI is an experimental software prototype developed for demonstration and hackathon evaluation purposes.

All payment, bank-health, gateway-health, transaction, and failure scenarios are simulated or derived from offline project-generated telemetry datasets.

The system is a decision-support prototype and does not connect directly to live UPI or NPCI payment switches.
