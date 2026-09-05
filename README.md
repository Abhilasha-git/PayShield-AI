# 🛡️ PayShield AI

## AI-Powered Payment Risk Prediction & Success Optimization

[![Landing Page](https://img.shields.io/badge/Landing_Page-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://payshield-ai-kappa.vercel.app/)
[![Streamlit App](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Codebase-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Abhilasha-git/PayShield-AI)


> **Landing Page (Vercel):** https://payshield-ai-kappa.vercel.app/

> **Live Demo:** https://abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app/

> **GitHub Repository:** [github.com/Abhilasha-git/PayShield-AI](https://github.com/Abhilasha-git/PayShield-AI)  

> **🎯 Target Track:** Track 3: AI Revenue Recovery (Razorpay AI Builder Hackathon)

---

## 📌 Overview

PayShield AI is an **AI-based payment intelligence system** designed to predict payment failure risk, identify payment-system anomalies, and recommend appropriate optimization actions.

The system combines payment transaction analytics, machine-learning risk scoring, operational diagnostics, pre-payment risk assessment, and AI-assisted optimization recommendations.

The project is designed as a **decision-support layer for payment reliability and success optimization**, using simulated and offline payment-system telemetry to demonstrate how machine learning can support earlier identification of potentially risky payment conditions.

> **Current implementation:** PayShield AI is a local simulation and decision-intelligence prototype. It does not connect to live UPI networks, bank accounts, payment gateways, or real payment authorization systems.

---

## ⚠️ Problem Statement

Payment failures can occur because of changing conditions across different parts of a payment ecosystem, including:

* Receiver-bank degradation
* Increased transaction failure rates
* Timeout spikes
* Gateway or network latency
* Bank error-rate increases
* Sudden changes from previous payment-system conditions

A simple transaction-success/failure view is often not enough to understand **when payment-system conditions are becoming risky**.

PayShield AI addresses this problem by using historical and rolling payment telemetry to:

1. Monitor payment-system conditions
2. Engineer meaningful risk features
3. Predict payment failure probability
4. Classify transactions into risk tiers
5. Diagnose major risk indicators
6. Generate AI-assisted optimization recommendations

---

# 🤖 AI / Machine Learning Component

The core AI component is a **Random Forest classification model** trained to estimate payment failure risk.

The model uses **21 engineered payment and operational features**, including:

* Transaction volume
* Failure rate
* Timeout rate
* Average latency
* Maximum latency
* P95 latency
* Bank error rate
* Transaction amount statistics
* Time-of-day information
* Previous payment-system conditions
* Failure-rate changes
* Latency changes
* Timeout-rate changes
* Rolling payment-system indicators

The model outputs a **risk probability**, which is then converted into a risk tier for monitoring and optimization decisions.

---

# 🔄 AI Prediction Pipeline

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
     +----+----+
     |         |
     v         v
Risk Drivers  Optimization
Diagnostics     Engine
                  |
                  v
        AI-Assisted Recommendation
```

---

# 🧠 Model Configuration

PayShield AI uses:

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

### Why Random Forest?

Random Forest was selected because it can model nonlinear relationships between payment-system indicators and failure risk while handling a mixture of operational and numerical features.

Balanced class weighting is used because the dataset contains significantly fewer risky observations than normal observations.

---

# 📊 Machine Learning Features

The model uses the following 21 features:

```text
transaction_count
failure_rate
timeout_rate
avg_latency
max_latency
p95_latency
bank_error_rate
avg_amount
max_amount
hour
day_of_week
is_weekend
previous_failure_rate
previous_latency
previous_timeout_rate
failure_rate_change
latency_change
timeout_rate_change
rolling_failure_rate
rolling_latency
rolling_timeout_rate
```

These features combine:

* Current payment-system behaviour
* Historical conditions
* Short-term changes
* Rolling operational indicators
* Transaction characteristics
* Time-based patterns

---

# 📈 Model Evaluation

The model was evaluated using a held-out test set.

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **99.82%** |
| Precision | **91.30%** |
| Recall    | **75.00%** |
| F1 Score  | **82.35%** |
| ROC-AUC   | **93.98%** |

### Holdout Confusion Matrix

```text
                  Predicted
                 Normal  Risky
Actual Normal     10129      4
Actual Risky         14     42
```

The evaluation shows strong discrimination between normal and risky observations, while also demonstrating that the model does not identify every risky case.

For this reason, PayShield AI uses multiple evaluation metrics rather than relying only on accuracy.

---

# 🎚️ Risk Classification

The model's probability output is converted into four operational risk tiers:

| Risk Probability | Risk Level |
| ---------------: | ---------- |
|          `< 20%` | LOW        |
|    `20% – < 50%` | ELEVATED   |
|    `50% – < 80%` | MEDIUM     |
|          `≥ 80%` | HIGH       |

A **50% threshold** is used for binary risky/non-risky classification based on threshold analysis and F1-score optimization.

The risk tiers are used for monitoring and recommendation purposes rather than automatically blocking or executing payments.

---

# 🔍 Risk Engine

The `risk_engine` module provides the prediction and evaluation layer.

```text
risk_engine/
├── __init__.py
├── engine.py
├── evaluate.py
├── predict.py
├── threshold_analysis.py
├── train.py
├── test_engine.py
├── test_predict.py
└── test_real_data.py
```

The engine is responsible for:

* Loading the trained model
* Preparing model inputs
* Generating risk probabilities
* Assigning risk levels
* Evaluating predictions
* Performing threshold analysis
* Validating prediction behaviour

---

# 🧪 Pre-Payment AI Risk Check

PayShield AI includes a **pre-payment risk simulation** that estimates payment failure risk before authorization using the latest available receiver-bank and payment-system conditions.

The feature is designed as a decision-support layer that can identify elevated payment risk before a payment is authorized and recommend an appropriate action.

The simulator allows the user to:

* Select a receiver bank
* Enter a transaction amount
* Select the payment hour
* Evaluate current payment-system conditions
* Run a controlled degraded-bank stress simulation
* Receive an AI risk probability
* Receive a risk tier
* Receive an AI-assisted recommended action

### Example validated scenarios

| Scenario                 | AI Risk | Risk Tier | Recommendation           |
| ------------------------ | ------: | --------- | ------------------------ |
| Healthy baseline         |   0.00% | LOW       | No intervention required |
| Degraded bank simulation |  84.50% | HIGH      | Route traffic away       |

The degraded-bank mode is a **controlled stress simulation** used to demonstrate how the risk engine responds to degraded payment-system conditions.

It does not represent live bank, UPI, or payment-gateway telemetry.

---

# 🤖 AI-Assisted Payment Success Optimization

PayShield AI converts risk predictions and payment-system indicators into **AI-assisted optimization recommendations**.

The current optimization layer is recommendation-based rather than an autonomous learned routing policy.

Example recommendations include:

### 🔄 Smart Rerouting

Recommend temporarily rerouting subsequent transactions away from high-latency receiver-bank endpoints.

### ⏱️ Adaptive Timeout Adjustments

Recommend extending transaction verification windows by 500 ms during detected latency surges.

### 🔔 Webhook Alerts

Recommend triggering webhook notifications to merchant integration systems for transactions flagged as `HIGH`.

These recommendations demonstrate how machine-learning risk predictions can be connected to operational payment optimization decisions.

---

# ⚙️ Optimization Decision Logic

```text
                 Risk Prediction
                       |
                       v
                Risk Diagnostics
                       |
                       v
             Payment-System Signals
                       |
                       v
            Optimization Engine
                       |
          +------------+-------------+
          |            |             |
          v            v             v
       Reroute       Monitor      Alert
```

Example decision mapping:

| Risk Condition               | AI-Assisted Recommendation     |
| ---------------------------- | ------------------------------ |
| HIGH + high failure rate     | Route traffic away             |
| HIGH + latency/timeout surge | Prefer alternate healthy route |
| MEDIUM                       | Monitor closely                |
| ELEVATED + high latency      | Reduce traffic exposure        |
| LOW                          | No intervention                |

---

# 🔎 Risk-Driver Diagnostics

PayShield AI also provides diagnostic insights for high-risk predictions.

Current analysis highlights:

### Latency Sensitivity

High-risk predictions correlate strongly with elevated gateway response latency.

### Bank Routing Clustering

Higher-risk observations cluster around specific downstream receiver-bank conditions in the simulated monitoring data.

### Failure Isolation

High-risk predictions represent a small share of total monitored volume, supporting targeted rather than broad optimization recommendations.

---

# 🏦 Bank Health Monitoring

The dashboard monitors receiver-bank conditions using operational indicators such as:

* Failure rate
* Timeout rate
* Average latency
* Bank health status

Example:

```text
Receiver Bank: AXIS
Health: NORMAL
Failure Rate: 0.00%
Timeout Rate: 0.00%
Average Latency: 1277 ms
```

The monitoring layer is based on the project's simulated/processed telemetry dataset.

---

# 📊 Dashboard & Web Deployment

PayShield AI provides an interactive operational dashboard along with a production landing page[cite: 5]:

* **Product Landing Page:** [https://payshield-ai-kappa.vercel.app/](https://payshield-ai-kappa.vercel.app/)
* **Interactive AI Dashboard:** [https://abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app/](https://abhilasha-git-payshield-ai-appdashboard-ffxsjf.streamlit.app/)[cite: 5]

The dashboard enables real-time decision intelligence across several core capabilities[cite: 5]:

* Payment-system monitoring[cite: 5]
* Payment success analysis[cite: 5]
* AI risk prediction[cite: 5]
* Risk distribution[cite: 5]
* High-risk diagnostics[cite: 5]
* Bank health monitoring[cite: 5]
* Pre-payment risk simulation[cite: 5]
* AI-assisted optimization recommendations[cite: 5]
* Risk probability visualization[cite: 5]

The dashboard is implemented using **Streamlit** and hosted on **Streamlit Cloud**, while the product interface is deployed on **Vercel**[cite: 4, 5].

---

# 🖥️ Dashboard Metrics

The current dashboard provides metrics including:

| Metric                 | Current Value |
| ---------------------- | ------------: |
| Payment Success Rate   |    **97.24%** |
| Average Latency        |   **1133 ms** |
| Transactions Monitored |   **100,000** |
| High-Risk Predictions  |        **56** |

### Current Risk Distribution

| Risk Level      |  Count |
| --------------- | -----: |
| HIGH ≥ 80%      |     56 |
| MEDIUM 50–80%   |      1 |
| ELEVATED 20–50% |      7 |
| LOW < 20%       | 10,125 |

---

# 🏗️ System Architecture

```text
                         PAYMENT / MONITORING DATA
                                   |
                                   v
                            DATA PROCESSING
                                   |
                                   v
                          FEATURE ENGINEERING
                                   |
                                   v
                              21 FEATURES
                                   |
                                   v
                       +-------------------------+
                       | Random Forest Classifier|
                       |       AI MODEL          |
                       +-------------------------+
                                   |
                                   v
                           RISK PROBABILITY
                                   |
                                   v
                              RISK TIER
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
              RISK DIAGNOSTICS          OPTIMIZATION ENGINE
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
                         Risk Engine
                              |
                              v
                    Risk + Recommendation
```

---

# 📁 Project Structure

```text
PayShield-AI/
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

# 💻 Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Abhilasha-git/PayShield-AI.git
cd PayShield-AI
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

# 🧪 Testing

The project includes standalone validation scripts for the risk engine.

```bash
python -m risk_engine.test_engine
python -m risk_engine.test_predict
python -m risk_engine.test_real_data
```

These tests validate:

* Model loading
* Model prediction
* Risk probability generation
* Risk classification
* Feature-vector compatibility
* Prediction behaviour on project data

---

# 🧰 Technology Stack

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Core development               |
| Pandas           | Data processing                |
| NumPy            | Numerical computation          |
| Scikit-learn     | Machine learning               |
| Joblib           | Model serialization            |
| Streamlit        | Interactive dashboard          |
| Plotly           | Data visualization             |
| Jupyter Notebook | Model development and analysis |

---

# 🧠 AI Architecture Summary

PayShield AI currently consists of four major intelligence layers:

### 1. Prediction Layer

Machine-learning model predicts payment failure probability.

### 2. Risk Classification Layer

Probability outputs are converted into operational risk tiers.

### 3. Diagnostic Layer

Payment-system indicators are analysed to identify important risk drivers.

### 4. Optimization Recommendation Layer

Risk predictions and operational indicators are converted into AI-assisted recommendations.

```text
             AI PAYMENT INTELLIGENCE
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
   Prediction     Diagnostics    Optimization
        |              |              |
        +--------------+--------------+
                       |
                       v
                Decision Support
```

---

# ⚠️ Important AI Limitation

The current optimization engine is an **AI-assisted recommendation layer**, not a fully learned autonomous optimization policy.

The project currently does not contain enough historical intervention-outcome data or counterfactual routing experiments to train a model that directly learns:

```text
Payment Condition
        ↓
Intervention
        ↓
Observed Success Improvement
```

Therefore, optimization actions are currently generated using model risk predictions and predefined operational decision logic.

A future version could learn the expected impact of different interventions using historical routing outcomes, controlled experiments, and counterfactual evaluation.

---

# 🚀 Future Scope

Potential future improvements include:

* Live payment telemetry integration
* Real-time payment gateway monitoring
* Online model inference
* Model drift detection
* Automated feature monitoring
* Learned routing optimization
* Counterfactual intervention modelling
* A/B testing of optimization strategies
* Reinforcement-learning-based routing policies
* Merchant-specific risk models
* Explainable AI for individual predictions
* Real-time alerting and webhook integration
* Production-grade payment infrastructure integration

---

# 📌 Current Project Status

PayShield AI is currently a **working AI/ML prototype** demonstrating:

* Payment failure risk prediction
* Machine-learning-based risk scoring
* Risk classification
* Payment-system monitoring
* Risk-driver diagnostics
* Pre-payment risk simulation
* AI-assisted optimization recommendations
* Interactive payment-risk dashboard

The project uses simulated and offline-generated payment-system telemetry for experimentation and demonstration.

---

# ⚠️ Disclaimer

PayShield AI is an experimental software prototype developed for demonstration, learning, and project evaluation purposes.

All payment, bank-health, gateway-health, transaction, and failure scenarios are simulated or derived from offline project-generated telemetry datasets.

The system does **not** connect directly to live UPI or NPCI payment switches, bank accounts, payment gateways, or real payment authorization systems.

The recommendations generated by the system are decision-support suggestions and should not be interpreted as autonomous payment execution or production payment-routing decisions.

---
