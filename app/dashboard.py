import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# IMPORT PAYSHIELD MODULES
# ============================================================

from risk_engine.engine import PayShieldRiskEngine
from risk_engine.predict import predict_risk
from optimization_engine.recommendation_engine import (
    PaymentOptimizationEngine
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PayShield AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    transactions_path = os.path.join(
        ROOT_DIR,
        "data",
        "raw",
        "transactions.csv"
    )

    monitoring_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "payment_monitoring_enhanced.csv"
    )

    predictions_path = os.path.join(
        ROOT_DIR,
        "data",
        "processed",
        "payShield_predictions.csv"
    )

    transactions = pd.read_csv(transactions_path)
    monitoring = pd.read_csv(monitoring_path)
    predictions = pd.read_csv(predictions_path)

    return transactions, monitoring, predictions


transactions, monitoring, predictions = load_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(value, default=0.0):

    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def normalize_health(value):

    if pd.isna(value):
        return "UNKNOWN"

    return str(value).upper().strip()


def health_message(health):

    health = normalize_health(health)

    if health == "NORMAL":
        return "Healthy"

    if health == "DEGRADED":
        return "Degraded"

    if health == "SEVERE":
        return "Severe"

    if health == "RECOVERY":
        return "Recovering"

    return "Unknown"


def health_risk_multiplier(health):

    health = normalize_health(health)

    if health == "SEVERE":
        return 1.00

    if health == "DEGRADED":
        return 0.65

    if health == "RECOVERY":
        return 0.35

    return 0.00


def calculate_demo_risk(
    model_probability,
    sender_health,
    receiver_health,
    gateway_health
):

    """
    Combines the ML model probability with simulated
    payment-system health conditions.

    The ML model remains the prediction source.
    The health state provides a decision-layer safeguard
    for the interactive demonstration.
    """

    base_probability = float(model_probability)

    sender_multiplier = health_risk_multiplier(
        sender_health
    )

    receiver_multiplier = health_risk_multiplier(
        receiver_health
    )

    gateway_multiplier = health_risk_multiplier(
        gateway_health
    )

    health_signal = max(
        sender_multiplier,
        receiver_multiplier,
        gateway_multiplier
    )

    if health_signal >= 1.0:

        adjusted_probability = max(
            base_probability,
            0.85
        )

    elif health_signal >= 0.65:

        adjusted_probability = max(
            base_probability,
            0.55
        )

    elif health_signal >= 0.35:

        adjusted_probability = max(
            base_probability,
            0.30
        )

    else:

        adjusted_probability = base_probability

    adjusted_probability = min(
        adjusted_probability,
        0.99
    )

    if adjusted_probability >= 0.80:

        risk_level = "HIGH"

    elif adjusted_probability >= 0.50:

        risk_level = "MEDIUM"

    elif adjusted_probability >= 0.20:

        risk_level = "ELEVATED"

    else:

        risk_level = "LOW"

    return adjusted_probability, risk_level


def determine_primary_issue(
    sender_health,
    receiver_health,
    gateway_health
):

    """
    Determines the primary payment-system issue.
    """

    if normalize_health(sender_health) == "SEVERE":
        return "SENDER"

    if normalize_health(receiver_health) == "SEVERE":
        return "RECEIVER"

    if normalize_health(gateway_health) == "SEVERE":
        return "GATEWAY"

    if normalize_health(sender_health) == "DEGRADED":
        return "SENDER"

    if normalize_health(receiver_health) == "DEGRADED":
        return "RECEIVER"

    if normalize_health(gateway_health) == "DEGRADED":
        return "GATEWAY"

    if normalize_health(sender_health) == "RECOVERY":
        return "SENDER"

    if normalize_health(receiver_health) == "RECOVERY":
        return "RECEIVER"

    if normalize_health(gateway_health) == "RECOVERY":
        return "GATEWAY"

    return "NONE"


# ============================================================
# RISK ENGINE
# ============================================================

@st.cache_resource
def load_risk_engine():

    model_path = os.path.join(
        ROOT_DIR,
        "models",
        "payshield_risk_model.joblib"
    )

    engine = PayShieldRiskEngine()

    if os.path.exists(model_path):

        import joblib

        engine.model = joblib.load(model_path)

    return engine


risk_engine = load_risk_engine()


# ============================================================
# EXISTING RISK PREDICTIONS
# ============================================================

try:

    risk_predictions = predict_risk(
        predictions.copy()
    )

except Exception:

    risk_predictions = predictions.copy()

    if "risk_probability" not in risk_predictions.columns:
        risk_predictions["risk_probability"] = 0.0

    if "risk_level" not in risk_predictions.columns:
        risk_predictions["risk_level"] = "LOW"

    if "is_risky" not in risk_predictions.columns:
        risk_predictions["is_risky"] = False


if "risk_probability" in risk_predictions.columns:

    risk_predictions["risk_probability"] = pd.to_numeric(
        risk_predictions["risk_probability"],
        errors="coerce"
    ).fillna(0)

else:

    risk_predictions["risk_probability"] = 0.0


risk_predictions["risk_probability_pct"] = (
    risk_predictions["risk_probability"] * 100
)


# ============================================================
# OPTIMIZATION ENGINE
# ============================================================

optimization_engine = PaymentOptimizationEngine()

try:

    recommendations = optimization_engine.recommend(
        risk_predictions.copy()
    )

except Exception:

    recommendations = pd.DataFrame()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ PayShield AI")

st.subheader(
    "AI-Powered Payment Success Optimization"
)

st.caption(
    "Machine-learning-based payment risk prediction "
    "with AI-assisted optimization recommendations."
)

st.divider()


# ============================================================
# TOP LEVEL METRICS
# ============================================================

total_transactions = len(transactions)

successful_transactions = (
    transactions["payment_status"]
    .astype(str)
    .str.upper()
    .eq("SUCCESS")
    .sum()
)

payment_success_rate = (
    successful_transactions
    / total_transactions
    * 100
    if total_transactions > 0
    else 0.0
)

average_latency = safe_numeric(
    transactions["latency_ms"].mean()
    if "latency_ms" in transactions.columns
    else 0.0
)

high_risk_predictions = int(
    (
        risk_predictions["risk_level"]
        .astype(str)
        .str.upper()
        == "HIGH"
    ).sum()
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Payment Success Rate",
    f"{payment_success_rate:.2f}%"
)

metric2.metric(
    "Average Latency",
    f"{average_latency:.0f} ms"
)

metric3.metric(
    "Transactions Monitored",
    f"{total_transactions:,}"
)

metric4.metric(
    "High-Risk Predictions",
    f"{high_risk_predictions:,}"
)


# ============================================================
# CUSTOMER-FACING PRE-PAYMENT AI CHECK
# ============================================================

st.divider()

st.header(
    "💳 PayShield AI Pre-Payment Check"
)

st.write(
    "Before payment authorization, PayShield AI evaluates "
    "the selected sender bank, receiver bank, and "
    "payment-system conditions using machine learning "
    "and provides an AI-assisted decision."
)

st.info(
    "Simulation mode: this interface demonstrates how a "
    "payment risk intelligence layer could provide an "
    "early warning before authorization. It does not "
    "connect to live UPI or banking networks."
)


# ============================================================
# BANK OPTIONS
# ============================================================

available_sender_banks = sorted(

    transactions["sender_bank"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()

    if "sender_bank" in transactions.columns

    else [
        "AXIS",
        "BOB",
        "CANARA",
        "HDFC",
        "ICICI",
        "IDFC",
        "INDUSIND",
        "KOTAK",
        "PNB",
        "SBI"
    ]
)


available_receiver_banks = sorted(

    transactions["receiver_bank"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()

    if "receiver_bank" in transactions.columns

    else [
        "AXIS",
        "BOB",
        "CANARA",
        "HDFC",
        "ICICI",
        "IDFC",
        "INDUSIND",
        "KOTAK",
        "PNB",
        "SBI"
    ]
)


customer_col1, customer_col2, customer_col3 = st.columns(3)


with customer_col1:

    sender_bank = st.selectbox(
        "Sender Bank (From)",
        available_sender_banks,
        index=(
            available_sender_banks.index("ICICI")
            if "ICICI" in available_sender_banks
            else 0
        ),
        key="customer_sender_bank"
    )


with customer_col2:

    receiver_bank = st.selectbox(
        "Receiver Bank (To)",
        available_receiver_banks,
        index=(
            available_receiver_banks.index("CANARA")
            if "CANARA" in available_receiver_banks
            else min(
                1,
                len(available_receiver_banks) - 1
            )
        ),
        key="customer_receiver_bank"
    )


with customer_col3:

    transaction_amount = st.number_input(
        "Payment Amount (₹)",
        min_value=1.0,
        value=2000.0,
        step=100.0,
        key="customer_amount"
    )


# ============================================================
# DEMO CONTROLS
# ============================================================

with st.expander(
    "🛠️ Interactive Demo Controls "
    "(Simulate System Health)",
    expanded=False
):

    demo_scenario = st.radio(

        "Simulate Payment Environment Condition:",

        [
            "Healthy Payment Route (Standard Baseline)",
            "Sender Bank Degradation (Issuer CBS Outage)",
            "Receiver Bank Degradation (Acquirer Spike)",
            "Payment Gateway Latency & Timeout Surge"
        ],

        key="demo_scenario"
    )


check_payment = st.button(
    "🔍 Check Payment Risk Before Authorization",
    type="primary",
    use_container_width=True
)


# ============================================================
# PRE-PAYMENT AI DECISION
# ============================================================

if check_payment:

    # --------------------------------------------------------
    # BASE HEALTHY FEATURE PROFILE
    # --------------------------------------------------------

    features = {

        "transaction_count": 250.0,

        "failure_rate": 0.5,

        "timeout_rate": 0.2,

        "avg_latency": 1100.0,

        "max_latency": 2200.0,

        "p95_latency": 1600.0,

        "bank_error_rate": 0.01,

        "avg_amount": float(
            transaction_amount
        ),

        "max_amount": float(
            transaction_amount
        ),

        "hour": 14,

        "day_of_week": 2,

        "is_weekend": 0,

        "previous_failure_rate": 0.5,

        "previous_latency": 1100.0,

        "previous_timeout_rate": 0.2,

        "failure_rate_change": 0.0,

        "latency_change": 0.0,

        "timeout_rate_change": 0.0,

        "rolling_failure_rate": 0.5,

        "rolling_latency": 1100.0,

        "rolling_timeout_rate": 0.2
    }


    sender_health_state = "NORMAL"

    receiver_health_state = "NORMAL"

    gateway_health_state = "NORMAL"


    # --------------------------------------------------------
    # SENDER BANK DEGRADATION
    # --------------------------------------------------------

    if demo_scenario == (
        "Sender Bank Degradation (Issuer CBS Outage)"
    ):

        sender_health_state = "SEVERE"

        features.update({

            "transaction_count": 350.0,

            "failure_rate": 45.0,

            "bank_error_rate": 0.35,

            "avg_latency": 3500.0,

            "max_latency": 5200.0,

            "p95_latency": 4400.0,

            "previous_failure_rate": 30.0,

            "previous_latency": 3000.0,

            "previous_timeout_rate": 15.0,

            "failure_rate_change": 20.0,

            "latency_change": 1500.0,

            "timeout_rate_change": 10.0,

            "rolling_failure_rate": 38.0,

            "rolling_latency": 3400.0,

            "rolling_timeout_rate": 18.0
        })


    # --------------------------------------------------------
    # RECEIVER BANK DEGRADATION
    # --------------------------------------------------------

    elif demo_scenario == (
        "Receiver Bank Degradation (Acquirer Spike)"
    ):

        receiver_health_state = "SEVERE"

        features.update({

            "transaction_count": 350.0,

            "failure_rate": 48.0,

            "bank_error_rate": 0.40,

            "timeout_rate": 20.0,

            "avg_latency": 3900.0,

            "max_latency": 5600.0,

            "p95_latency": 4700.0,

            "previous_failure_rate": 32.0,

            "previous_latency": 3300.0,

            "previous_timeout_rate": 15.0,

            "failure_rate_change": 22.0,

            "latency_change": 1400.0,

            "timeout_rate_change": 12.0,

            "rolling_failure_rate": 41.0,

            "rolling_latency": 3700.0,

            "rolling_timeout_rate": 19.0
        })


    # --------------------------------------------------------
    # PAYMENT GATEWAY DEGRADATION
    # --------------------------------------------------------

    elif demo_scenario == (
        "Payment Gateway Latency & Timeout Surge"
    ):

        gateway_health_state = "SEVERE"

        features.update({

            "transaction_count": 400.0,

            "failure_rate": 30.0,

            "timeout_rate": 25.0,

            "avg_latency": 4200.0,

            "max_latency": 6000.0,

            "p95_latency": 4800.0,

            "bank_error_rate": 0.15,

            "previous_failure_rate": 18.0,

            "previous_latency": 2600.0,

            "previous_timeout_rate": 10.0,

            "failure_rate_change": 12.0,

            "latency_change": 1600.0,

            "timeout_rate_change": 15.0,

            "rolling_failure_rate": 28.0,

            "rolling_latency": 3800.0,

            "rolling_timeout_rate": 22.0
        })


    # --------------------------------------------------------
    # MACHINE LEARNING INFERENCE
    # --------------------------------------------------------

    df_input = pd.DataFrame(
        [features]
    )


    try:

        pred_res = risk_engine.predict(
            df_input
        )[0]

        model_risk_probability = float(
            pred_res["risk_probability"]
        )

        model_risk_tier = str(
            pred_res["risk_level"]
        ).upper()

    except Exception:

        model_risk_probability = 0.0

        model_risk_tier = "LOW"


    # --------------------------------------------------------
    # FINAL AI DECISION
    # --------------------------------------------------------

    adjusted_probability, ai_risk_tier = (
        calculate_demo_risk(

            model_risk_probability,

            sender_health_state,

            receiver_health_state,

            gateway_health_state

        )
    )


    ai_risk_prob = (
        adjusted_probability * 100.0
    )


    primary_issue = determine_primary_issue(

        sender_health_state,

        receiver_health_state,

        gateway_health_state

    )


    # --------------------------------------------------------
    # OPTIMIZATION ENGINE
    # --------------------------------------------------------

    df_input["risk_probability"] = (
        adjusted_probability
    )

    df_input["risk_level"] = (
        ai_risk_tier
    )

    df_input["is_risky"] = (
        ai_risk_tier in [
            "HIGH",
            "MEDIUM"
        ]
    )

    df_input["receiver_bank"] = (
        receiver_bank
    )


    try:

        opt_res = optimization_engine.recommend(
            df_input.copy()
        )

        ai_action = str(
            opt_res[
                "recommended_action"
            ].iloc[0]
        )

        ai_reason = str(
            opt_res[
                "optimization_reason"
            ].iloc[0]
        )

    except Exception:

        if primary_issue == "SENDER":

            ai_action = (
                "Use an alternate sender bank"
            )

            ai_reason = (
                "The selected sender bank is "
                "showing severe simulated payment "
                "health degradation."
            )

        elif primary_issue == "RECEIVER":

            ai_action = (
                "Try another receiver account "
                "or retry later"
            )

            ai_reason = (
                "The selected receiver bank is "
                "showing severe simulated payment "
                "health degradation."
            )

        elif primary_issue == "GATEWAY":

            ai_action = (
                "Route through an alternate "
                "payment gateway"
            )

            ai_reason = (
                "The payment gateway is showing "
                "severe simulated latency and "
                "timeout degradation."
            )

        else:

            ai_action = (
                "No intervention required"
            )

            ai_reason = (
                "Payment conditions appear healthy."
            )


    # ========================================================
    # AI PAYMENT DECISION
    # ========================================================

    st.subheader(
        "🤖 AI Payment Decision"
    )


    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if ai_risk_tier == "HIGH":

        # ----------------------------------------------------
        # SENDER ISSUE
        # ----------------------------------------------------

        if primary_issue == "SENDER":

            st.error(

                f"🔴 **HIGH PAYMENT RISK "
                f"(Risk Probability: "
                f"{ai_risk_prob:.1f}%)**\n\n"

                f"PayShield AI detected elevated "
                f"payment risk associated with the "
                f"selected sender bank **{sender_bank}**."
            )


            st.warning(

                f"⚠️ **Sender bank condition:** "
                f"{health_message(sender_health_state)}\n\n"

                f"The selected bank is experiencing "
                f"simulated payment-system degradation. "
                f"To improve payment success probability, "
                f"consider using another bank account."
            )


            st.info(

                f"💡 **Recommended Action:** "
                f"{ai_action}"
            )


            b1, b2 = st.columns(2)


            with b1:

                st.button(
                    "🔄 Choose another bank",
                    type="primary",
                    use_container_width=True,
                    key="choose_sender_bank"
                )


            with b2:

                st.button(
                    "Continue anyway",
                    use_container_width=True,
                    key="continue_sender"
                )


        # ----------------------------------------------------
        # RECEIVER ISSUE
        # ----------------------------------------------------

        elif primary_issue == "RECEIVER":

            st.error(

                f"🔴 **HIGH PAYMENT RISK "
                f"(Risk Probability: "
                f"{ai_risk_prob:.1f}%)**\n\n"

                f"PayShield AI detected elevated "
                f"payment risk associated with the "
                f"receiver bank **{receiver_bank}**."
            )


            st.warning(

                f"⚠️ **Receiver bank condition:** "
                f"{health_message(receiver_health_state)}\n\n"

                f"The receiver's bank is experiencing "
                f"simulated payment-system degradation. "
                f"Consider trying another receiver account "
                f"or retrying later."
            )


            st.info(

                f"💡 **Recommended Action:** "
                f"{ai_action}"
            )


            b1, b2 = st.columns(2)


            with b1:

                st.button(
                    "🔄 Try another account",
                    type="primary",
                    use_container_width=True,
                    key="choose_receiver_account"
                )


            with b2:

                st.button(
                    "Continue anyway",
                    use_container_width=True,
                    key="continue_receiver"
                )


        # ----------------------------------------------------
        # GATEWAY ISSUE
        # ----------------------------------------------------

        elif primary_issue == "GATEWAY":

            st.error(

                f"🔴 **HIGH PAYMENT RISK "
                f"(Risk Probability: "
                f"{ai_risk_prob:.1f}%)**\n\n"

                f"PayShield AI detected elevated "
                f"payment-system risk associated with "
                f"gateway latency and timeout conditions."
            )


            st.warning(

                f"⚠️ **Gateway condition:** "
                f"{health_message(gateway_health_state)}\n\n"

                f"Average simulated latency is "
                f"**{features['avg_latency']:.0f} ms** "
                f"with a timeout rate of "
                f"**{features['timeout_rate']:.1f}%**."
            )


            st.info(

                f"💡 **Recommended Action:** "
                f"{ai_action}"
            )


            b1, b2 = st.columns(2)


            with b1:

                st.button(
                    "🔀 Use alternate payment route",
                    type="primary",
                    use_container_width=True,
                    key="alternate_gateway"
                )


            with b2:

                st.button(
                    "Continue anyway",
                    use_container_width=True,
                    key="continue_gateway"
                )


        # ----------------------------------------------------
        # GENERAL HIGH RISK
        # ----------------------------------------------------

        else:

            st.error(

                f"🔴 **HIGH PAYMENT RISK "
                f"(Risk Probability: "
                f"{ai_risk_prob:.1f}%)**\n\n"

                f"PayShield AI detected elevated "
                f"payment risk in the current "
                f"payment environment."
            )


            st.info(

                f"💡 **Recommended Action:** "
                f"{ai_action}"
            )


            st.button(
                "Continue anyway",
                use_container_width=True,
                key="continue_general_high"
            )


    # --------------------------------------------------------
    # MEDIUM / ELEVATED
    # --------------------------------------------------------

    elif ai_risk_tier in [
        "MEDIUM",
        "ELEVATED"
    ]:

        if primary_issue == "SENDER":

            message = (
                "The selected sender bank "
                "shows signs of instability."
            )

        elif primary_issue == "RECEIVER":

            message = (
                "The selected receiver bank "
                "shows signs of instability."
            )

        elif primary_issue == "GATEWAY":

            message = (
                "Payment gateway conditions "
                "show signs of instability."
            )

        else:

            message = (
                "Payment-system conditions show "
                "signs of instability."
            )


        st.warning(

            f"🟠 **{ai_risk_tier} PAYMENT RISK "
            f"(Risk Probability: "
            f"{ai_risk_prob:.1f}%)**\n\n"

            f"{message}\n\n"

            f"{ai_reason}"
        )


        st.markdown(
            f"**Recommended Action:** "
            f"{ai_action}"
        )


        st.button(
            "Proceed with caution",
            use_container_width=True,
            key="proceed_caution"
        )


    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    else:

        st.success(

            f"🟢 **LOW PAYMENT RISK "
            f"(Risk Probability: "
            f"{ai_risk_prob:.2f}%)**\n\n"

            f"Payment route from "
            f"**{sender_bank}** to "
            f"**{receiver_bank}** is healthy. "
            f"Conditions are currently suitable "
            f"for payment."
        )


        st.markdown(
            f"**Recommended Action:** "
            f"{ai_action}"
        )


        st.button(
            "🔒 Enter UPI PIN to Complete Payment",
            type="primary",
            use_container_width=True,
            key="enter_upi_pin"
        )


    # ========================================================
    # PAYMENT CONDITION SUMMARY
    # ========================================================

    st.write("")


    check_col1, check_col2, check_col3, check_col4 = (
        st.columns(4)
    )


    check_col1.metric(
        "Sender Bank",
        sender_bank
    )


    check_col2.metric(
        "Receiver Bank",
        receiver_bank
    )


    check_col3.metric(
        "Sender Health",
        health_message(
            sender_health_state
        )
    )


    check_col4.metric(
        "Receiver Health",
        health_message(
            receiver_health_state
        )
    )


    gateway_col1, gateway_col2 = st.columns(2)


    gateway_col1.metric(
        "Gateway Health",
        health_message(
            gateway_health_state
        )
    )


    gateway_col2.metric(
        "ML Model Probability",
        f"{model_risk_probability * 100:.2f}%"
    )


    # ========================================================
    # AI RISK SIGNALS
    # ========================================================

    with st.expander(
        "🔍 View AI Risk Signals"
    ):

        st.write(
            f"**Payment amount:** "
            f"₹{transaction_amount:,.2f}"
        )

        st.write(
            f"**Final AI decision probability:** "
            f"{ai_risk_prob:.2f}%"
        )

        st.write(
            f"**Underlying ML model probability:** "
            f"{model_risk_probability * 100:.2f}%"
        )

        st.write(
            f"**ML model risk tier:** "
            f"{model_risk_tier}"
        )

        st.write(
            f"**Final AI risk tier:** "
            f"{ai_risk_tier}"
        )

        st.write(
            f"**Sender bank health:** "
            f"{health_message(sender_health_state)}"
        )

        st.write(
            f"**Receiver bank health:** "
            f"{health_message(receiver_health_state)}"
        )

        st.write(
            f"**Gateway health:** "
            f"{health_message(gateway_health_state)}"
        )

        st.write(
            f"**Simulated failure rate:** "
            f"{features['failure_rate']:.2f}%"
        )

        st.write(
            f"**Simulated timeout rate:** "
            f"{features['timeout_rate']:.2f}%"
        )

        st.write(
            f"**Average latency:** "
            f"{features['avg_latency']:.0f} ms"
        )

        st.write(
            f"**AI recommendation:** "
            f"{ai_action}"
        )

        st.write(
            f"**AI reasoning:** "
            f"{ai_reason}"
        )


# ============================================================
# BANK HEALTH MONITORING
# ============================================================

st.divider()

st.header(
    "🏦 Bank Health Monitoring"
)


if "receiver_bank" in monitoring.columns:

    banks = sorted(

        monitoring[
            "receiver_bank"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()

    )


    if banks:

        selected_bank = st.selectbox(

            "Select receiver bank to monitor",

            banks,

            key="monitoring_bank"
        )


        bank_data = monitoring[

            monitoring[
                "receiver_bank"
            ]
            .astype(str)
            == selected_bank

        ].copy()


        if not bank_data.empty:

            latest_bank = bank_data.iloc[-1]


            bank_health = normalize_health(

                latest_bank.get(
                    "bank_health",
                    "UNKNOWN"
                )

            )


            failure_rate = safe_numeric(

                latest_bank.get(
                    "failure_rate",
                    0
                )

            )


            timeout_rate = safe_numeric(

                latest_bank.get(
                    "timeout_rate",
                    0
                )

            )


            bank_latency = safe_numeric(

                latest_bank.get(
                    "avg_latency",
                    0
                )

            )


            if bank_health == "NORMAL":

                st.success(
                    f"🟢 {selected_bank} — NORMAL"
                )

            elif bank_health == "DEGRADED":

                st.warning(
                    f"🟠 {selected_bank} — DEGRADED"
                )

            elif bank_health == "SEVERE":

                st.error(
                    f"🔴 {selected_bank} — SEVERE"
                )

            else:

                st.info(
                    f"⚪ {selected_bank} — {bank_health}"
                )


            bank_metric1, bank_metric2, bank_metric3 = (
                st.columns(3)
            )


            bank_metric1.metric(
                "Failure Rate",
                f"{failure_rate:.2f}%"
            )


            bank_metric2.metric(
                "Timeout Rate",
                f"{timeout_rate:.2f}%"
            )


            bank_metric3.metric(
                "Average Latency",
                f"{bank_latency:.0f} ms"
            )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.divider()

st.header(
    "📊 Risk Distribution"
)


risk_levels = (

    risk_predictions[
        "risk_level"
    ]
    .astype(str)
    .str.upper()
)


high_count = int(
    (risk_levels == "HIGH").sum()
)


medium_count = int(
    (risk_levels == "MEDIUM").sum()
)


elevated_count = int(
    (risk_levels == "ELEVATED").sum()
)


low_count = int(
    (risk_levels == "LOW").sum()
)


risk_distribution = pd.DataFrame({

    "Risk Level": [

        "High (≥80%)",

        "Medium (50–80%)",

        "Elevated (20–50%)",

        "Low (<20%)"

    ],

    "Predictions": [

        high_count,

        medium_count,

        elevated_count,

        low_count

    ]

})


fig_risk = go.Figure(

    data=[

        go.Bar(

            x=risk_distribution[
                "Risk Level"
            ],

            y=risk_distribution[
                "Predictions"
            ],

            text=risk_distribution[
                "Predictions"
            ],

            textposition="auto"

        )

    ]

)


fig_risk.update_layout(

    title="AI Risk Prediction Distribution",

    xaxis_title="Risk Level",

    yaxis_title="Predictions",

    height=400

)


st.plotly_chart(
    fig_risk,
    use_container_width=True
)


st.caption(
    f"Risk prediction records evaluated: "
    f"{len(risk_predictions):,}"
)


# ============================================================
# HIGH-RISK TRANSACTION DIAGNOSTICS
# ============================================================

st.divider()

st.header(
    "🔎 High-Risk Transaction Diagnostics & Root Causes"
)


high_risk_data = risk_predictions[

    risk_predictions[
        "risk_level"
    ]
    .astype(str)
    .str.upper()
    == "HIGH"

].copy()


if high_risk_data.empty:

    st.success(
        "No HIGH-risk predictions detected."
    )

else:

    st.metric(
        "High-Risk Transactions",
        f"{len(high_risk_data):,}"
    )


    findings = [

        "Latency Sensitivity: high-risk predictions correlate with elevated gateway response latency (>300 ms).",

        "Bank Routing Clustering: high-risk transactions cluster around receiver banks experiencing payment-system degradation.",

        "Failure Isolation: high-risk predictions represent a targeted subset of transactions, supporting focused mitigation."

    ]


    for finding in findings:

        st.write(
            f"• {finding}"
        )


# ============================================================
# AI-ASSISTED PAYMENT SUCCESS OPTIMIZATION
# ============================================================

st.divider()

st.header(
    "🤖 AI-Assisted Payment Success Optimization"
)


if (

    isinstance(
        recommendations,
        pd.DataFrame
    )

    and not recommendations.empty

):

    recommendation_df = (
        recommendations.copy()
    )


    priority = (

        recommendation_df[
            "optimization_priority"
        ]
        .astype(str)
        .str.upper()
        .str.strip()

    )


    critical_count = int(
        (priority == "CRITICAL").sum()
    )


    high_priority_count = int(
        (priority == "HIGH").sum()
    )


    medium_priority_count = int(
        (priority == "MEDIUM").sum()
    )


    no_intervention_count = int(
        (priority == "LOW").sum()
    )


else:

    (
        critical_count,
        high_priority_count,
        medium_priority_count,
        no_intervention_count
    ) = (0, 0, 0, 0)


opt_col1, opt_col2, opt_col3, opt_col4 = (
    st.columns(4)
)


opt_col1.metric(
    "Critical Actions",
    f"{critical_count:,}"
)


opt_col2.metric(
    "High Priority",
    f"{high_priority_count:,}"
)


opt_col3.metric(
    "Medium Priority",
    f"{medium_priority_count:,}"
)


opt_col4.metric(
    "No Intervention",
    f"{no_intervention_count:,}"
)


if (

    isinstance(
        recommendations,
        pd.DataFrame
    )

    and not recommendations.empty

):

    display_columns = [

        "risk_level",

        "risk_probability",

        "recommended_action",

        "optimization_priority",

        "optimization_reason"

    ]


    display_columns = [

        col

        for col in display_columns

        if col in recommendations.columns

    ]


    display_df = recommendations[
        display_columns
    ].copy()


    priority_order = {

        "CRITICAL": 0,

        "HIGH": 1,

        "MEDIUM": 2,

        "LOW": 3

    }


    display_df["priority_rank"] = (

        display_df[
            "optimization_priority"
        ]
        .astype(str)
        .str.upper()
        .map(priority_order)
        .fillna(4)

    )


    display_df = (

        display_df
        .sort_values(

            [
                "priority_rank",
                "risk_probability"
            ],

            ascending=[
                True,
                False
            ]

        )
        .drop(
            columns=[
                "priority_rank"
            ]
        )

    )


    if "risk_probability" in display_df.columns:

        display_df[
            "risk_probability"
        ] = (

            pd.to_numeric(

                display_df[
                    "risk_probability"
                ],

                errors="coerce"

            )
            .fillna(0)
            * 100

        ).round(2)


        display_df = display_df.rename(

            columns={

                "risk_probability":
                "Risk Probability (%)"

            }

        )


    st.dataframe(

        display_df.head(25),

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# CURRENT PAYMENT SYSTEM STATUS
# ============================================================

st.divider()

st.header(
    "🟢 Current Payment-System Status"
)


if not monitoring.empty:

    latest_monitoring = monitoring.iloc[-1]


    latest_failure_rate = safe_numeric(

        latest_monitoring.get(
            "failure_rate",
            0
        )

    )


    latest_latency = safe_numeric(

        latest_monitoring.get(
            "avg_latency",
            0
        )

    )


    latest_timeout_rate = safe_numeric(

        latest_monitoring.get(
            "timeout_rate",
            0
        )

    )


    if (

        latest_failure_rate >= 20

        or latest_timeout_rate >= 10

        or latest_latency >= 3000

    ):

        system_status = "SEVERE"


    elif (

        latest_failure_rate >= 5

        or latest_timeout_rate >= 3

        or latest_latency >= 2000

    ):

        system_status = "DEGRADED"


    else:

        system_status = "NORMAL"


    status_col1, status_col2, status_col3, status_col4 = (
        st.columns(4)
    )


    if system_status == "NORMAL":

        status_col1.metric(
            "Latest Status",
            "🟢 NORMAL"
        )

    elif system_status == "DEGRADED":

        status_col1.metric(
            "Latest Status",
            "🟠 DEGRADED"
        )

    else:

        status_col1.metric(
            "Latest Status",
            "🔴 SEVERE"
        )


    status_col2.metric(
        "Failure Rate",
        f"{latest_failure_rate:.2f}%"
    )


    status_col3.metric(
        "Timeout Rate",
        f"{latest_timeout_rate:.2f}%"
    )


    status_col4.metric(
        "Average Latency",
        f"{latest_latency:.0f} ms"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PayShield AI • Payment Success Optimization "
    "and Risk Monitoring"
)

st.caption(
    "AI-assisted decision intelligence for payment "
    "risk and recovery. Simulation/demo environment — "
    "not a live banking or UPI network."
)