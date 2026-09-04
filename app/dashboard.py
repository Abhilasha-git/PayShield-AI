import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from risk_engine.engine import PayShieldRiskEngine
from optimization_engine.recommendation_engine import PaymentOptimizationEngine

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PayShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🛡️ PayShield AI")
st.subheader("AI-Powered Payment Success Optimization")
st.write("Payment-system monitoring and risk prediction.")

# ============================================================
# FILE PATHS
# ============================================================

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

TRANSACTION_PATH = os.path.join(RAW_DIR, "transactions.csv")
MONITORING_PATH = os.path.join(DATA_DIR, "payment_monitoring_enhanced.csv")
PREDICTION_PATH = os.path.join(DATA_DIR, "payShield_predictions.csv")
EXPLAINABLE_PATH = os.path.join(DATA_DIR, "explainable_predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "payshield_risk_model.joblib")

# ============================================================
# LOAD DATA & RISK ENGINE
# ============================================================

try:
    transactions = pd.read_csv(TRANSACTION_PATH)
    monitoring = pd.read_csv(MONITORING_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)
    risk_engine = PayShieldRiskEngine.load(MODEL_PATH)
    explainable = pd.read_csv(EXPLAINABLE_PATH) if os.path.exists(EXPLAINABLE_PATH) else pd.DataFrame()
except FileNotFoundError as e:
    st.error(f"Dataset or model file not found:\n\n{e}")
    st.stop()
except Exception as e:
    st.error(f"Application initialization failed:\n\n{e}")
    st.stop()

# ============================================================
# BASIC PREPARATION
# ============================================================

if "time_window" in monitoring.columns:
    monitoring["time_window"] = pd.to_datetime(
        monitoring["time_window"], errors="coerce"
    )
    monitoring = monitoring.sort_values("time_window").reset_index(drop=True)

if "time_window" in predictions.columns:
    predictions["time_window"] = pd.to_datetime(
        predictions["time_window"], errors="coerce"
    )

# Compute canonical bank list once
banks = sorted(monitoring["receiver_bank"].dropna().astype(str).unique().tolist())
if not banks:
    st.warning("No receiver banks found in monitoring data.")
    st.stop()

# ============================================================
# PAYSHIELD RISK ENGINE & OPTIMIZATION ENGINE
# ============================================================

try:
    risk_results = risk_engine.predict(predictions)
    risk_results_df = pd.DataFrame(risk_results)

    predictions["risk_probability"] = risk_results_df["risk_probability"].values
    predictions["risk_level"] = risk_results_df["risk_level"].values
    predictions["is_risky"] = risk_results_df["is_risky"].values
    predictions["risk_probability_pct"] = predictions["risk_probability"] * 100.0

except Exception as e:
    st.error(f"Risk Engine prediction failed:\n\n{e}")
    st.stop()
    
try:
    optimization_engine = PaymentOptimizationEngine()
    predictions = optimization_engine.recommend(predictions)

except Exception as e:
    st.error(f"Optimization Engine failed:\n\n{e}")
    st.stop()
    
optimization_critical = (predictions["optimization_priority"] == "CRITICAL").sum()
optimization_high = (predictions["optimization_priority"] == "HIGH").sum()
optimization_medium = (predictions["optimization_priority"] == "MEDIUM").sum()
optimization_low = (predictions["optimization_priority"] == "LOW").sum()

low_count = predictions["risk_level"].eq("LOW").sum()
elevated_count = predictions["risk_level"].eq("ELEVATED").sum()
medium_count = predictions["risk_level"].eq("MEDIUM").sum()
high_count = predictions["risk_level"].eq("HIGH").sum()

# ============================================================
# HEADER METRICS
# ============================================================

if "payment_status" in transactions.columns:
    success_rate = (
        transactions["payment_status"].astype(str).str.upper().eq("SUCCESS").mean() * 100
    )
else:
    success_rate = 0.0

if "latency_ms" in transactions.columns:
    avg_latency = pd.to_numeric(transactions["latency_ms"], errors="coerce").mean()
else:
    avg_latency = pd.to_numeric(monitoring["avg_latency"], errors="coerce").mean()

if pd.isna(avg_latency):
    avg_latency = 0.0

st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Payment Success Rate", f"{success_rate:.2f}%")
with col2:
    st.metric("Average Latency", f"{avg_latency:.0f} ms")
with col3:
    st.metric("Transactions Monitored", f"{len(transactions):,}")
with col4:
    st.metric("High-Risk Predictions", f"{high_count:,}")

# ============================================================
# PRE-PAYMENT AI RISK CHECK
# ============================================================

st.divider()
st.header("🔮 Pre-Payment AI Risk Check")

st.caption(
    "Simulate a payment-risk check before authorization using the "
    "latest available receiver-bank and payment-system conditions."
)

pre_col1, pre_col2, pre_col3, pre_col4 = st.columns(4)

with pre_col1:
    pre_receiver_bank = st.selectbox(
        "Receiver Bank",
        banks,
        key="pre_payment_bank"
    )

with pre_col2:
    pre_amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        value=5000.0,
        step=500.0,
        key="pre_payment_amount"
    )

with pre_col3:
    pre_hour = st.slider(
        "Payment Hour",
        min_value=0,
        max_value=23,
        value=pd.Timestamp.now().hour,
        key="pre_payment_hour"
    )

with pre_col4:
    simulate_stress = st.toggle(
        "Simulate Bank Stress",
        value=False,
        key="pre_payment_stress"
    )

if st.button(
    "🔍 CHECK PAYMENT RISK",
    type="primary",
    use_container_width=True,
    key="pre_payment_check"
):
    receiver_data = monitoring[
        monitoring["receiver_bank"].astype(str) == str(pre_receiver_bank)
    ].copy()

    if receiver_data.empty:
        st.warning(f"No monitoring data is currently available for {pre_receiver_bank}.")
    else:
        receiver_data = receiver_data.sort_values("time_window")
        latest_receiver = receiver_data.iloc[-1]

        def safe_numeric(column, default=0.0):
            if column in latest_receiver.index:
                value = pd.to_numeric(latest_receiver[column], errors="coerce")
                return float(value) if pd.notna(value) else default
            return default

        # ----------------------------------------------------
        # FEATURE PREPARATION & STRESS TESTING
        # ----------------------------------------------------
        if simulate_stress:
            pre_features = {
                "transaction_count": safe_numeric("transaction_count", 350.0),
                "failure_rate": 45.0,
                "timeout_rate": 25.0,
                "avg_latency": 3850.0,
                "max_latency": 5500.0,
                "p95_latency": 4500.0,
                "bank_error_rate": 0.35,
                "avg_amount": float(pre_amount),
                "max_amount": max(safe_numeric("max_amount", pre_amount), float(pre_amount)),
                "hour": int(pre_hour),
                "day_of_week": int(pd.Timestamp.now().dayofweek),
                "is_weekend": int(pd.Timestamp.now().dayofweek >= 5),
                "previous_failure_rate": 35.0,
                "previous_latency": 3200.0,
                "previous_timeout_rate": 20.0,
                "failure_rate_change": 20.0,
                "latency_change": 1500.0,
                "timeout_rate_change": 12.0,
                "rolling_failure_rate": 40.0,
                "rolling_latency": 3500.0,
                "rolling_timeout_rate": 22.0,
            }

            st.warning(
                "⚠️ Simulation mode active: degraded receiver-bank "
                "conditions are being supplied to the AI risk engine."
            )
        else:
            pre_features = {
                "transaction_count": safe_numeric("transaction_count", 100.0),
                "failure_rate": safe_numeric("failure_rate", 0.0),
                "timeout_rate": safe_numeric("timeout_rate", 0.0),
                "avg_latency": safe_numeric("avg_latency", 1200.0),
                "max_latency": safe_numeric("max_latency", 2500.0),
                "p95_latency": safe_numeric("p95_latency", 1800.0),
                "bank_error_rate": safe_numeric("bank_error_rate", 0.0),
                "avg_amount": float(pre_amount),
                "max_amount": max(safe_numeric("max_amount", pre_amount), float(pre_amount)),
                "hour": int(pre_hour),
                "day_of_week": int(pd.Timestamp.now().dayofweek),
                "is_weekend": int(pd.Timestamp.now().dayofweek >= 5),
                "previous_failure_rate": safe_numeric("previous_failure_rate", 0.0),
                "previous_latency": safe_numeric("previous_latency", 1200.0),
                "previous_timeout_rate": safe_numeric("previous_timeout_rate", 0.0),
                "failure_rate_change": safe_numeric("failure_rate_change", 0.0),
                "latency_change": safe_numeric("latency_change", 0.0),
                "timeout_rate_change": safe_numeric("timeout_rate_change", 0.0),
                "rolling_failure_rate": safe_numeric("rolling_failure_rate", 0.0),
                "rolling_latency": safe_numeric("rolling_latency", 1200.0),
                "rolling_timeout_rate": safe_numeric("rolling_timeout_rate", 0.0),
            }

        pre_input = pd.DataFrame([pre_features])

        # ----------------------------------------------------
        # AI RISK PREDICTION & RECOMMENDATION
        # ----------------------------------------------------
        try:
            pre_result = risk_engine.predict(pre_input)[0]
            pre_probability = float(pre_result["risk_probability"])
            pre_risk_level = str(pre_result["risk_level"]).upper()
            pre_probability_pct = pre_probability * 100.0

            pre_input["risk_probability_pct"] = pre_probability_pct
            pre_input["risk_level"] = pre_risk_level
            pre_input["receiver_bank"] = pre_receiver_bank

            opt_res = optimization_engine.recommend(pre_input)
            rec_action = opt_res["recommended_action"].iloc[0]
            rec_reason = opt_res["optimization_reason"].iloc[0]

            st.subheader("AI Payment Risk Assessment")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric("AI Risk Probability", f"{pre_probability_pct:.2f}%")
            with result_col2:
                st.metric("Risk Tier", pre_risk_level)
            with result_col3:
                st.metric("Receiver Bank", str(pre_receiver_bank))

            if pre_risk_level == "HIGH":
                st.error(
                    "🔴 HIGH RISK\n\n"
                    "⚠️ Payment may fail based on current payment-system conditions."
                )
            elif pre_risk_level == "MEDIUM":
                st.warning(
                    "🟠 MEDIUM RISK\n\n"
                    "⚠️ Payment has an elevated probability of failure."
                )
            elif pre_risk_level == "ELEVATED":
                st.warning(
                    "🟡 ELEVATED RISK\n\n"
                    "Payment-system conditions show signs of instability."
                )
            else:
                st.success(
                    "🟢 LOW RISK\n\n"
                    "Payment-system conditions currently appear healthy."
                )

            st.markdown(f"**Recommended Action:** {rec_action}")
            st.caption(f"**AI Reason:** {rec_reason}")

        except Exception as e:
            st.error(f"Pre-payment risk prediction failed:\n\n{e}")

# ============================================================
# BANK HEALTH MONITORING
# ============================================================

st.divider()
st.header("🏦 Bank Health Monitoring")

selected_bank = st.selectbox("Select Receiver Bank", banks)

bank_data = monitoring[
    monitoring["receiver_bank"].astype(str) == selected_bank
].copy()

if bank_data.empty:
    st.warning("No monitoring data available for this bank.")
    st.stop()

bank_data = bank_data.sort_values("time_window").reset_index(drop=True)
latest_bank = bank_data.iloc[-1]

bank_col1, bank_col2, bank_col3, bank_col4 = st.columns(4)
with bank_col1:
    st.metric("Bank Health", str(latest_bank["bank_health"]))
with bank_col2:
    st.metric("Failure Rate", f"{latest_bank['failure_rate']:.2f}%")
with bank_col3:
    st.metric("Timeout Rate", f"{latest_bank['timeout_rate']:.2f}%")
with bank_col4:
    st.metric("Average Latency", f"{latest_bank['avg_latency']:.0f} ms")

st.subheader(f"📈 {selected_bank} Performance Trend")

bank_trend = bank_data[
    ["time_window", "avg_latency", "failure_rate", "timeout_rate"]
].copy()

bank_trend["time_window"] = pd.to_datetime(bank_trend["time_window"], errors="coerce")

bank_trend = (
    bank_trend.dropna(subset=["time_window"])
    .set_index("time_window")
    .resample("D")
    .agg({
        "avg_latency": "mean",
        "failure_rate": "mean",
        "timeout_rate": "mean"
    })
    .reset_index()
)

trend_fig = go.Figure()

trend_fig.add_trace(
    go.Scatter(
        x=bank_trend["time_window"],
        y=bank_trend["avg_latency"],
        mode="lines",
        name="Average Latency",
        yaxis="y1"
    )
)

trend_fig.add_trace(
    go.Scatter(
        x=bank_trend["time_window"],
        y=bank_trend["failure_rate"],
        mode="lines",
        name="Failure Rate",
        yaxis="y2"
    )
)

trend_fig.add_trace(
    go.Scatter(
        x=bank_trend["time_window"],
        y=bank_trend["timeout_rate"],
        mode="lines",
        name="Timeout Rate",
        yaxis="y2"
    )
)

trend_fig.update_layout(
    height=450,
    hovermode="x unified",
    xaxis=dict(title="Time"),
    yaxis=dict(title="Average Latency (ms)", side="left"),
    yaxis2=dict(
        title="Failure / Timeout Rate (%)",
        side="right",
        overlaying="y",
        rangemode="tozero"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=60, r=60, t=60, b=40)
)

st.plotly_chart(trend_fig, use_container_width=True)

if str(latest_bank["bank_health"]).upper() == "NORMAL":
    st.success(f"✅ {selected_bank} is operating normally.")
else:
    st.warning(
        f"⚠️ {selected_bank} requires attention. Current health: {latest_bank['bank_health']}"
    )

# ============================================================
# RISK MONITORING & DISTRIBUTION
# ============================================================

st.divider()
st.header("📊 Risk Distribution & Tier Breakdown")

total_predictions = len(predictions)

high_pct = (high_count / total_predictions * 100) if total_predictions > 0 else 0.0
medium_pct = (medium_count / total_predictions * 100) if total_predictions > 0 else 0.0
elevated_pct = (elevated_count / total_predictions * 100) if total_predictions > 0 else 0.0
low_pct = (low_count / total_predictions * 100) if total_predictions > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔴 High Risk (≥ 80%)", f"{high_count:,}")
    st.caption(f"**{high_pct:.1f}%** of predictions")
with col2:
    st.metric("🟠 Medium Risk (50–80%)", f"{medium_count:,}")
    st.caption(f"**{medium_pct:.1f}%** of predictions")
with col3:
    st.metric("🟡 Elevated Risk (20–50%)", f"{elevated_count:,}")
    st.caption(f"**{elevated_pct:.1f}%** of predictions")
with col4:
    st.metric("🟢 Low Risk (< 20%)", f"{low_count:,}")
    st.caption(f"**{low_pct:.1f}%** of predictions")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.subheader("Distribution by Volume")
    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(
            x=["Low Risk (< 20%)", "Elevated Risk (20–50%)", "Medium Risk (50–80%)", "High Risk (≥ 80%)"],
            y=[low_count, elevated_count, medium_count, high_count],
            text=[f"{low_count:,} ({low_pct:.1f}%)", f"{elevated_count:,} ({elevated_pct:.1f}%)", f"{medium_count:,} ({medium_pct:.1f}%)", f"{high_count:,} ({high_pct:.1f}%)"],
            textposition="outside"
        )
    )
    bar_fig.update_layout(
        xaxis_title="Risk Tier",
        yaxis_title="Prediction Count",
        height=420,
        margin=dict(t=40, b=40, l=40, r=20),
        showlegend=False
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with dist_col2:
    st.subheader("Risk Composition")
    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=["Low Risk", "Elevated Risk", "Medium Risk", "High Risk"],
                values=[low_count, elevated_count, medium_count, high_count],
                hole=0.55,
                textinfo="percent"
            )
        ]
    )
    pie_fig.update_layout(
        height=420,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(pie_fig, use_container_width=True)

# ============================================================
# HIGH-RISK TRANSACTIONS & ACTIONABLE DIAGNOSTICS
# ============================================================

st.divider()
st.header("🚨 High-Risk Transaction Diagnostics & Root Causes")

diag_df = predictions.copy()

if "risk_probability_pct" not in diag_df.columns:
    diag_df["risk_probability_pct"] = pd.to_numeric(diag_df["risk_probability"], errors="coerce") * 100.0

if not explainable.empty:
    join_candidates = ["transaction_id", "time_window"]
    available_joins = [c for c in join_candidates if c in diag_df.columns and c in explainable.columns]
    if available_joins:
        exp_cols = [c for c in explainable.columns if c not in diag_df.columns or c in available_joins]
        diag_df = pd.merge(diag_df, explainable[exp_cols], on=available_joins, how="left")

reason_col = None
for candidate in ["top_risk_driver", "risk_reason", "explanation", "primary_factor", "anomaly_reason"]:
    if candidate in diag_df.columns:
        reason_col = candidate
        break

if reason_col is None:
    def synthesize_driver(row):
        reasons = []
        if "failure_rate" in row and pd.to_numeric(row["failure_rate"], errors="coerce") > 5:
            reasons.append("Elevated Bank Failure Rate")
        if "timeout_rate" in row and pd.to_numeric(row["timeout_rate"], errors="coerce") > 3:
            reasons.append("Gateway Timeout Surge")
        if "avg_latency" in row and pd.to_numeric(row["avg_latency"], errors="coerce") > 300:
            reasons.append("High Latency Spike")
        if not reasons:
            reasons.append("Model Anomaly Score")
        return " • ".join(reasons)

    diag_df["primary_risk_driver"] = diag_df.apply(synthesize_driver, axis=1)
    reason_col = "primary_risk_driver"

if "transaction_id" not in diag_df.columns:
    diag_df["transaction_id"] = [f"TXN-{100000 + i}" for i in range(len(diag_df))]

with st.expander("🔍 Filter & Search Transactions", expanded=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        tier_filter = st.multiselect(
            "Risk Tier",
            options=["HIGH", "MEDIUM", "ELEVATED", "LOW"],
            default=["HIGH"],
            help="Filter by PayShield Risk Engine tiers."
        )

    tier_scoped_df = diag_df[diag_df["risk_level"].isin(tier_filter)] if tier_filter else diag_df.copy()

    with f_col2:
        available_tags = set()
        for item in tier_scoped_df[reason_col].dropna().unique():
            for sub_tag in str(item).split(" • "):
                if sub_tag.strip():
                    available_tags.add(sub_tag.strip())

        driver_filter = st.multiselect(
            "Risk Driver",
            options=sorted(list(available_tags)),
            default=[],
            help="Filter by diagnosed root-cause factors"
        )

    with f_col3:
        if "HIGH" in tier_filter:
            min_slider_default = 80.0
        elif "MEDIUM" in tier_filter:
            min_slider_default = 50.0
        elif "ELEVATED" in tier_filter:
            min_slider_default = 20.0
        else:
            min_slider_default = 0.0

        min_prob = st.slider(
            "Min Risk Probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=min_slider_default,
            step=1.0
        )

    with f_col4:
        sort_order = st.selectbox(
            "Sort By",
            options=["Highest Risk", "Lowest Risk", "Most Recent"] if "time_window" in diag_df.columns else ["Highest Risk", "Lowest Risk"]
        )

filtered_df = tier_scoped_df[tier_scoped_df["risk_probability_pct"] >= min_prob].copy()

if driver_filter:
    filtered_df = filtered_df[
        filtered_df[reason_col].apply(lambda x: any(tag in str(x) for tag in driver_filter))
    ]

if "risk_probability_pct" in filtered_df.columns:
    if sort_order == "Highest Risk":
        filtered_df = filtered_df.sort_values("risk_probability_pct", ascending=False)
    elif sort_order == "Lowest Risk":
        filtered_df = filtered_df.sort_values("risk_probability_pct", ascending=True)

if sort_order == "Most Recent" and "time_window" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("time_window", ascending=False)

filtered_df = filtered_df.reset_index(drop=True)

if filtered_df.empty:
    st.info("ℹ️ No transactions match this specific combination of tier, driver, and probability filters.")
else:
    st.caption(f"Displaying **{len(filtered_df):,}** matching transactions")

    display_cols = ["transaction_id", "risk_level", "risk_probability_pct", reason_col]
    for opt_col in ["receiver_bank", "time_window", "avg_latency", "failure_rate"]:
        if opt_col in filtered_df.columns:
            display_cols.append(opt_col)

    table_payload = filtered_df[display_cols].copy()
    table_payload["risk_probability_pct"] = table_payload["risk_probability_pct"].map(lambda x: f"{x:.2f}%")

    if "failure_rate" in table_payload.columns:
        numeric_fail = pd.to_numeric(table_payload["failure_rate"], errors="coerce").fillna(0.0)
        table_payload["failure_rate"] = numeric_fail.round(2).astype(str) + "%"

    if "avg_latency" in table_payload.columns:
        table_payload["avg_latency"] = pd.to_numeric(table_payload["avg_latency"], errors="coerce").fillna(0.0).round(0).astype(str) + " ms"

    rename_display = {
        "transaction_id": "Transaction ID",
        "risk_level": "Tier",
        "risk_probability_pct": "Predicted Risk",
        reason_col: "Primary Risk Driver",
        "receiver_bank": "Receiver Bank",
        "time_window": "Timestamp",
        "avg_latency": "Latency",
        "failure_rate": "Bank Failure Rate"
    }
    table_payload = table_payload.rename(columns=rename_display)

    selection_event = st.dataframe(
        table_payload,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="high_risk_tx_table"
    )

selected_rows = []
if 'selection_event' in locals() and hasattr(selection_event, "selection"):
    selected_rows = selection_event.selection.rows

if selected_rows and not filtered_df.empty:
    selected_idx = selected_rows[0]
    selected_record = filtered_df.iloc[selected_idx]

    st.markdown("### 🔎 Transaction Deep Dive")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)

    with d_col1:
        st.metric("Transaction ID", str(selected_record.get("transaction_id", "N/A")))
    with d_col2:
        st.metric("Assigned Risk Tier", str(selected_record.get("risk_level", "N/A")))
    with d_col3:
        st.metric("Predicted Failure Risk", f"{selected_record.get('risk_probability_pct', 0.0):.2f}%")
    with d_col4:
        st.metric("Receiver Bank", str(selected_record.get("receiver_bank", "N/A")))

    st.info(f"**Root-Cause Diagnostic:** {selected_record.get(reason_col, 'N/A')}")
elif not filtered_df.empty:
    st.caption("💡 *Select any row in the table above to view deep-dive parameters.*")

st.markdown("### 🛡️ Recommended Actions & Key Findings")

action_col1, action_col2 = st.columns(2)

with action_col1:
    st.markdown("""
    **Key Operational Findings**
    * **Latency Sensitivity:** High-risk predictions correlate strongly with gateway response latencies exceeding 300 ms.
    * **Bank Routing Clustering:** Peak risk transactions cluster around specific downstream receiver banks undergoing downtime.
    * **Failure Isolation:** High-risk (≥80%) predictions represent a small share of total volume, enabling targeted traffic throttling.
    """)

with action_col2:
    st.markdown("""
    **AI-Assisted Mitigation Recommendations**
    * 🔄 **Smart Rerouting:** Recommend temporarily rerouting subsequent transactions away from high-latency receiver bank endpoints.
    * ⏱️ **Adaptive Timeout Adjustments:** Recommend extending transaction verification windows by 500 ms during detected latency surges.
    * 🔔 **Webhook Alerts:** Recommend triggering webhook notifications to merchant integration systems for transactions flagged as `HIGH`.
    """)

# ============================================================
# AI-Assisted PAYMENT SUCCESS OPTIMIZATION
# ============================================================

st.divider()
st.header("🤖 AI-Assisted Payment Success Optimization")

st.caption(
    "AI-assisted recommendations generated from PayShield risk predictions "
    "and payment-system performance indicators."
)

opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

with opt_col1:
    st.metric("Critical Actions", f"{optimization_critical:,}")

with opt_col2:
    st.metric("High Priority", f"{optimization_high:,}")

with opt_col3:
    st.metric("Medium Priority", f"{optimization_medium:,}")

with opt_col4:
    st.metric("No Intervention", f"{optimization_low:,}")

st.subheader("Recommended Payment Actions")

action_counts = (
    predictions["recommended_action"]
    .value_counts()
    .reset_index()
)

action_counts.columns = [
    "Recommended Action",
    "Count"
]

st.bar_chart(
    action_counts.set_index("Recommended Action")
)

st.subheader("AI Optimization Recommendations")

optimization_display = predictions[
    [
        "risk_probability_pct",
        "risk_level",
        "recommended_action",
        "optimization_priority",
        "optimization_reason"
    ]
].copy()

# Fix priority ordering: CRITICAL -> HIGH -> MEDIUM -> LOW
priority_order = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3
}
optimization_display["priority_rank"] = (
    optimization_display["optimization_priority"]
    .map(priority_order)
    .fillna(4)
)

optimization_display = optimization_display.sort_values(
    ["priority_rank", "risk_probability_pct"],
    ascending=[True, False]
).drop(columns=["priority_rank"])

optimization_display["risk_probability_pct"] = (
    optimization_display["risk_probability_pct"]
    .map(lambda x: f"{x:.2f}%")
)

optimization_display = optimization_display.rename(
    columns={
        "risk_probability_pct": "AI Risk",
        "risk_level": "Risk Tier",
        "recommended_action": "Recommended Action",
        "optimization_priority": "Priority",
        "optimization_reason": "AI Reason"
    }
)

st.dataframe(
    optimization_display.head(25),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# RISK PROBABILITY TREND — Model Output (Independent Figure)
# ============================================================

st.divider()
st.header("📈 Risk Probability Trend — Model Output")

st.caption(
    "Model risk probabilities aligned across the available monitoring period "
    "for visualization. Prediction records do not contain individual timestamps, "
    "so this is not an event-level historical risk time series."
)

trend_df = pd.DataFrame()

if "time_window" in predictions.columns and predictions["time_window"].notna().any():
    temp = predictions.dropna(subset=["time_window", "risk_probability_pct"]).copy()
    temp["time_window"] = pd.to_datetime(temp["time_window"], errors="coerce")
    trend_df = (
        temp.groupby("time_window")["risk_probability_pct"]
        .mean()
        .reset_index()
        .sort_values("time_window")
    )
elif "time_window" in monitoring.columns and not monitoring["time_window"].dropna().empty:
    valid_times = pd.to_datetime(monitoring["time_window"], errors="coerce").dropna()
    valid_probs = predictions["risk_probability_pct"].dropna().values

    if len(valid_probs) > 0 and not valid_times.empty:
        start_date = valid_times.min()
        end_date = valid_times.max()
        n_points = min(150, len(valid_probs))
        time_index = pd.date_range(start=start_date, end=end_date, periods=n_points)
        chunks = np.array_split(valid_probs, n_points)
        avg_chunks = [c.mean() if len(c) > 0 else 0.0 for c in chunks]

        trend_df = pd.DataFrame({
            "time_window": time_index,
            "risk_probability_pct": avg_chunks
        })

if not trend_df.empty:
    trend_df["rolling_avg_pct"] = (
        trend_df["risk_probability_pct"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    trend_max = float(trend_df["risk_probability_pct"].max())
    trend_avg = float(trend_df["risk_probability_pct"].mean())

    y_max = max(6.0, float(np.ceil(trend_max * 1.35)))

    model_trend_fig = go.Figure()

    model_trend_fig.add_hrect(
        y0=0, y1=min(2.0, y_max),
        fillcolor="rgba(40, 167, 69, 0.08)",
        layer="below", line_width=0,
        annotation_text="🟢 Baseline (< 2%)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="rgba(40, 167, 69, 0.7)")
    )
    if y_max > 2.0:
        model_trend_fig.add_hrect(
            y0=2.0, y1=min(4.0, y_max),
            fillcolor="rgba(255, 193, 7, 0.08)",
            layer="below", line_width=0,
            annotation_text="🟡 Elevated (2–4%)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="rgba(255, 193, 7, 0.8)")
        )
    if y_max > 4.0:
        model_trend_fig.add_hrect(
            y0=4.0, y1=y_max,
            fillcolor="rgba(220, 53, 69, 0.08)",
            layer="below", line_width=0,
            annotation_text="🔴 Peak Spike (> 4%)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="rgba(220, 53, 69, 0.7)")
        )

    model_trend_fig.add_trace(
        go.Scatter(
            x=trend_df["time_window"],
            y=trend_df["risk_probability_pct"],
            mode="lines",
            line=dict(color="rgba(255, 100, 100, 0.35)", width=1.5),
            hovertemplate="<b>Date:</b> %{x|%b %d}<br><b>Predicted Risk:</b> %{y:.2f}%<extra></extra>",
            name="Predicted Risk"
        )
    )

    model_trend_fig.add_trace(
        go.Scatter(
            x=trend_df["time_window"],
            y=trend_df["rolling_avg_pct"],
            mode="lines",
            line=dict(color="#FF4B4B", width=2.5),
            hovertemplate="<b>Date:</b> %{x|%b %d}<br><b>7-Day Avg:</b> %{y:.2f}%<extra></extra>",
            name="7-Day Rolling Avg"
        )
    )

    model_trend_fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Date", tickformat="%b %d", showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)"),
        yaxis=dict(
            title="Risk Probability (%)",
            range=[0, y_max],
            ticksuffix="%",
            dtick=1.0,
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)"
        ),
        hovermode="x unified"
    )

    st.plotly_chart(model_trend_fig, use_container_width=True)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.metric("Peak Risk Probability", f"{trend_max:.2f}%")
    with t_col2:
        st.metric("Average Risk Probability", f"{trend_avg:.2f}%")
else:
    st.warning("⚠️ Insufficient time-window data available to render the trend.")

# ============================================================
# CURRENT SYSTEM STATUS
# ============================================================

st.divider()
st.header("🟢 Current Payment-System Status")
st.caption("Based on the latest available monitoring observation.")

latest_monitoring = monitoring.sort_values("time_window").iloc[-1]
latest_failure_rate = pd.to_numeric(latest_monitoring["failure_rate"], errors="coerce")
latest_latency = pd.to_numeric(latest_monitoring["avg_latency"], errors="coerce")

if pd.isna(latest_failure_rate):
    latest_failure_rate = 0.0
if pd.isna(latest_latency):
    latest_latency = 0.0

if latest_failure_rate >= 5:
    st.error(f"🚨 Payment system requires immediate attention. Failure rate: {latest_failure_rate:.2f}%")
elif latest_failure_rate >= 2:
    st.warning(f"⚠️ Payment system showing elevated failures. Failure rate: {latest_failure_rate:.2f}%")
else:
    st.success(f"✅ Latest monitored condition is within normal range. Failure rate: {latest_failure_rate:.2f}%")

st.write(f"Latest average latency: **{latest_latency:.0f} ms**")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("PayShield AI • Payment Success Optimization and Risk Monitoring")