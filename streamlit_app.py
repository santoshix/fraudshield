"""Optional Streamlit data-analysis dashboard for the exact same FastAPI backend."""
import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API = os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="FraudShield — Analytics", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudShield | Analytics Studio")
st.caption("Portfolio demonstration. Model trained on synthetic transactions; predictions are not verified fraud findings.")

def get(path):
    response = requests.get(API + path, timeout=15)
    response.raise_for_status()
    return response.json()

def post(path, data=None):
    response = requests.post(API + path, json=data, timeout=90)
    response.raise_for_status()
    return response.json()

try:
    metrics, records, model = get("/api/metrics"), get("/api/transactions?limit=500"), get("/api/model")
except requests.RequestException:
    st.error("API is not running. First launch `python run.py` on your computer (or `docker compose up --build`).")
    st.stop()

left, right, third, fourth = st.columns(4)
left.metric("Transactions", metrics["total"])
right.metric("High risk", metrics["high_risk"])
third.metric("Requires review", metrics["review"])
fourth.metric("Average risk", f'{metrics["average_risk"]:.1f}/100')

tab1, tab2, tab3 = st.tabs(["📊 Insights", "🧠 Try the AI", "⚙️ Model report"])
with tab1:
    if records:
        df = pd.DataFrame(records)
        col1, col2 = st.columns([2, 1])
        with col1:
            chart = df.sort_values("created_at")[["created_at", "risk_score"]]
            st.plotly_chart(px.line(chart, x="created_at", y="risk_score", title="Risk scores over time", labels={"risk_score":"Risk score", "created_at":"Time"}), use_container_width=True)
        with col2:
            fig = px.pie(df, names="status", title="Risk category mix", hole=.55)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Saved transactions")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data yet. Generate demo transactions from the main dashboard (localhost:8000).")
with tab2:
    with st.form("score_form"):
        a, b = st.columns(2)
        amount = a.number_input("Amount ($)", min_value=.01, max_value=1_000_000.0, value=2400.0)
        hour = b.slider("Hour of day", 0, 23, 2)
        distance = a.number_input("Distance from home (km)", 0.0, 30000.0, 650.0)
        velocity = b.slider("Transactions in last 24h", 0, 100, 11)
        age = a.number_input("Account age (days)", 0, 36500, 14)
        merchant = b.slider("Merchant risk (synthetic feature)", 0.0, 1.0, .8)
        foreign = a.checkbox("International transaction", value=True)
        online = b.checkbox("Online purchase", value=True)
        submitted = st.form_submit_button("Predict and save", type="primary")
    if submitted:
        try:
            result = post("/api/score", {"amount": amount, "hour": hour, "distance_from_home":distance,
                "velocity_24h":velocity, "account_age_days":age, "merchant_risk":merchant,
                "is_foreign":foreign, "is_online":online})
            st.success(f'Prediction: {result["status"]} | Risk score: {result["risk_score"]}/100 | ID: {result["transaction_id"]}')
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
with tab3:
    st.metric("XGBoost ROC-AUC (synthetic validation only)", model["roc_auc"])
    st.metric("Average precision (synthetic validation only)", model["average_precision"])
    st.json(model)
    st.warning("Do not use this synthetic model to approve, reject or flag real people's payments.")
