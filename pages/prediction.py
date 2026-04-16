import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

# Page config
st.set_page_config(
    page_title="🏏 IPL 50+ Score Predictor",
    layout="centered"
)

# Title
st.markdown("""
<h1 style="text-align:center;">🏏 IPL 50+ Score Predictor</h1>
<p style="text-align:center;">Predict whether a batsman will score <b>50+ runs</b></p>
<hr>
""", unsafe_allow_html=True)

# Input Form
st.subheader("📋 Enter Current Innings Details")

col1, col2 = st.columns(2)

with col1:
    ballsfaced = st.number_input("Balls Faced", 1, 200, 30)
    minutes = st.number_input("Minutes Batted", 1, 300, 40)
    runningscore = st.number_input("Current Score", 0, 300, 25)
    runningover = st.number_input("Current Over", 0.0, 50.0, 5.2)

with col2:
    isnotout = st.selectbox("Is Not Out?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    captain = st.selectbox("Is Captain?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    current_innings = st.selectbox("Current Innings", [1, 2])
    innings_id = st.selectbox("Innings ID", [1, 2, 3, 4])

# Predict
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict 50+ Probability", use_container_width=True):

    input_df = pd.DataFrame([{
        "ballsfaced": ballsfaced,
        "minutes": minutes,
        "runningscore": runningscore,
        "runningover": runningover,
        "isnotout": isnotout,
        "captain": captain,
        "current_innings": current_innings,
        "innings_id": innings_id
    }])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric("🎯 Probability of 50+", f"{prob*100:.2f}%")

    if pred == 1:
        st.success("✅ Batsman is likely to score 50+ runs!")
    else:
        st.error("❌ Batsman is unlikely to reach 50 runs.")

    # Explanation
    st.markdown("### 🧠 Why this prediction?")

    reasons = []
    if ballsfaced > 30: reasons.append("Faced many balls")
    if runningscore > 25: reasons.append("Already scored well")
    if minutes > 30: reasons.append("Spent good time at crease")
    if isnotout == 1: reasons.append("Still not out")

    if not reasons:
        reasons.append("Current innings indicators are weak")

    for r in reasons:
        st.write("•", r)

# Footer
st.markdown("""
<hr>
<p style="text-align:center;">Built by Rahul Singh | ML Project 🚀</p>
""", unsafe_allow_html=True)