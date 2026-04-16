import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Feature Engineering", layout="wide")

st.title("🔧 Feature Engineering")

st.markdown("""
This page demonstrates the feature engineering process for the ML model.

## Steps:
1. Load cleaned data
2. Create target variable: `high_score` (1 if runs >= 50)
3. Remove data leakage columns
4. Select final features
""")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_batting_card.csv")

df = load_data()

st.subheader("1. Original Data")
st.dataframe(df.head())

# Create target
df_fe = df.copy()
df_fe["high_score"] = (df_fe["runs"] >= 50).astype(int)

st.subheader("2. Target Variable Created")
st.write("high_score = 1 if runs >= 50, else 0")
st.write(df_fe["high_score"].value_counts())

# Remove leakage
leakage_cols = ["runs", "fours", "sixes", "strikerate"]
df_fe = df_fe.drop(columns=leakage_cols)

st.subheader("3. Removed Data Leakage Columns")
st.write(f"Removed: {leakage_cols}")
st.dataframe(df_fe.head())

# Final features
st.subheader("4. Final Features for ML")
st.write("Features used:")
features = ["ballsfaced", "minutes", "runningscore", "runningover", "isnotout", "captain", "current_innings", "innings_id"]
st.write(features)
st.dataframe(df_fe[features + ["high_score"]].head())

# Save final_ml.csv if not exists
if not os.path.exists("data/final_ml.csv"):
    df_fe.to_csv("data/final_ml.csv", index=False)
    st.success("Saved final_ml.csv")