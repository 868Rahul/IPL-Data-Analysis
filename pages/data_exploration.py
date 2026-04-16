import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Data Exploration", layout="wide")

st.title("📊 Data Exploration")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_batting_card.csv")

df = load_data()

st.subheader("Dataset Overview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

# Top batsmen
st.subheader("Top 10 Batsmen by Runs")
top_batsmen = (
    df.groupby("fullname")["runs"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(top_batsmen)

with col2:
    fig, ax = plt.subplots()
    ax.pie(top_batsmen.values, labels=top_batsmen.index, autopct="%1.1f%%")
    st.pyplot(fig)

# Other plots from EDA
# Add more as needed, like strike rate, etc.

st.subheader("Strike Rate Leaders")
top_sr = (
    df.groupby("fullname")["strikerate"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_sr)

# Season trends
if "season" in df.columns:
    st.subheader("Season-wise Trends")
    season_runs = df.groupby("season")["runs"].sum()
    st.line_chart(season_runs)