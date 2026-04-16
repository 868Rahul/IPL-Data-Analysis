import streamlit as st

st.set_page_config(
    page_title="🏏 IPL Data Analysis & Prediction",
    layout="wide"
)

st.markdown("""
# 🏏 IPL Data Analysis & Prediction System

Welcome to the comprehensive IPL analysis platform! This application provides:

## 📊 Features

- **Data Exploration**: View and analyze IPL batting data
- **Feature Engineering**: Understand how features are prepared for ML
- **Model Training**: Train and evaluate machine learning models
- **Prediction**: Predict if a batsman will score 50+ runs

## 🚀 Navigation

Use the sidebar to navigate between different sections of the application.

---

Built with ❤️ using Streamlit | Data Science Project
""")

# Sidebar navigation (though Streamlit handles it automatically with pages/)
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a page from above.")
