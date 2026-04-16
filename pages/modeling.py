import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Modeling", layout="wide")

st.title("🤖 Model Training & Evaluation")

st.markdown("""
This page shows the machine learning modeling process.
""")

# Load data
@st.cache_data
def load_ml_data():
    return pd.read_csv("data/final_ml.csv")

df = load_ml_data()

st.subheader("ML Dataset")
st.dataframe(df.head())

# Features and target
features = ["ballsfaced", "minutes", "runningscore", "runningover", "isnotout", "captain", "current_innings", "innings_id"]
X = df[features]
y = df["high_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("Train/Test Split")
st.write(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Train model (or load)
if st.button("Train Random Forest Model"):
    with st.spinner("Training model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save
        joblib.dump(model, "models/best_model.pkl")
        st.success("Model trained and saved!")

# Load model
model = joblib.load("models/best_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.write("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

with col2:
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")

def extract_feature_importance(model, features):
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=features)
    if hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            if hasattr(step, "feature_importances_"):
                return pd.Series(step.feature_importances_, index=features)
    if hasattr(model, "steps"):
        for _, step in reversed(model.steps):
            if hasattr(step, "feature_importances_"):
                return pd.Series(step.feature_importances_, index=features)
    return None


def get_transformed_feature_names(model, features):
    if hasattr(model, "named_steps") and "prep" in model.named_steps:
        prep = model.named_steps["prep"]
        if hasattr(prep, "get_feature_names_out"):
            try:
                return list(prep.get_feature_names_out(features))
            except Exception:
                pass
    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out(features))
                except Exception:
                    pass
    return features


def extract_linear_coefficients(model, features):
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", estimator)
    if not hasattr(estimator, "coef_"):
        return None

    coefs = estimator.coef_
    if coefs.ndim == 2:
        coefs = coefs[0]

    transformed_features = get_transformed_feature_names(model, features)
    if len(transformed_features) == len(coefs):
        return pd.Series(coefs, index=transformed_features)
    return pd.Series(coefs)

importance = extract_feature_importance(model, features)
if importance is not None:
    importance = importance.sort_values(ascending=False)
    st.bar_chart(importance)
else:
    coefs = extract_linear_coefficients(model, features)
    if coefs is not None:
        coef_importance = coefs.abs().sort_values(ascending=False)
        st.bar_chart(coef_importance)
        st.write("Showing feature importance from linear model coefficient magnitudes.")
    else:
        st.warning("Feature importance is not available for this model type.")