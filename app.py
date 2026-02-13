import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("Breast Cancer Classification Dashboard")
st.markdown(
"""
This application demonstrates multiple Machine Learning models used to predict
whether a tumor is **Malignant** or **Benign**.

Upload the provided **testdata.csv** and select a model to view predictions.
"""
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ["logistic", "decision_tree", "knn",
     "naive_bayes", "random_forest", "xgboost"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# ---------------- MAIN ----------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # -------- VALIDATION --------
    if len(data) > 200:
        st.error("Only testdata.csv is allowed.")
        st.stop()

    if "target" not in data.columns:
        st.error("Invalid dataset uploaded.")
        st.stop()

    st.success(f"Test dataset uploaded successfully ({len(data)} rows)")

    # -------- DATA PREVIEW --------
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # -------- LOAD MODEL --------
    model = joblib.load(f"model/{model_name}.pkl")

    X = data.drop("target", axis=1)
    y = data["target"]

    preds = model.predict(X)

    # -------- METRICS SECTION --------
    st.subheader("Model Evaluation Metrics")

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    # -------- PREDICTION SUMMARY --------
    st.subheader("Prediction Summary")

    benign = sum(preds == 0)
    malignant = sum(preds == 1)

    col1, col2 = st.columns(2)
    col1.metric("Benign Predictions", benign)
    col2.metric("Malignant Predictions", malignant)

    # -------- CONFUSION MATRIX --------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    st.pyplot(fig)

    # -------- CLASSIFICATION REPORT --------
    st.subheader("Classification Report")

    report = classification_report(y, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

else:
    st.info("Upload testdata.csv from the sidebar to begin.")
