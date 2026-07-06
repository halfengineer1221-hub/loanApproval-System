"""
==============================================================
 PHASE 8 — MODEL DEPLOYMENT (Streamlit)
==============================================================
Loads the trained model + preprocessing artifacts produced by
loan_approval_pipeline.py and serves an interactive form where a
user enters applicant details and gets an instant loan approval
prediction with probability.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ARTIFACTS_PATH = "models/preprocessing_artifacts.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    return model, scaler, artifacts


def build_feature_row(raw: dict, encoders: dict, selected_columns: list) -> pd.DataFrame:
    """Recreates, in the same order, every preprocessing/feature-engineering
    step from the training pipeline for a single new applicant."""
    df = pd.DataFrame([raw])

    # Encode categorical columns with the SAME encoders fit during training
    for col in ["Gender", "Married", "Education", "Self_Employed"]:
        le = encoders[col]
        df[col] = le.transform(df[col])

    df["Dependents"] = df["Dependents"].astype(int)

    # One-hot Property_Area exactly like training (drop_first=True -> Rural is baseline)
    df["Property_Area_Semiurban"] = int(raw["Property_Area"] == "Semiurban")
    df["Property_Area_Urban"] = int(raw["Property_Area"] == "Urban")
    df = df.drop(columns=["Property_Area"])

    # Feature engineering — identical formulas to the training pipeline
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = (df["LoanAmount"] * 1000) / df["Loan_Amount_Term"]
    df["Income_to_Loan_Ratio"] = df["Total_Income"] / df["LoanAmount"].replace(0, np.nan)
    df["Balance_Income"] = df["Total_Income"] - df["EMI"]
    for c in ["ApplicantIncome", "CoapplicantIncome", "Total_Income", "LoanAmount"]:
        df[f"Log_{c}"] = np.log1p(df[c])

    # Keep only the columns the model was trained on, in the right order
    missing = [c for c in selected_columns if c not in df.columns]
    for c in missing:
        df[c] = 0  # safety net; shouldn't trigger if schemas match
    return df[selected_columns]


def main():
    st.title("🏦 Loan Approval Prediction System")
    st.caption("Fill in applicant details to get an instant approval prediction.")

    try:
        model, scaler, artifacts = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run `python loan_approval_pipeline.py "
            "--data data/loan_data.csv` first to train and save the model."
        )
        st.stop()

    encoders = artifacts["encoders"]
    selected_columns = artifacts["selected_columns"]

    st.subheader("Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    with col2:
        applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Coapplicant Monthly Income (₹)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=1, value=120, step=5)
        loan_term = st.selectbox("Loan Term (days)", [360, 180, 120, 60, 300, 240], index=0)
        credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

    if st.button("Predict Loan Approval", type="primary"):
        raw = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": 1.0 if credit_history.startswith("Good") else 0.0,
            "Property_Area": property_area,
        }

        try:
            X_new = build_feature_row(raw, encoders, selected_columns)
            X_scaled = scaler.transform(X_new)
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]
        except Exception as e:
            st.error(f"Could not generate a prediction: {e}")
            st.stop()

        st.divider()
        if pred == 1:
            st.success(f"✅ Loan Approved — confidence {prob:.1%}")
        else:
            st.error(f"❌ Loan Rejected — confidence {1 - prob:.1%}")

        st.progress(float(prob))
        st.caption(f"Model's estimated approval probability: {prob:.1%}")

        with st.expander("See the engineered features used for this prediction"):
            st.dataframe(X_new.T.rename(columns={0: "value"}))

    st.divider()
    st.caption(
        "Model trained with Logistic Regression / Random Forest / Gradient Boosting, "
        "selected by highest F1 score, with SMOTE balancing and ANOVA-based feature "
        "selection. Predictions are for demonstration purposes only and not a substitute "
        "for a real underwriting decision."
    )


if __name__ == "__main__":
    main()
