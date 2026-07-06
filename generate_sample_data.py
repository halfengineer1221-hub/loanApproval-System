"""
Generates a synthetic dataset matching the schema of the Kaggle
'Loan Prediction Dataset' (Loan_ID, Gender, Married, Dependents, Education,
Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status).

This is ONLY for testing the pipeline end-to-end before you plug in the
real Kaggle CSV (train_ctrl_loan.csv / loan_prediction.csv). Replace
'data/loan_data.csv' with the real file for actual use.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 614  # same size as the real Kaggle dataset

df = pd.DataFrame({
    "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(1, n + 1)],
    "Gender": np.random.choice(["Male", "Female", np.nan], n, p=[0.75, 0.22, 0.03]),
    "Married": np.random.choice(["Yes", "No", np.nan], n, p=[0.65, 0.32, 0.03]),
    "Dependents": np.random.choice(["0", "1", "2", "3+", np.nan], n, p=[0.55, 0.18, 0.15, 0.09, 0.03]),
    "Education": np.random.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22]),
    "Self_Employed": np.random.choice(["Yes", "No", np.nan], n, p=[0.13, 0.82, 0.05]),
    "ApplicantIncome": np.random.gamma(shape=2.5, scale=2200, size=n).astype(int) + 1500,
    "CoapplicantIncome": np.random.choice(
        [0, *np.random.gamma(shape=2, scale=1200, size=n).astype(int)], n
    ),
    "LoanAmount": np.round(np.random.gamma(shape=3, scale=45, size=n) + 20, 0),
    "Loan_Amount_Term": np.random.choice([360, 180, 120, 60, 300, 240, np.nan], n,
                                          p=[0.75, 0.08, 0.05, 0.03, 0.03, 0.03, 0.03]),
    "Credit_History": np.random.choice([1.0, 0.0, np.nan], n, p=[0.78, 0.14, 0.08]),
    "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], n, p=[0.38, 0.38, 0.24]),
})

# Inject a realistic, learnable relationship into Loan_Status
score = (
    (df["Credit_History"].fillna(0) * 3)
    + (df["Education"].eq("Graduate").astype(int) * 0.6)
    + (np.log1p(df["ApplicantIncome"] + df["CoapplicantIncome"]) * 0.4)
    - (df["LoanAmount"] / 100)
    + np.random.normal(0, 1.2, n)
)
threshold = np.percentile(score, 32)  # ~68% approval rate, matching real dataset
df["Loan_Status"] = np.where(score > threshold, "Y", "N")

# A handful of missing LoanAmount values, like the real dataset
mask = np.random.rand(n) < 0.035
df.loc[mask, "LoanAmount"] = np.nan

df.to_csv("data/loan_data.csv", index=False)
print(f"Synthetic dataset written to data/loan_data.csv  |  shape={df.shape}")
print(df["Loan_Status"].value_counts(normalize=True))
