# Loan Approval Prediction System — Project Report

This report documents the full ML lifecycle for this project, phase by
phase. Phases 3–7 are implemented in `loan_approval_pipeline.py` and
Phase 8 in `app.py`; this document covers Phases 1–2 and summarizes the
rest with references to the code and generated outputs.

---

## Phase 1: Problem Understanding

**Business problem.** Banks and NBFCs receive large volumes of loan
applications and must decide, quickly and consistently, whether to
approve or reject each one. Manual underwriting is slow, and human
decisions can be inconsistent across officers or introduce unconscious
bias. A predictive model that flags likely-approved and likely-rejected
applications lets a loan officer prioritize review time and gives
applicants a faster preliminary answer.

**Objectives.**
1. Build a binary classifier that predicts `Loan_Status` (Approved /
   Rejected) from applicant demographic and financial attributes.
2. Identify which applicant attributes most strongly drive approval
   decisions (interpretability, not just accuracy).
3. Handle the class imbalance inherent in approval data (most
   applications in this dataset are approved, ~69/31 split).
4. Deliver the model as an interactive tool a non-technical user (e.g.
   a loan officer) can use without touching code.

**Success criteria.** F1 score and ROC-AUC are prioritized over raw
accuracy, because the classes are imbalanced and a model that always
predicts "Approved" would score ~69% accuracy while being useless for
catching risky applications.

**Dataset study.** The Kaggle "Loan Prediction Dataset" contains 614
historical applications with these fields:

| Column | Type | Description |
|---|---|---|
| Loan_ID | ID | Unique application identifier (dropped before modeling) |
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0, 1, 2, 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numeric | Monthly income of the primary applicant |
| CoapplicantIncome | Numeric | Monthly income of a co-applicant, if any |
| LoanAmount | Numeric | Requested loan amount, in thousands |
| Loan_Amount_Term | Numeric | Repayment term, in days |
| Credit_History | Binary | 1 = meets credit guidelines, 0 = does not |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| Loan_Status | Target | Y = Approved, N = Rejected |

The dataset carries known missing values (`Gender`, `Married`,
`Dependents`, `Self_Employed`, `Credit_History`, `LoanAmount`,
`Loan_Amount_Term`) and a well-documented, strong relationship between
`Credit_History` and `Loan_Status`, which the EDA and feature-selection
outputs in this project confirm.

---

## Phase 2: Data Collection

- **Source:** Kaggle, "Loan Prediction Dataset" (also listed under
  "Loan Prediction Problem Dataset" by different uploaders — all
  mirrors of the original Analytics Vidhya hackathon dataset carry the
  same 13 columns above).
- **How to obtain it:** search that name on Kaggle, download the
  training CSV (not the unlabeled test.csv, which has no
  `Loan_Status`), and save it as `data/loan_data.csv` in this project.
- **Attribute understanding:** see the table in Phase 1. All fields are
  either applicant demographics (Gender, Married, Dependents,
  Education, Self_Employed, Property_Area), applicant financials
  (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term), or
  a credit signal (Credit_History) — the target is `Loan_Status`.
- **For development/testing without the real file:**
  `generate_sample_data.py` produces a schema-identical synthetic
  dataset with the same approval rate (~69%) so the pipeline can be
  run and verified before the real Kaggle file is available.

---

## Phase 3: Data Preprocessing — implemented in `load_and_clean()` and `treat_outliers()`

- **Missing values:** categorical columns filled with the mode,
  numeric columns (`LoanAmount`, `Loan_Amount_Term`) filled with the
  median — chosen over mean because both are right-skewed.
- **Duplicates:** `df.drop_duplicates()` applied after loading; the
  pipeline prints how many rows were removed.
- **Outliers:** IQR capping (winsorization) on `ApplicantIncome`,
  `CoapplicantIncome`, and `LoanAmount` — values beyond
  Q1 − 1.5·IQR / Q3 + 1.5·IQR are clipped rather than dropped, to avoid
  losing otherwise-valid applications. The capped bounds and count are
  logged and saved into `models/preprocessing_artifacts.pkl`.
- **Encoding:** `LabelEncoder` for binary categoricals (Gender,
  Married, Education, Self_Employed), integer cast for Dependents, and
  one-hot encoding (`drop_first=True`) for `Property_Area`.
- **Normalization:** `StandardScaler` fit on the training split only,
  applied to train and test — done at model-training time so scaling
  parameters never leak test-set information.

## Phase 4: EDA — implemented in `run_eda()`

Generates, in `outputs/`:
- `univariate_histograms.png` — distribution of each numeric feature
- `univariate_categorical.png` — count plots for each categorical feature
- `bivariate_boxplots.png` — numeric features split by Loan_Status
- `bivariate_scatter.png` — ApplicantIncome vs LoanAmount, colored by outcome
- `bivariate_approval_rates.png` — approval rate by Credit_History /
  Education / Property_Area
- `correlation_heatmap.png` — correlation across numeric features

## Phase 5: Feature Engineering — implemented in `engineer_features()` and `select_features()`

New features created:
- `Total_Income` = ApplicantIncome + CoapplicantIncome
- `EMI` = LoanAmount × 1000 / Loan_Amount_Term (approximate monthly installment)
- `Income_to_Loan_Ratio` = Total_Income / LoanAmount
- `Balance_Income` = Total_Income − EMI (disposable income after loan payment)
- `Log_*` transforms of the skewed monetary columns

Feature selection: raw income columns are dropped once their
log/engineered equivalents exist (redundancy reduction), then the top
10 remaining features are kept by ANOVA F-test score
(`SelectKBest(f_classif)`). `Credit_History` dominates every run.

## Phase 6 & 7: Model Building and Evaluation — implemented in `train_and_evaluate()`

Three classifiers are trained on SMOTE-balanced training data and
evaluated on an untouched, stratified 20% test split:
- Logistic Regression
- Random Forest
- Gradient Boosting

Metrics reported per model: accuracy, precision, recall, F1, ROC-AUC,
and 5-fold cross-validated F1 (mean ± std). All are saved to
`outputs/model_comparison.csv` and `outputs/results_summary.json`; the
model with the best test F1 is saved to `models/best_model.pkl`.

## Phase 8: Model Deployment — implemented in `app.py`

A Streamlit app (`streamlit run app.py`) that:
- collects applicant details through a form (income, loan amount,
  credit history, education, etc.)
- reproduces the exact preprocessing and feature-engineering steps used
  in training
- loads the saved model, scaler, and encoders
- displays an Approved/Rejected verdict with the model's confidence,
  plus the engineered feature values used for that prediction

---

## Reproducing this project end-to-end

```bash
pip install -r requirements.txt

# If you don't have the real Kaggle file yet, generate a test one:
python generate_sample_data.py

# Otherwise, place the real Kaggle CSV at data/loan_data.csv, then:
python loan_approval_pipeline.py --data data/loan_data.csv

# Launch the deployed app:
streamlit run app.py
```
