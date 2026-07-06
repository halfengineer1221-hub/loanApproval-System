# Loan Approval Prediction System

A complete, ready-to-run ML pipeline that predicts whether a loan application
should be **approved or rejected**, built around the Kaggle *Loan Prediction
Dataset*.

This project follows the complete ML lifecycle (Problem Understanding →
Data Collection → Preprocessing → EDA → Feature Engineering → Model
Building → Evaluation → Deployment). See `PROJECT_REPORT.md` for the
full phase-by-phase write-up.

## What it covers

| Phase | Where it happens |
|---|---|
| **1. Problem Understanding** | `PROJECT_REPORT.md` |
| **2. Data Collection** | `PROJECT_REPORT.md`, `generate_sample_data.py` |
| **3. Preprocessing** — missing values, duplicates, outliers (IQR capping), encoding, scaling | `loan_approval_pipeline.py` → `load_and_clean()`, `treat_outliers()` |
| **4. EDA** — univariate histograms/countplots, bivariate boxplots/scatter/approval-rate bars, correlation heatmap | `loan_approval_pipeline.py` → `run_eda()` |
| **5. Feature Engineering** — Total_Income, EMI, Income_to_Loan_Ratio, Balance_Income, log-transforms, ANOVA feature selection | `loan_approval_pipeline.py` → `engineer_features()`, `select_features()` |
| **6. Model Building** — Logistic Regression, Random Forest, Gradient Boosting | `loan_approval_pipeline.py` → `train_and_evaluate()` |
| **7. Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold CV, confusion matrices | `loan_approval_pipeline.py` → `train_and_evaluate()` |
| **8. Deployment** — Streamlit app, accepts user inputs, shows prediction | `app.py` |
| **Data Balancing** | `SMOTE` oversampling on the training set only (Loan_Status is ~69/31 in the real dataset) |

## Project structure

```
loan_project/
├── PROJECT_REPORT.md           # Phase 1-2 write-up + summary of all phases
├── loan_approval_pipeline.py   # Phases 3-7 — run this first
├── app.py                      # Phase 8 — Streamlit deployment, run second
├── generate_sample_data.py     # makes a synthetic test CSV (schema-matched)
├── requirements.txt
├── data/
│   └── loan_data.csv           # <- put the real Kaggle CSV here
├── outputs/                    # EDA plots, confusion matrices, ROC curve, metrics
└── models/                     # saved model + scaler + encoders (.pkl)
```

## Setup

```bash
pip install -r requirements.txt
```

## Get the real dataset

1. Download **"Loan Prediction Dataset"** / **"Loan Prediction Problem
   Dataset"** from Kaggle (search that exact name — there are a couple of
   near-identical uploads, any of them work as long as the columns match
   the table below).
2. Save it as `data/loan_data.csv`. Expected columns:

   `Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
   ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
   Credit_History, Property_Area, Loan_Status`

   If your file only has training data with a `Loan_Status` column, that's
   the one you want (Kaggle also ships an unlabeled test.csv — don't use
   that one for training).

### Don't have the file yet? Test the pipeline right now

```bash
python generate_sample_data.py       # writes data/loan_data.csv (synthetic, schema-matched)
python loan_approval_pipeline.py --data data/loan_data.csv
```

This lets you confirm everything runs before swapping in the real Kaggle
file — just overwrite `data/loan_data.csv` with the real one afterward and
re-run.

## Run it

```bash
python loan_approval_pipeline.py --data data/loan_data.csv
```

Then launch the deployed prediction app (Phase 8):

```bash
streamlit run app.py
```

This opens a browser form where you enter applicant details (income,
loan amount, credit history, etc.) and get an instant Approved/Rejected
prediction with the model's confidence.

## What you get afterward

- `outputs/eda_overview.png` — class balance + income distribution
- `outputs/correlation_heatmap.png` — numeric feature correlations
- `outputs/confusion_matrix_*.png` — one per model
- `outputs/roc_curves.png` — all three models overlaid
- `outputs/model_comparison.csv` — metrics table
- `outputs/results_summary.json` — best model + all metrics
- `models/best_model.pkl` — the top-performing model, ready to load with `joblib.load()`
- `models/scaler.pkl`, `models/preprocessing_artifacts.pkl` — needed to preprocess new applicants the same way before predicting

## Notes on real-world results

On the actual Kaggle dataset, **Credit_History** is by a wide margin the
single strongest predictor — this is a well-known property of this
dataset, not a bug. Random Forest or Gradient Boosting typically edge out
Logistic Regression on F1/recall, though Logistic Regression is the easiest
to explain to a loan officer (coefficients map directly to "this factor
raises/lowers approval odds by X").

## Extending this project

- Swap `SelectKBest` for `RFECV` or feature importances from a fitted tree model for a second opinion on feature selection.
- Try `RandomizedSearchCV` for hyperparameter tuning on the best model.
- Add SHAP values to explain individual predictions (important for loan decisions, which often require explainability for compliance reasons).
- Wrap `models/best_model.pkl` in a small Flask/FastAPI endpoint for a live demo.
