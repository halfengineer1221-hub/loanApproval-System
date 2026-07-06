"""
==============================================================
 LOAN APPROVAL PREDICTION SYSTEM — Full ML Lifecycle Pipeline
==============================================================
Covers Phases 3-7 of the project methodology:
  Phase 3: Data Preprocessing   (missing values, duplicates, outliers, encoding, scaling)
  Phase 4: EDA                  (univariate, bivariate, correlation — histograms,
                                  boxplots, scatter plots, heatmaps)
  Phase 5: Feature Engineering  (new features + feature selection)
  Phase 6: Model Building       (Logistic Regression, Random Forest, Gradient Boosting)
  Phase 7: Model Evaluation     (accuracy, precision, recall, F1, ROC-AUC, CV)

Phase 8 (Deployment) is a separate file: app.py (Streamlit).

Dataset : Kaggle "Loan Prediction Dataset"
          (columns: Loan_ID, Gender, Married, Dependents, Education,
           Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
           Loan_Amount_Term, Credit_History, Property_Area, Loan_Status)

Usage:
    python loan_approval_pipeline.py --data data/loan_data.csv
"""

import argparse
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RANDOM_STATE = 42
NUMERIC_RAW = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]


# ==================================================================
# PHASE 3a — LOAD, DEDUPLICATE, HANDLE MISSING VALUES
# ==================================================================
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Phase 3] Loaded {path}  ->  shape={df.shape}")

    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # --- Remove duplicates ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Phase 3] Duplicates removed: {before - len(df)}  ->  shape={df.shape}")

    # --- Missing values ---
    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])

    num_cols = ["LoanAmount", "Loan_Amount_Term"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(str).str.replace("3+", "3", regex=False)

    print(f"[Phase 3] Missing values remaining: {df.isna().sum().sum()}")
    return df


# ==================================================================
# PHASE 3b — OUTLIER TREATMENT (IQR capping / winsorization)
# ==================================================================
def treat_outliers(df: pd.DataFrame, cols=("ApplicantIncome", "CoapplicantIncome", "LoanAmount")):
    df = df.copy()
    report = {}
    for c in cols:
        if c not in df.columns:
            continue
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((df[c] < lower) | (df[c] > upper)).sum())
        df[c] = df[c].clip(lower=lower, upper=upper)
        report[c] = {"lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
                      "capped_count": n_outliers}
    print("[Phase 3] Outlier treatment (IQR capping):")
    for c, r in report.items():
        print(f"    {c}: capped {r['capped_count']} values to [{r['lower_bound']}, {r['upper_bound']}]")
    return df, report


# ==================================================================
# PHASE 4 — EXPLORATORY DATA ANALYSIS
# ==================================================================
def run_eda(df: pd.DataFrame, outdir: str = "outputs"):
    # ---- Univariate: numeric histograms ----
    numeric_cols = [c for c in NUMERIC_RAW if c in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#1565C0")
        ax.set_title(f"Univariate: {col} Distribution")
    plt.tight_layout()
    plt.savefig(f"{outdir}/univariate_histograms.png", dpi=120)
    plt.close()

    # ---- Univariate: categorical counts ----
    cat_cols = [c for c in ["Gender", "Married", "Education", "Self_Employed",
                             "Property_Area", "Loan_Status"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), cat_cols):
        sns.countplot(x=df[col], ax=ax, hue=df[col], palette="viridis", legend=False)
        ax.set_title(f"Univariate: {col} Counts")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(f"{outdir}/univariate_categorical.png", dpi=120)
    plt.close()

    # ---- Bivariate: boxplots of numeric features vs Loan_Status ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.boxplot(x="Loan_Status", y=col, data=df, ax=ax, hue="Loan_Status",
                    palette="Set2", legend=False)
        ax.set_title(f"Bivariate: {col} by Loan_Status")
    plt.tight_layout()
    plt.savefig(f"{outdir}/bivariate_boxplots.png", dpi=120)
    plt.close()

    # ---- Bivariate: scatter plot Income vs LoanAmount, colored by status ----
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="ApplicantIncome", y="LoanAmount",
        hue="Loan_Status", palette={"Y": "#2E7D32", "N": "#C62828"}, alpha=0.7
    )
    plt.title("Bivariate: Applicant Income vs Loan Amount")
    plt.tight_layout()
    plt.savefig(f"{outdir}/bivariate_scatter.png", dpi=120)
    plt.close()

    # ---- Bivariate: approval rate by categorical feature ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["Credit_History", "Education", "Property_Area"]):
        if col not in df.columns:
            continue
        rate = df.groupby(col)["Loan_Status"].apply(lambda s: (s == "Y").mean())
        rate.plot(kind="bar", ax=ax, color="#4A148C")
        ax.set_title(f"Approval Rate by {col}")
        ax.set_ylabel("Approval Rate")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(f"{outdir}/bivariate_approval_rates.png", dpi=120)
    plt.close()

    # ---- Correlation heatmap ----
    plt.figure(figsize=(7, 6))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Analysis — Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{outdir}/correlation_heatmap.png", dpi=120)
    plt.close()

    print(f"[Phase 4] EDA plots saved to {outdir}/ "
          f"(univariate_histograms, univariate_categorical, bivariate_boxplots, "
          f"bivariate_scatter, bivariate_approval_rates, correlation_heatmap)")


# ==================================================================
# PHASE 5a — FEATURE ENGINEERING (new features)
# ==================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Avoid div-by-zero
    safe_loan_term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = (df["LoanAmount"] * 1000) / safe_loan_term
    df["EMI"] = df["EMI"].fillna(df["EMI"].median())

    df["Income_to_Loan_Ratio"] = df["Total_Income"] / df["LoanAmount"].replace(0, np.nan)
    df["Income_to_Loan_Ratio"] = df["Income_to_Loan_Ratio"].fillna(df["Income_to_Loan_Ratio"].median())

    df["Balance_Income"] = df["Total_Income"] - df["EMI"]

    # Log-transform skewed monetary features (reduces the influence of outliers/heavy tails)
    for c in ["ApplicantIncome", "CoapplicantIncome", "Total_Income", "LoanAmount"]:
        df[f"Log_{c}"] = np.log1p(df[c])

    print(f"[Phase 5] Feature engineering added: Total_Income, EMI, "
          f"Income_to_Loan_Ratio, Balance_Income, and log-transforms. New shape={df.shape}")
    return df


# ==================================================================
# ENCODING (categorical -> numeric)
# ==================================================================
def encode_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    binary_map_cols = ["Gender", "Married", "Education", "Self_Employed"]
    for c in binary_map_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            encoders[c] = le

    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(int)

    if "Property_Area" in df.columns:
        df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

    target_le = LabelEncoder()
    df["Loan_Status"] = target_le.fit_transform(df["Loan_Status"])  # Y=1, N=0
    encoders["Loan_Status"] = target_le

    return df, encoders


# ==================================================================
# PHASE 5b — FEATURE SELECTION (drop irrelevant, keep top-k by ANOVA F-test)
# ==================================================================
def select_features(X: pd.DataFrame, y: pd.Series, k: int = 10):
    # Drop raw monetary columns once their log/engineered versions exist, to reduce redundancy
    redundant = [c for c in ["ApplicantIncome", "CoapplicantIncome", "Total_Income"] if c in X.columns]
    X_reduced = X.drop(columns=redundant, errors="ignore")

    selector = SelectKBest(score_func=f_classif, k=min(k, X_reduced.shape[1]))
    selector.fit(X_reduced, y)
    scores = pd.Series(selector.scores_, index=X_reduced.columns).sort_values(ascending=False)
    print(f"\n[Phase 5] Dropped redundant raw features: {redundant}")
    print("[Phase 5] ANOVA F-scores (higher = more predictive):")
    print(scores.to_string())

    selected_cols = scores.head(k).index.tolist()
    return selected_cols, scores


# ==================================================================
# PHASE 6 & 7 — MODEL BUILDING, BALANCING, EVALUATION
# ==================================================================
def train_and_evaluate(X: pd.DataFrame, y: pd.Series, outdir: str = "outputs", modeldir: str = "models"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[Phase 6] Split: Train={X_train.shape[0]}  Test={X_test.shape[0]}")
    print(f"[Phase 6] Before SMOTE class balance:\n{y_train.value_counts(normalize=True).round(3)}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"[Phase 6] After SMOTE class balance:\n{pd.Series(y_train_bal).value_counts(normalize=True).round(3)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        cv_scores = cross_val_score(
            model, X_train_bal, y_train_bal,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1"
        )

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        }
        results[name] = metrics

        print(f"\n[Phase 7] === {name} ===")
        for k_, v_ in metrics.items():
            print(f"  {k_:>12}: {v_}")
        print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
        plt.title(f"Confusion Matrix — {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(f"{outdir}/confusion_matrix_{safe_name}.png", dpi=120)
        plt.close()

        RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)

    ax.set_title("ROC Curve Comparison")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.tight_layout()
    plt.savefig(f"{outdir}/roc_curves.png", dpi=120)
    plt.close()

    results_df = pd.DataFrame(results).T.sort_values("f1_score", ascending=False)
    print("\n[Phase 7] === Model Comparison (sorted by F1) ===")
    print(results_df)
    results_df.to_csv(f"{outdir}/model_comparison.csv")

    best_name = results_df.index[0]
    best_model = models[best_name]
    joblib.dump(best_model, f"{modeldir}/best_model.pkl")
    joblib.dump(scaler, f"{modeldir}/scaler.pkl")
    print(f"\n[Phase 7] Saved best model = '{best_name}' -> {modeldir}/best_model.pkl")

    with open(f"{outdir}/results_summary.json", "w") as f:
        json.dump({"best_model": best_name, "metrics": results}, f, indent=2)

    return results_df, best_name, best_model, scaler


# ==================================================================
# MAIN — runs the full lifecycle Phases 3 -> 7
# ==================================================================
def main(data_path: str):
    # Phase 3: preprocessing
    df = load_and_clean(data_path)
    df, outlier_report = treat_outliers(df)

    # Phase 4: EDA (on cleaned, pre-engineered data so raw distributions are visible)
    run_eda(df)

    # Phase 5a: feature engineering
    df = engineer_features(df)

    # Encoding (categorical -> numeric) + scaling happens inside train_and_evaluate
    df_enc, encoders = encode_features(df)
    X_full = df_enc.drop(columns=["Loan_Status"])
    y = df_enc["Loan_Status"]

    # Phase 5b: feature selection
    selected_cols, scores = select_features(X_full, y, k=10)
    X = X_full[selected_cols]

    joblib.dump({"selected_columns": selected_cols, "encoders": encoders,
                 "outlier_report": outlier_report},
                "models/preprocessing_artifacts.pkl")

    # Phase 6 + 7: model building, balancing, evaluation
    results_df, best_name, best_model, scaler = train_and_evaluate(X, y)

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE. Best model: {best_name}")
    print(f"F1 score: {results_df.loc[best_name, 'f1_score']}  |  "
          f"ROC-AUC: {results_df.loc[best_name, 'roc_auc']}")
    print("Outputs saved to ./outputs   |   Model saved to ./models")
    print("Next: run `streamlit run app.py` for Phase 8 (deployment)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Approval Prediction Pipeline")
    parser.add_argument("--data", type=str, default="data/loan_data.csv",
                         help="Path to the Kaggle Loan Prediction CSV")
    args = parser.parse_args()
    main(args.data)
