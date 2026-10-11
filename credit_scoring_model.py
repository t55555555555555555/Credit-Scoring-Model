import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_sample_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic sample financial data for credit evaluation."""
    np.random.seed(random_state)

    age = np.random.randint(21, 68, size=n_samples)
    annual_income = np.random.normal(55000, 18000, size=n_samples).clip(15000, 150000)
    total_debt = np.random.normal(18000, 12000, size=n_samples).clip(500, 80000)
    missed_payments_last_2yrs = np.random.poisson(lam=0.8, size=n_samples)
    open_credit_lines = np.random.randint(1, 12, size=n_samples)
    credit_inquiries_last_6mo = np.random.poisson(lam=1.0, size=n_samples)

    # Risk score calculation for assigning creditworthiness (0 = Default/High Risk, 1 = Creditworthy)
    # Higher income -> better; higher debt, inquiries, and missed payments -> worse
    risk_score = (
        (annual_income / 10000) * 1.2
        - (total_debt / 10000) * 1.5
        - (missed_payments_last_2yrs * 2.2)
        - (credit_inquiries_last_6mo * 0.8)
        + np.random.normal(0, 1.5, size=n_samples)
    )

    creditworthy = (risk_score > np.median(risk_score)).astype(int)

    return pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "total_debt": total_debt,
        "missed_payments": missed_payments_last_2yrs,
        "open_credit_lines": open_credit_lines,
        "credit_inquiries": credit_inquiries_last_6mo,
        "is_creditworthy": creditworthy,
    })


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates domain-specific financial ratios."""
    df_engineered = df.copy()

    # Debt-to-Income (DTI) Ratio: primary lending metric
    df_engineered["debt_to_income_ratio"] = (
        df_engineered["total_debt"] / (df_engineered["annual_income"] + 1e-6)
    )

    # Debt burden per active credit line
    df_engineered["debt_per_credit_line"] = (
        df_engineered["total_debt"] / (df_engineered["open_credit_lines"] + 1)
    )

    # Risk indicator combining missed payments and recent inquiries
    df_engineered["adverse_events_count"] = (
        df_engineered["missed_payments"] + df_engineered["credit_inquiries"]
    )

    return df_engineered


def evaluate_model(name: str, model, X_test, y_test):
    """Outputs accuracy, precision, recall, f1, and roc-auc metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*20} {name} Evaluation {'='*20}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["High Risk (0)", "Creditworthy (1)"]))


def main():
    print("[1/4] Generating financial dataset...")
    raw_df = generate_sample_dataset()

    print("[2/4] Engineering financial indicators...")
    data = engineer_features(raw_df)

    # Split features (X) and label (y)
    X = data.drop(columns=["is_creditworthy"])
    y = data["is_creditworthy"]

    # 80/20 train-test split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling numeric features for algorithms sensitive to magnitude (Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[3/4] Training Models...")

    # 1. Logistic Regression (Baseline linear classifier)
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train_scaled, y_train)

    # 2. Random Forest Classifier (Non-linear ensemble classifier)
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    print("[4/4] Evaluating Performance...")
    evaluate_model("Logistic Regression", lr_model, X_test_scaled, y_test)
    evaluate_model("Random Forest Classifier", rf_model, X_test, y_test)

    # Feature Importance analysis from Random Forest
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Key Driving Features (Random Forest) ---")
    for feat, imp in importances.items():
        print(f"{feat:<25}: {imp:.4f}")


if __name__ == "__main__":
    main()