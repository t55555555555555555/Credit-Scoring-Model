import os
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


def load_and_prepare_csv(file_path: str) -> pd.DataFrame:
    """Loads dataset from CSV and maps common column names."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File '{file_path}' not found. Ensure it is placed in the project folder."
        )

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")

    # OPTIONAL: Map your actual CSV column names to standardized names below.
    # Change the left side to match the exact headers in your CSV file.
    column_mapping = {
        "Age": "age",
        "Annual_Income": "annual_income",
        "Total_Debt": "total_debt",
        "Num_Past_Due": "missed_payments",
        "Open_Accounts": "open_credit_lines",
        "Recent_Inquiries": "credit_inquiries",
        "Target": "is_creditworthy",  # 0 for bad/default, 1 for good credit
    }

    # Rename only columns that exist in the CSV
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Drop rows missing the target variable
    if "is_creditworthy" in df.columns:
        df = df.dropna(subset=["is_creditworthy"])
    else:
        raise KeyError(
            "Target column 'is_creditworthy' was not found. Please map your label column."
        )

    # Impute missing values in numeric feature columns with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates financial ratios for credit assessment."""
    df_engineered = df.copy()

    # Debt-to-Income (DTI)
    df_engineered["debt_to_income_ratio"] = (
        df_engineered["total_debt"] / (df_engineered["annual_income"] + 1e-6)
    )

    # Debt per open line
    df_engineered["debt_per_credit_line"] = (
        df_engineered["total_debt"] / (df_engineered["open_credit_lines"] + 1)
    )

    # Combined adverse events
    df_engineered["adverse_events_count"] = (
        df_engineered["missed_payments"] + df_engineered["credit_inquiries"]
    )

    return df_engineered


def evaluate_model(name: str, model, X_test, y_test):
    """Outputs performance evaluation metrics."""
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
    csv_file = "credit_data.csv"

    print("[1/4] Loading dataset from CSV...")
    raw_df = load_and_prepare_csv(csv_file)

    print("[2/4] Engineering financial indicators...")
    data = engineer_features(raw_df)

    # Ensure all required features are present
    feature_cols = [
        "age",
        "annual_income",
        "total_debt",
        "missed_payments",
        "open_credit_lines",
        "credit_inquiries",
        "debt_to_income_ratio",
        "debt_per_credit_line",
        "adverse_events_count",
    ]

    X = data[feature_cols]
    y = data["is_creditworthy"].astype(int)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale numeric features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[3/4] Training Models...")

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train_scaled, y_train)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    print("[4/4] Evaluating Performance...")
    evaluate_model("Logistic Regression", lr_model, X_test_scaled, y_test)
    evaluate_model("Random Forest Classifier", rf_model, X_test, y_test)

    # Feature Importance
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Key Driving Features (Random Forest) ---")
    for feat, imp in importances.items():
        print(f"{feat:<25}: {imp:.4f}")


if __name__ == "__main__":
    main()