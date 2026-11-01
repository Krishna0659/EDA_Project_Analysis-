"""Deterministic, row-local cleaning; learned imputation belongs inside training."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

LABELS = ["Poor", "Standard", "Good"]
NUMERIC = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]
CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
DERIVED = ["Debt_to_Annual_Income", "EMI_to_Monthly_Salary"]
FEATURES = NUMERIC + CATEGORICAL + DERIVED
PLACEHOLDERS = {"", "_", "_______", "!@9#%8", "nan", "None", "NA", "N/A", "null"}
MONTHS = {name: i + 1 for i, name in enumerate([
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
])}


def history_months(value, numeric_unit="months"):
    """Text 'X Years and Y Months' or numeric months (years only when explicit)."""
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)\s+Years?\s+and\s+(\d+)\s+Months?\s*", str(value), re.I)
    if match:
        years, months = map(int, match.groups())
        return years * 12 + months if months < 12 else np.nan
    try:
        result = float(str(value).strip().strip("_"))
        return result * (12 if numeric_unit == "years" else 1)
    except ValueError:
        return np.nan


def clean_data(raw: pd.DataFrame, history_unit: str = "months"):
    """Return a cleaned copy and an audit without imputing or deleting outliers."""
    if history_unit not in {"months", "years"}:
        raise ValueError("history_unit must be months or years")
    required = set(NUMERIC + CATEGORICAL + ["Customer_ID", "Month", "Credit_Score"])
    absent = sorted(required - set(raw.columns))
    if absent:
        raise ValueError(f"Missing required columns: {', '.join(absent)}")
    if raw.empty:
        raise ValueError("Input data is empty")
    df = raw.copy(deep=True)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    # Never include direct identifiers in public aggregate outputs or features.
    df = df.drop(columns=["Name", "SSN"], errors="ignore")
    audit = {"input_rows": len(raw), "exact_duplicates_removed": duplicates, "clean_rows": len(df),
             "history_numeric_unit": history_unit, "columns": {}}
    for col in required:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = df[col].replace(list(PLACEHOLDERS), np.nan)
    if df["Customer_ID"].isna().any():
        raise ValueError("Customer_ID is missing; a leakage-safe customer split is impossible")
    unexpected = set(df["Credit_Score"].dropna().unique()) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unsupported Credit_Score labels: {sorted(map(str, unexpected))}")
    for col in NUMERIC:
        before_missing = int(df[col].isna().sum())
        if col == "Credit_History_Age":
            parsed = df[col].map(lambda value: history_months(value, history_unit))
        else:
            parsed = pd.to_numeric(df[col].astype(str).str.strip().str.strip("_").str.replace(",", "", regex=False), errors="coerce")
        parsed = parsed.astype(float).replace([np.inf, -np.inf], np.nan)
        parse_failures = int(parsed.isna().sum()) - before_missing
        invalid = pd.Series(False, index=df.index)
        if col == "Age":
            invalid = (parsed < 18) | (parsed > 100)
        elif col in {"Annual_Income", "Monthly_Inhand_Salary"}:
            invalid = parsed <= 0
        elif col in {"Credit_Utilization_Ratio", "Interest_Rate"}:
            invalid = (parsed < 0) | (parsed > 100)
        elif col not in {"Monthly_Balance", "Changed_Credit_Limit", "Delay_from_due_date"}:
            invalid = parsed < 0
        if col.startswith("Num_"):
            invalid |= parsed.notna() & (parsed % 1 != 0)
        if col == "Credit_History_Age":
            invalid |= parsed > df["Age"] * 12
        df[col] = parsed.mask(invalid)
        audit["columns"][col] = {"missing_before": before_missing, "parse_failures": parse_failures,
                                  "domain_invalid": int(invalid.sum()), "missing_after": int(df[col].isna().sum())}
    allowed = {"Credit_Mix": {"Bad", "Standard", "Good"},
               "Payment_of_Min_Amount": {"Yes", "No", "NM"},
               "Payment_Behaviour": {f"{spend}_spent_{value}_value_payments" for spend in ["Low", "High"] for value in ["Small", "Medium", "Large"]}}
    for col in CATEGORICAL:
        invalid = df[col].notna() & ~df[col].isin(allowed[col]) if col in allowed else pd.Series(False, index=df.index)
        df[col] = df[col].mask(invalid).astype(object).where(df[col].notna() & ~invalid, np.nan)
        audit["columns"][col] = {"domain_invalid": int(invalid.sum()), "missing_after": int(df[col].isna().sum())}
    month_values = df["Month"].replace(MONTHS)
    df["Month"] = pd.to_numeric(month_values, errors="coerce")
    if (~df["Month"].isin(range(1, 13))).any():
        raise ValueError("Month must be an English month name or integer 1-12")
    if df.duplicated(["Customer_ID", "Month"]).any():
        raise ValueError("Conflicting customer-month rows; resolve duplicates or provide a dated schema")
    df["Debt_to_Annual_Income"] = (df["Outstanding_Debt"] / df["Annual_Income"]).replace([np.inf, -np.inf], np.nan)
    df["EMI_to_Monthly_Salary"] = (df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"]).replace([np.inf, -np.inf], np.nan)
    audit["customers"] = int(df["Customer_ID"].nunique())
    audit["unlabeled_rows"] = int(df["Credit_Score"].isna().sum())
    audit["missing_cells_after"] = int(df[FEATURES].isna().sum().sum())
    return df, audit
