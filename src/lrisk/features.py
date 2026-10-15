"""Lending Club cleaning + feature engineering, shared by training and live scoring.

Only fields known at application time are used. Post-origination columns
(total_pymnt, recoveries, last_pymnt_d, ...) are excluded because they leak the outcome.
The lender-assigned grade / sub_grade / int_rate are excluded too: they are the lender's
own prior risk opinion, and the point here is to learn risk from the applicant's profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "id"
LABEL_COL = "loan_status"
DATE_COL = "issue_d"

NUMERIC_RAW = [
    "loan_amnt", "installment", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "fico_range_low", "fico_range_high",
]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state", "application_type"]
STRING_RAW = ["term", "emp_length", "earliest_cr_line"]

# Columns that travel through Kafka / HDFS.
RAW_COLUMNS = [ID_COL, DATE_COL, LABEL_COL] + NUMERIC_RAW + CATEGORICAL + STRING_RAW

ENGINEERED = [
    "term_months", "emp_years", "fico", "credit_history_years",
    "payment_to_income", "loan_to_income", "revol_bal_to_income",
]
FEATURES = (
    [c for c in NUMERIC_RAW if c not in ("fico_range_low", "fico_range_high")]
    + ENGINEERED + CATEGORICAL
)

DEFAULT_STATUSES = ("Charged Off", "Default")
PAID_STATUSES = ("Fully Paid",)

# Human-readable names for the underwriter-facing reason codes.
FEATURE_LABELS = {
    "loan_amnt": "Loan amount", "installment": "Monthly installment", "annual_inc": "Annual income",
    "dti": "Debt-to-income ratio", "delinq_2yrs": "Delinquencies (2y)", "inq_last_6mths": "Credit inquiries (6m)",
    "open_acc": "Open accounts", "pub_rec": "Public records", "revol_bal": "Revolving balance",
    "revol_util": "Revolving utilisation %", "total_acc": "Total accounts", "mort_acc": "Mortgage accounts",
    "pub_rec_bankruptcies": "Bankruptcies", "term_months": "Loan term (months)", "emp_years": "Employment (years)",
    "fico": "FICO score", "credit_history_years": "Credit history (years)",
    "payment_to_income": "Annual payments / income", "loan_to_income": "Loan / income",
    "revol_bal_to_income": "Revolving balance / income", "home_ownership": "Home ownership",
    "verification_status": "Income verification", "purpose": "Loan purpose", "addr_state": "State",
    "application_type": "Application type",
}


def _num(s: pd.Series) -> pd.Series:
    """Numeric parse that tolerates '13.5%' style strings."""
    if s.dtype == object or str(s.dtype).startswith("str"):
        s = s.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _month_year(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def parse_emp_length(s: pd.Series) -> pd.Series:
    t = s.astype("object").astype(str).str.lower()
    out = t.str.extract(r"(\d+)")[0].astype(float)
    out[t.str.contains("<")] = 0.0
    return out


def make_target(status: pd.Series) -> pd.Series:
    """1 = defaulted, 0 = fully paid, NaN = outcome not yet known (Current, Late, ...)."""
    s = status.astype("object").astype(str).str.replace(
        r"^Does not meet the credit policy\. Status:", "", regex=True).str.strip()
    y = pd.Series(np.nan, index=status.index)
    y[s.isin(DEFAULT_STATUSES)] = 1.0
    y[s.isin(PAID_STATUSES)] = 0.0
    return y


def build_features(raw: pd.DataFrame, categories: dict[str, list[str]] | None = None,
                   now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Raw Lending Club-style rows -> model matrix (columns == FEATURES).

    `categories` fixes the category vocabulary (taken from training) so scoring is consistent;
    unseen values become NaN, which LightGBM treats as missing.
    """
    df = raw.copy()
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in NUMERIC_RAW:
        df[c] = _num(df[c])

    issue = _month_year(df[DATE_COL])
    if now is not None:
        issue = issue.fillna(now)
    earliest = _month_year(df["earliest_cr_line"])

    out = pd.DataFrame(index=df.index)
    for c in NUMERIC_RAW:
        if c not in ("fico_range_low", "fico_range_high"):
            out[c] = df[c]
    out["term_months"] = _num(df["term"].astype("object").astype(str).str.extract(r"(\d+)")[0])
    out["emp_years"] = parse_emp_length(df["emp_length"])
    out["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    out["credit_history_years"] = (issue - earliest).dt.days / 365.25
    inc = df["annual_inc"].where(df["annual_inc"] > 0)
    out["payment_to_income"] = df["installment"] * 12 / inc
    out["loan_to_income"] = df["loan_amnt"] / inc
    out["revol_bal_to_income"] = df["revol_bal"] / inc

    for c in CATEGORICAL:
        vals = df[c].astype("object").where(df[c].notna(), None)
        vals = vals.map(lambda v: None if v is None else str(v).strip())
        if categories is not None:
            known = set(categories[c])
            out[c] = pd.Categorical(vals.where(vals.isin(known), None), categories=categories[c])
        else:
            out[c] = vals.astype("category")
    return out[FEATURES]


def category_vocab(X: pd.DataFrame) -> dict[str, list[str]]:
    return {c: [str(v) for v in X[c].cat.categories] for c in CATEGORICAL}
