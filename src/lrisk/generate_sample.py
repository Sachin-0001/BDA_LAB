"""Generate a SYNTHETIC dataset in the Lending Club accepted-loans raw format.

This is NOT real data. It exists so the pipeline can be run end to end before the Kaggle
file is downloaded. Default risk is driven by FICO, DTI, income, revolving utilisation,
delinquencies, term and loan size *jointly* (the point of the project), plus noise.

    python -m lrisk.generate_sample --rows 60000 --out data/sample_lending_club.csv
"""
import argparse

import numpy as np
import pandas as pd

STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "AZ", "WA", "MA"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house"]
EMP = ["< 1 year", "1 year"] + [f"{i} years" for i in range(2, 10)] + ["10+ years"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def generate(rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = rows
    fico_low = np.clip(rng.normal(692, 32, n), 625, 845).round(-0).astype(int) // 5 * 5
    inc = np.exp(rng.normal(11.1, 0.55, n)).round(0)
    amnt = (np.clip(np.exp(rng.normal(9.5, 0.6, n)), 1000, 40000) // 25 * 25)
    term = rng.choice([36, 60], n, p=[0.72, 0.28])
    rate = np.clip(26 - (fico_low - 620) * 0.065 + (term == 60) * 2.5 + rng.normal(0, 1.5, n), 5.3, 30.9)
    r = rate / 1200
    installment = (amnt * r / (1 - (1 + r) ** -term)).round(2)
    dti = np.clip(rng.gamma(4.0, 4.4, n) + (amnt / inc) * 8, 0, 60).round(2)
    revol_util = np.clip(rng.normal(52, 25, n), 0, 130).round(1)
    revol_bal = (inc * np.clip(rng.normal(0.18, 0.12, n), 0.005, None)).round(0)
    delinq = rng.poisson(np.clip(0.6 - (fico_low - 650) * 0.004, 0.05, 1.5))
    inq = rng.poisson(0.9, n)
    pub_rec = rng.poisson(0.12, n)
    bankr = np.minimum(pub_rec, rng.poisson(0.08, n))
    open_acc = np.clip(rng.poisson(11, n), 1, 40)
    total_acc = open_acc + rng.poisson(12, n)
    mort = rng.poisson(1.2, n)
    hist_years = np.clip(rng.gamma(5, 3.0, n), 1, 55)

    year = rng.choice(np.arange(2012, 2019), n, p=[.04, .10, .17, .24, .20, .15, .10])
    month = rng.integers(0, 12, n)
    issue_ts = pd.to_datetime({"year": year, "month": month + 1, "day": 1})
    earliest_ts = issue_ts - pd.to_timedelta((hist_years * 365.25).astype(int), unit="D")

    emp_idx = rng.integers(0, len(EMP), n)
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[.40, .50, .10])
    verif = rng.choice(["Verified", "Source Verified", "Not Verified"], n, p=[.32, .38, .30])
    purpose = rng.choice(PURPOSES, n, p=[.57, .2, .06, .06, .03, .02, .02, .01, .01, .01, .01])

    # Joint risk model: effects interact (high DTI hurts more at low FICO / low income).
    z = (-2.15
         - 0.020 * (fico_low - 690)
         + 0.030 * (dti - 18)
         + 0.55 * (term == 60)
         + 0.65 * (installment * 12 / inc) * 3
         + 0.004 * (revol_util - 50)
         + 0.20 * delinq + 0.15 * np.minimum(inq, 4) + 0.18 * pub_rec
         + 0.0006 * np.maximum(dti - 25, 0) * np.maximum(700 - fico_low, 0)
         - 0.012 * np.minimum(hist_years, 30)
         + 0.35 * (purpose == "small_business") + 0.10 * (purpose == "medical")
         - 0.10 * (home == "MORTGAGE") + 0.10 * (verif == "Not Verified")
         + rng.normal(0, 0.9, n))
    p_default = 1 / (1 + np.exp(-z))
    defaulted = rng.random(n) < p_default
    status = np.where(defaulted, "Charged Off", "Fully Paid")
    # Recent loans have no resolved outcome yet.
    status = np.where((year == 2018) & (rng.random(n) < 0.7), "Current", status)

    df = pd.DataFrame({
        "id": np.arange(10_000_000, 10_000_000 + n).astype(str),
        "loan_amnt": amnt, "term": [f" {t} months" for t in term], "installment": installment,
        "emp_length": [EMP[i] for i in emp_idx], "home_ownership": home, "annual_inc": inc,
        "verification_status": verif,
        "issue_d": [f"{MONTHS[m]}-{y}" for m, y in zip(month, year)],
        "loan_status": status, "purpose": purpose, "addr_state": rng.choice(STATES, n),
        "dti": dti, "delinq_2yrs": delinq,
        "earliest_cr_line": [f"{MONTHS[d.month - 1]}-{d.year}" for d in earliest_ts],
        "fico_range_low": fico_low, "fico_range_high": fico_low + 4,
        "inq_last_6mths": inq, "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
        "revol_util": revol_util, "total_acc": total_acc, "mort_acc": mort,
        "pub_rec_bankruptcies": bankr, "application_type": "Individual",
    })
    # Realistic missingness.
    for col, frac in [("emp_length", .06), ("revol_util", .005), ("mort_acc", .03), ("dti", .002)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--out", default="data/sample_lending_club.csv")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    d = generate(a.rows, a.seed)
    d.to_csv(a.out, index=False)
    res = d[d.loan_status.isin(["Charged Off", "Fully Paid"])]
    print(f"wrote {len(d)} rows to {a.out}; default rate among resolved: {(res.loan_status == 'Charged Off').mean():.1%}")
