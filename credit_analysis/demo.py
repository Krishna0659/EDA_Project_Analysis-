"""Generate an explicitly synthetic, correlated monthly customer panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(customers=1500, months=8, seed=42, dirty=True):
    if customers < 30 or not 1 <= months <= 12:
        raise ValueError("Use at least 30 customers and 1-12 months")
    rng = np.random.default_rng(seed)
    n = customers * months
    customer = np.repeat(np.arange(customers), months)
    month = np.tile(np.arange(1, months + 1), customers)
    age_base = rng.integers(21, 71, customers)
    income_base = np.clip(rng.lognormal(np.log(52000), .63, customers), 8000, 350000)
    stress_base = rng.normal(0, 1, customers)
    stress = stress_base[customer] + rng.normal(0, .23, n)
    income = income_base[customer] * (1 + (month - 1) * .0015)
    salary = income / 12 * np.repeat(rng.uniform(.67, .88, customers), months)
    utilization = np.clip(34 + 14 * stress + rng.normal(0, 9, n), 2, 98)
    debt_base = income_base * np.clip(.21 + .16 * stress_base + rng.normal(0, .07, customers), .005, 1.2)
    debt = np.maximum(0, debt_base[customer] * (1 - month * .007) + rng.normal(0, 250, n))
    delays = np.maximum(0, np.rint(10 + 8 * stress + rng.normal(0, 5, n)))
    missed = np.maximum(0, np.rint(3 + 2.1 * stress + rng.normal(0, 1.5, n)))
    history_base = np.array([rng.integers(6, max(7, (age - 18) * 12)) for age in age_base])
    history = history_base[customer] + month - 1
    loans_base = np.clip(rng.poisson(2, customers) + (stress_base > .9), 0, 8)
    loans = loans_base[customer]
    emi = np.where(loans > 0, salary * np.clip(.13 + .07 * stress + rng.normal(0, .035, n), .015, .6), 0)
    # Hidden customer effects plus fresh noise make labels imperfect, but designed.
    latent = (-.9 * stress - .018 * delays - .009 * utilization + .0014 * history
              + np.repeat(rng.normal(0, .45, customers), months) + rng.normal(0, .65, n))
    labels = np.select([latent < -.9, latent > .95], ["Poor", "Good"], default="Standard")
    investment = salary * np.clip(.1 - .025 * stress + rng.normal(0, .025, n), .005, .35)
    expenses = salary * np.clip(.55 + .08 * stress + rng.normal(0, .06, n), .2, .95)
    frame = pd.DataFrame({
        "ID": [f"DEMO_{i:06d}" for i in range(n)],
        "Customer_ID": [f"SYNTH_{i:05d}" for i in customer], "Month": month,
        "Age": age_base[customer],
        "Occupation": rng.choice(["Engineer", "Teacher", "Doctor", "Entrepreneur", "Accountant", "Mechanic", "Writer", "Manager"], customers)[customer],
        "Annual_Income": income.round(2), "Monthly_Inhand_Salary": salary.round(2),
        "Num_Bank_Accounts": rng.integers(1, 7, customers)[customer],
        "Num_Credit_Card": rng.integers(1, 8, customers)[customer],
        "Interest_Rate": np.clip(12 + 4 * stress + rng.normal(0, 2, n), 3, 35).round(2),
        "Num_of_Loan": loans, "Type_of_Loan": np.where(loans > 0, "Personal Loan, Auto Loan", "No loan"),
        "Delay_from_due_date": delays, "Num_of_Delayed_Payment": missed,
        "Changed_Credit_Limit": rng.normal(7, 4, n).round(2),
        "Num_Credit_Inquiries": np.maximum(0, np.rint(3 + stress + rng.normal(0, 1.3, n))),
        "Credit_Mix": np.select([stress > .8, stress < -.5], ["Bad", "Good"], default="Standard"),
        "Outstanding_Debt": debt.round(2), "Credit_Utilization_Ratio": utilization.round(2),
        "Credit_History_Age": [f"{h // 12} Years and {h % 12} Months" for h in history],
        "Payment_of_Min_Amount": np.where(rng.random(n) < 1 / (1 + np.exp(-stress)), "Yes", "No"),
        "Total_EMI_per_month": emi.round(2), "Amount_invested_monthly": investment.round(2),
        "Payment_Behaviour": np.where(stress > .3, "High_spent_Small_value_payments", "Low_spent_Large_value_payments"),
        "Monthly_Balance": (salary - emi - expenses - investment).round(2), "Credit_Score": labels,
    })
    if dirty:
        # Independent corruption, not targeted by credit class; no real identifiers.
        for col in ["Monthly_Inhand_Salary", "Num_Credit_Inquiries", "Credit_History_Age", "Occupation"]:
            frame.loc[rng.choice(n, max(1, n // 50), replace=False), col] = np.nan
        for col in ["Annual_Income", "Num_of_Delayed_Payment"]:
            frame[col] = frame[col].astype(object)
            ix = rng.choice(n, max(1, n // 100), replace=False)
            frame.loc[ix, col] = frame.loc[ix, col].map(lambda v: f"{v}_")
        frame.loc[rng.choice(n, max(1, n // 200), replace=False), "Age"] = -500
        frame.loc[rng.choice(n, max(1, n // 100), replace=False), "Credit_Mix"] = "_"
        frame = pd.concat([frame, frame.sample(n=max(1, n // 500), random_state=seed)], ignore_index=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo/synthetic_credit.csv"))
    parser.add_argument("--customers", type=int, default=1500)
    parser.add_argument("--months", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = generate_demo(args.customers, args.months, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, lineterminator="\n")
    manifest = {"synthetic": True, "generator": "credit_analysis.demo", "seed": args.seed,
                "customers": args.customers, "months": args.months, "rows_including_duplicates": len(frame),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "currency": "unspecified monetary units; not observed customer currency",
                "limitations": "Designed correlations and noisy labels; no real customers, fraud labels, or empirical performance claims."}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(frame):,} SYNTHETIC rows: {args.output}")


if __name__ == "__main__":
    main()
