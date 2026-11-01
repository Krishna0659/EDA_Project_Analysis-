# Data dictionary

The source notebook describes 28 columns. The generated demonstration has **26**:
`Name` and `SSN` are intentionally absent. Cleaning adds two ratios, resulting in
28 cleaned columns. Each row is one customer in one month of a single year.
No real person's identity or private information is used.

| Field | Meaning / unit | Handling |
| --- | --- | --- |
| ID | Synthetic observation ID | Not modeled |
| Customer_ID | Synthetic customer identifier | Required for grouping; not modeled |
| Month | Integer 1–12; English month names also accepted | Used for trend plots; not modeled |
| Age | Age in years | 18–100; outside becomes missing |
| Occupation | Occupation category | Missing retained for EDA; Unknown during modeling |
| Annual_Income | Annual income, unspecified monetary units | Positive; trailing underscores and commas parsed |
| Monthly_Inhand_Salary | Net monthly salary, same monetary units | Positive |
| Num_Bank_Accounts | Account count | Nonnegative integer |
| Num_Credit_Card | Card count | Nonnegative integer |
| Interest_Rate | Percentage | 0–100 |
| Num_of_Loan | Loan count | Nonnegative integer |
| Type_of_Loan | Illustrative loan-description text | Retained in cleaned data; excluded from features |
| Delay_from_due_date | Payment delay in days | Negative values may represent early payments |
| Num_of_Delayed_Payment | Delayed payment count | Nonnegative integer |
| Changed_Credit_Limit | Source-defined credit-limit change | Signed numeric; unit not documented upstream |
| Num_Credit_Inquiries | Inquiry count | Nonnegative integer |
| Credit_Mix | Bad / Standard / Good | Unexpected values become missing |
| Outstanding_Debt | Debt balance, monetary units | Nonnegative |
| Credit_Utilization_Ratio | Credit used as a percentage of available credit | 0–100 |
| Credit_History_Age | Credit history in months after cleaning | Parses “X Years and Y Months”; numeric default months |
| Payment_of_Min_Amount | Yes / No / NM | NM retained as a separate category; meaning not inferred |
| Total_EMI_per_month | Monthly installment payments, monetary units | Nonnegative |
| Amount_invested_monthly | Monthly investment, monetary units | Nonnegative |
| Payment_Behaviour | Low/High spending × Small/Medium/Large payment category | Unexpected values become missing |
| Monthly_Balance | Monthly cash balance, monetary units | Signed numeric; negative balances retained |
| Credit_Score | Poor / Standard / Good | Target; unknown labels rejected; missing excluded from modeling |
| Debt_to_Annual_Income | Outstanding debt / annual income | Derived; missing denominator stays missing |
| EMI_to_Monthly_Salary | Monthly installments / net salary | Derived; missing denominator stays missing |

`Name` and `SSN`, if supplied, are dropped by cleaning. Direct identifiers are
never model features. Unknown extra columns are retained in memory but not
modeled or exported. Only aggregate outputs are written by the analysis command.

## Input contract

- All 17 numeric fields, four modeled categories, Customer_ID, Month and
  Credit_Score are required. ID and Type_of_Loan are optional.
- Exact duplicate rows are removed. Conflicting rows for the same customer and
  month are rejected. Multi-year datasets require a dated schema extension.
- Missing Customer_ID is rejected because safe customer separation is impossible.
- Numeric credit history has ambiguous units in the source; confirm them and use
  `--history-unit years` if your numeric values are years. Text values have explicit units.
- The evaluation requires at least 30 labeled customers and every score class in
  each deterministic partition. If this condition fails, it stops with a clear error.
- There is no arbitrary percentile trimming. Unusual but valid monetary values
  remain visible; inspect distribution charts before interpreting another dataset.
