import numpy as np
import pandas as pd
import pytest

from credit_analysis.data import clean_data, history_months
from credit_analysis.demo import generate_demo


@pytest.fixture
def raw():
    return generate_demo(customers=60, months=3, dirty=False)


def test_demo_is_deterministic_and_contains_no_personal_identifiers():
    first = generate_demo(60, 3)
    pd.testing.assert_frame_equal(first, generate_demo(60, 3))
    assert not {"Name", "SSN"} & set(first.columns)
    assert first.Customer_ID.str.startswith("SYNTH_").all()
    assert first.duplicated().sum() > 0


def test_cleaning_preserves_input_and_audits_duplicates(raw):
    original = raw.copy(deep=True)
    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    cleaned, audit = clean_data(duplicated)
    assert len(cleaned) == len(raw)
    assert audit["exact_duplicates_removed"] == 1
    pd.testing.assert_frame_equal(raw, original)


def test_parse_invalid_and_zero_denominators(raw):
    raw["Annual_Income"] = raw.Annual_Income.astype(object)
    raw.loc[0, "Annual_Income"] = "50,000_"
    raw.loc[1, "Annual_Income"] = "bad"
    raw.loc[2, "Annual_Income"] = 0
    raw.loc[3, "Age"] = -500
    raw.loc[4, "Monthly_Inhand_Salary"] = 0
    raw.loc[5, "Credit_Utilization_Ratio"] = 101
    raw.loc[6, "Outstanding_Debt"] = np.inf
    cleaned, audit = clean_data(raw)
    assert cleaned.loc[0, "Annual_Income"] == 50000
    assert cleaned.loc[[1, 2], "Debt_to_Annual_Income"].isna().all()
    assert np.isnan(cleaned.loc[4, "EMI_to_Monthly_Salary"])
    assert np.isnan(cleaned.loc[3, "Age"])
    assert np.isnan(cleaned.loc[5, "Credit_Utilization_Ratio"])
    assert np.isnan(cleaned.loc[6, "Outstanding_Debt"])
    assert audit["columns"]["Annual_Income"]["parse_failures"] == 1
    assert audit["columns"]["Annual_Income"]["domain_invalid"] == 1


@pytest.mark.parametrize("value,unit,expected", [("5 Years and 3 Months", "months", 63), ("1 Year and 1 Month", "years", 13), (5, "months", 5), (5, "years", 60)])
def test_credit_history_units(value, unit, expected):
    assert history_months(value, unit) == expected


def test_bad_history_does_not_become_zero():
    assert np.isnan(history_months("5 Years and 15 Months"))
    assert np.isnan(history_months("unknown"))


def test_schema_and_target_errors_are_actionable(raw):
    with pytest.raises(ValueError, match="Missing required columns.*Annual_Income"):
        clean_data(raw.drop(columns="Annual_Income"))
    raw.loc[0, "Credit_Score"] = "Fraud"
    with pytest.raises(ValueError, match="Unsupported Credit_Score"):
        clean_data(raw)


def test_conflicting_customer_month_is_rejected(raw):
    conflict = raw.iloc[[0]].copy()
    conflict["Annual_Income"] = 123456
    with pytest.raises(ValueError, match="Conflicting customer-month"):
        clean_data(pd.concat([raw, conflict], ignore_index=True))


def test_missing_customer_id_is_rejected(raw):
    raw.loc[0, "Customer_ID"] = np.nan
    with pytest.raises(ValueError, match="Customer_ID is missing"):
        clean_data(raw)


def test_categories_months_and_identifier_removal(raw):
    raw["Name"] = "DO NOT EXPORT"
    raw["SSN"] = "DO NOT EXPORT"
    raw["Month"] = raw.Month.astype(object)
    raw.loc[0, "Month"] = "January"
    raw.loc[0, "Credit_Mix"] = "unknown_category"
    cleaned, audit = clean_data(raw)
    assert not {"Name", "SSN"} & set(cleaned.columns)
    assert cleaned.loc[0, "Month"] == 1
    assert pd.isna(cleaned.loc[0, "Credit_Mix"])
    assert audit["columns"]["Credit_Mix"]["domain_invalid"] == 1


def test_high_income_and_negative_balance_are_preserved(raw):
    raw.loc[0, "Annual_Income"] = 1000000
    raw.loc[0, "Monthly_Balance"] = -100
    cleaned, _ = clean_data(raw)
    assert cleaned.loc[0, "Annual_Income"] == 1000000
    assert cleaned.loc[0, "Monthly_Balance"] == -100
