import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from credit_analysis.data import FEATURES, NUMERIC, clean_data
from credit_analysis.demo import generate_demo
from credit_analysis.model import customer_split, estimator


def test_partitions_are_disjoint_complete_and_repeatable():
    data, _ = clean_data(generate_demo(150, 4))
    split = customer_split(data)
    repeat = customer_split(data)
    groups = {key: set(data.iloc[ix].Customer_ID) for key, ix in split.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["validation"]
    assert not groups["validation"] & groups["test"]
    assert len(set(np.concatenate(list(split.values())))) == len(data)
    assert [len(groups[k]) for k in ["train", "validation", "test"]] == [90, 30, 30]
    for key in split:
        np.testing.assert_array_equal(split[key], repeat[key])
    assert not {"ID", "Customer_ID", "Name", "SSN", "Credit_Score", "Month"} & set(FEATURES)


def test_imputer_learns_only_from_training_and_handles_new_categories():
    data, _ = clean_data(generate_demo(90, 3))
    parts = customer_split(data)
    train = data.iloc[parts["train"]].copy()
    test = data.iloc[parts["test"]].copy()
    model = estimator(LogisticRegression(max_iter=2000))
    model.fit(train[FEATURES], train.Credit_Score)
    imputer = model.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
    observed = imputer.statistics_.copy()
    annual_index = NUMERIC.index("Annual_Income")
    assert observed[annual_index] == pytest.approx(train.Annual_Income.median())
    test["Annual_Income"] = 1e9
    test["Occupation"] = "Never seen in training"
    assert len(model.predict(test[FEATURES])) == len(test)
    np.testing.assert_array_equal(observed, imputer.statistics_)


def test_small_customer_set_fails_clearly():
    data, _ = clean_data(generate_demo(30, 2))
    with pytest.raises(ValueError, match="at least 30"):
        customer_split(data.iloc[:20])
