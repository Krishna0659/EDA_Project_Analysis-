"""Customer-separated validation, model selection and a single holdout evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, DERIVED, FEATURES, LABELS, NUMERIC


def customer_split(df, seed=42):
    """60/20/20 by customer, approximately; no cross-customer imputation."""
    if df["Customer_ID"].nunique() < 30:
        raise ValueError("Evaluation requires at least 30 distinct labeled customers")
    outer = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    development, test = next(outer.split(df, groups=df["Customer_ID"]))
    inside = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1)
    train, validation = next(inside.split(df.iloc[development], groups=df.iloc[development]["Customer_ID"]))
    parts = {"train": development[train], "validation": development[validation], "test": test}
    for name, positions in parts.items():
        if set(df.iloc[positions]["Credit_Score"].unique()) != set(LABELS):
            raise ValueError(f"{name} lacks a target class; use more customers or a reviewed split")
    groups = {k: set(df.iloc[ix]["Customer_ID"]) for k, ix in parts.items()}
    if any(groups[a] & groups[b] for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]):
        raise AssertionError("Customer overlap in evaluation partitions")
    return parts


def estimator(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scale", StandardScaler())])
    category = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
                         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessing = ColumnTransformer([("numeric", numeric, NUMERIC + DERIVED),
                                       ("category", category, CATEGORICAL)])
    return Pipeline([("prepare", preprocessing), ("model", model)])


def scores(actual, predicted):
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0))}


def evaluate_models(df, seed=42):
    labeled = df.dropna(subset=["Credit_Score"]).reset_index(drop=True)
    parts = customer_split(labeled, seed)
    train, validation, test = (labeled.iloc[parts[k]] for k in ["train", "validation", "test"])
    factories = {
        "majority_baseline": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=160, max_depth=12, min_samples_leaf=5,
                                                        class_weight="balanced_subsample", random_state=seed, n_jobs=2),
    }
    validation_scores = {}
    for name, factory in factories.items():
        candidate = estimator(factory())
        candidate.fit(train[FEATURES], train["Credit_Score"])
        validation_scores[name] = scores(validation["Credit_Score"], candidate.predict(validation[FEATURES]))
    selected = max(validation_scores, key=lambda key: validation_scores[key]["macro_f1"])
    development = pd.concat([train, validation])
    fitted = estimator(factories[selected]())
    fitted.fit(development[FEATURES], development["Credit_Score"])
    predicted = fitted.predict(test[FEATURES])
    baseline = estimator(factories["majority_baseline"]())
    baseline.fit(development[FEATURES], development["Credit_Score"])
    # Resample customers, preserving the within-customer month dependence.
    groups = test["Customer_ID"].to_numpy()
    actual = test["Credit_Score"].to_numpy()
    unique_groups = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(200):
        ix = np.concatenate([positions[group] for group in rng.choice(unique_groups, len(unique_groups), replace=True)])
        bootstrap.append(f1_score(actual[ix], predicted[ix], labels=LABELS, average="macro", zero_division=0))
    split_info = {name: {"rows": len(ix), "customers": int(labeled.iloc[ix]["Customer_ID"].nunique()),
                         "class_counts": {label: int((labeled.iloc[ix]["Credit_Score"] == label).sum()) for label in LABELS}}
                  for name, ix in parts.items()}
    return {
        "seed": seed, "selection_metric": "validation macro F1", "selected_model": selected,
        "validation": validation_scores, "test": scores(actual, predicted),
        "test_majority_baseline": scores(actual, baseline.predict(test[FEATURES])),
        "test_macro_f1_customer_bootstrap_95pct": np.quantile(bootstrap, [.025, .975]).tolist(),
        "bootstrap_repetitions": 200, "split": split_info, "customer_overlap": 0,
        "classification_report": classification_report(actual, predicted, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(), "label_order": LABELS,
        "excluded_features": ["ID", "Customer_ID", "Name", "SSN", "Month", "Type_of_Loan", "Credit_Score"],
        "interpretation": "New-customer holdout within the same synthetic period; not a future-month backtest or fraud detector.",
    }
