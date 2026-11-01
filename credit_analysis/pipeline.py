"""Generate aggregate EDA, plots, a Markdown report, and holdout evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import FEATURES, LABELS, clean_data
from .model import evaluate_models

COLORS = ["#ea795c", "#6294d5", "#3bb9a0"]


def save_chart(fig, path, label):
    fig.text(.01, .008, label + " | customer-month observations", fontsize=8, color="#556477")
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def make_charts(df, output, kind, metrics):
    charts = output / "figures"
    charts.mkdir(parents=True, exist_ok=True)
    label = "SYNTHETIC DEMONSTRATION" if kind == "synthetic" else "USER-PROVIDED DATA"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155", "text.color": "#172b4d"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    counts = df["Credit_Score"].value_counts().reindex(LABELS, fill_value=0)
    axes[0].bar(counts.index, counts.values, color=COLORS)
    axes[0].set(title="Credit-score mix", ylabel="Labeled observations")
    for i, value in enumerate(counts):
        axes[0].text(i, value, f"{value:,}\n{value / counts.sum():.1%}" if counts.sum() else "0", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, max(counts.max() * 1.25, 1))
    axes[1].hist(df["Annual_Income"].dropna(), bins=40, color="#6294d5", edgecolor="white")
    axes[1].set(title="Income distribution", xlabel="Annual income (monetary units)", ylabel="Observations")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[2].hist(df["Credit_Utilization_Ratio"].dropna(), bins=30, color="#3bb9a0", edgecolor="white")
    axes[2].set(title="Credit utilization", xlabel="Utilization (%)", ylabel="Observations")
    save_chart(fig, charts / "overview.png", label)

    fig, ax = plt.subplots(figsize=(10, 5))
    missing = df[FEATURES].isna().mean().mul(100).sort_values().tail(12)
    ax.barh(missing.index, missing.values, color="#6294d5")
    ax.set(title="Data quality after parsing and domain checks", xlabel="Missing values (%)")
    save_chart(fig, charts / "data_quality.png", label)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, col, title, ylabel in zip(axes, ["Delay_from_due_date", "Outstanding_Debt", "EMI_to_Monthly_Salary"],
                                     ["Payment delays", "Outstanding debt", "Installment burden"], ["Days", "Monetary units", "EMI / net monthly salary"]):
        values = [df.loc[df["Credit_Score"] == target, col].dropna().to_numpy() for target in LABELS]
        boxes = ax.boxplot(values, tick_labels=LABELS, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
        ax.set(title=title, ylabel=ylabel)
    fig.suptitle("Associations by score class | outlier markers hidden, observations retained", fontsize=12)
    save_chart(fig, charts / "credit_profiles.png", label)

    columns = ["Annual_Income", "Outstanding_Debt", "Delay_from_due_date", "Credit_Utilization_Ratio", "Interest_Rate", "Credit_History_Age", "EMI_to_Monthly_Salary"]
    corr = df[columns].corr(method="spearman")
    corr.to_csv(output / "spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    graphic = ax.imshow(corr, vmin=-1, vmax=1, cmap="BrBG")
    ax.set_xticks(range(len(columns)), [s.replace("_", " ") for s in columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), [s.replace("_", " ") for s in columns], fontsize=9)
    for row in range(len(columns)):
        for col in range(len(columns)):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}" if pd.notna(value) else "N/A", ha="center", va="center", fontsize=8,
                    color="white" if pd.notna(value) and abs(value) > .7 else "#172b4d")
    ax.set_title("Spearman correlations | pairwise complete observations")
    fig.colorbar(graphic, ax=ax, shrink=.7)
    save_chart(fig, charts / "correlations.png", label)

    mix = pd.crosstab(df["Month"], df["Credit_Score"], normalize="index").reindex(columns=LABELS, fill_value=0)
    mix.to_csv(output / "monthly_score_mix.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for score, color in zip(LABELS, COLORS):
        ax.plot(mix.index, mix[score] * 100, marker="o", label=score, color=color)
    ax.set(title="Monthly score mix", xlabel="Month number (single-year input)", ylabel="Share of labeled observations (%)")
    ax.legend(frameon=False, ncols=3)
    save_chart(fig, charts / "monthly_mix.png", label)

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max() / 2 else "#172b4d")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set(title=f"Held-out customers | {metrics['selected_model']}", xlabel="Predicted class", ylabel="Actual class")
    save_chart(fig, charts / "confusion_matrix.png", label)


def markdown_table(frame):
    # Avoid an optional tabulate dependency and never emit raw input strings as HTML.
    def value(x):
        return f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(value(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def write_report(df, audit, metrics, output, kind):
    title = "Synthetic demonstration" if kind == "synthetic" else "User-provided dataset (provenance not independently verified)"
    counts = df["Credit_Score"].value_counts().reindex(LABELS, fill_value=0)
    score_table = pd.DataFrame({"Score": LABELS, "Observations": counts.values,
                                "Share of labeled rows (%)": counts.values / max(counts.sum(), 1) * 100})
    score_table.to_csv(output / "score_distribution.csv", index=False)
    profile = df.groupby("Credit_Score")[["Annual_Income", "Outstanding_Debt", "Delay_from_due_date", "Credit_Utilization_Ratio", "EMI_to_Monthly_Salary"]].median().reindex(LABELS)
    profile.to_csv(output / "median_credit_profiles.csv")
    df[FEATURES].describe().T.to_csv(output / "descriptive_statistics.csv")
    validation = pd.DataFrame(metrics["validation"]).T.rename_axis("Model").reset_index()
    validation.to_csv(output / "validation_metrics.csv", index=False)
    lower, upper = metrics["test_macro_f1_customer_bootstrap_95pct"]
    poor, good = profile.loc["Poor"], profile.loc["Good"]
    text = f"""# Credit-profile analysis: {title}

**Data status: {kind.upper()}.** {"All patterns and performance below come from generated data. They are not findings about real customers or Paisabazaar." if kind == "synthetic" else "These are descriptive associations, not causal effects or verified lending outcomes."}

## Dataset and quality

- Input: **{audit['input_rows']:,} rows**; **{audit['exact_duplicates_removed']:,} exact duplicates** removed.
- Analysis: **{len(df):,} customer-month observations**, **{audit['customers']:,} customers**.
- Unlabeled rows: **{audit['unlabeled_rows']:,}** (excluded from score comparisons and modeling).
- Missing feature cells after cleaning: **{audit['missing_cells_after']:,}**; EDA uses available observations.
- Parsing failures and invalid values are set to missing and counted in `data_quality.json`.
- Original monetary outliers remain; no global clipping, mean filling, or label-dependent cleaning.

{markdown_table(score_table)}

![Overview](figures/overview.png)
![Quality](figures/data_quality.png)

## Descriptive observations

- Median delay: **{poor['Delay_from_due_date']:.1f} days** for Poor versus **{good['Delay_from_due_date']:.1f} days** for Good.
- Median utilization: **{poor['Credit_Utilization_Ratio']:.1f}%** for Poor versus **{good['Credit_Utilization_Ratio']:.1f}%** for Good.
- Median outstanding debt: **{poor['Outstanding_Debt']:,.0f}** versus **{good['Outstanding_Debt']:,.0f} monetary units**, respectively.

These summaries weight each customer-month equally. Repeated observations are
not independent customers. No causal or population-significance claim is made.
For the synthetic demo, the relationships are intentionally introduced by the generator.

![Profiles](figures/credit_profiles.png)
![Correlations](figures/correlations.png)
![Monthly score mix](figures/monthly_mix.png)

`Debt_to_Annual_Income` is outstanding debt divided by annual income, a stock-to-income
ratio. `EMI_to_Monthly_Salary` is monthly installments divided by net monthly salary.
Neither is a complete underwriting debt-to-income measure. Currency is unspecified.

## Model experiment

Customers are separated into train/validation/test groups (60/20/20 by customer).
Imputation, scaling and category encoding are fitted only on training rows during
selection. Validation macro F1 selects the model; it is refitted on train plus
validation before a single test-set evaluation. Direct identifiers, month,
loan-description text and the target are excluded from predictors.

{markdown_table(validation)}

- Selected model: **{metrics['selected_model']}**.
- Held-out accuracy: **{metrics['test']['accuracy']:.3f}**.
- Held-out balanced accuracy: **{metrics['test']['balanced_accuracy']:.3f}**.
- Held-out macro F1: **{metrics['test']['macro_f1']:.3f}**.
- Majority baseline test macro F1: **{metrics['test_majority_baseline']['macro_f1']:.3f}**.
- Customer-bootstrap 95% interval for test macro F1: **{lower:.3f}–{upper:.3f}** (200 resamples; fixed fitted model).
- Customer overlap across partitions: **0**.

![Confusion matrix](figures/confusion_matrix.png)

The interval describes test-customer sampling variability for this fitted model;
it does not capture retraining variability or generator-to-world differences.
This split estimates new-customer performance within the same period, not future
performance. Contemporaneous credit attributes may encode the score itself.

## Conclusion and next steps

The project demonstrates an auditable EDA and model-evaluation workflow. Real
data is needed to establish any customer insight or business value. Before using
it beyond a portfolio exercise: verify data rights and units, define label timing,
check whether predictors precede outcomes, evaluate a future-period holdout,
assess subgroup performance and calibration, and review missingness and drift.
The dataset has no confirmed fraud target or default outcome, so fraud-detection
accuracy, default reduction and lending recommendations cannot be measured here.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")


def run_analysis(input_path, output_dir, data_kind=None, seed=42, history_unit="months"):
    source, output = Path(input_path), Path(output_dir)
    manifest_path = source.with_suffix(".metadata.json")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != digest:
            raise ValueError("Dataset checksum does not match its manifest; regenerate or review the input")
        if manifest.get("synthetic"):
            if data_kind not in {None, "synthetic"}:
                raise ValueError("Synthetic input cannot be labeled as user-provided observations")
            data_kind = "synthetic"
    if data_kind not in {"synthetic", "user-provided"}:
        raise ValueError("Specify data_kind='synthetic' or 'user-provided' when no synthetic manifest is available")
    raw = pd.read_csv(source, low_memory=False)
    cleaned, audit = clean_data(raw, history_unit)
    metrics = evaluate_models(cleaned, seed)
    output.mkdir(parents=True, exist_ok=True)
    make_charts(cleaned, output, data_kind, metrics)
    write_report(cleaned, audit, metrics, output, data_kind)
    run = {"data_kind": data_kind, "input_filename": source.name, "input_sha256": digest,
           "seed": seed, "python": platform.python_version(), "history_unit": history_unit,
           "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib", "scikit-learn"]}}
    for filename, contents in [("data_quality.json", audit), ("model_metrics.json", metrics), ("run_manifest.json", run)]:
        (output / filename).write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Completed {data_kind.upper()} analysis: {output / 'REPORT.md'}")
    return {"data": cleaned, "quality": audit, "metrics": metrics, "run": run}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/demo/synthetic_credit.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/local"))
    parser.add_argument("--data-kind", choices=["synthetic", "user-provided"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-unit", choices=["months", "years"], default="months")
    args = parser.parse_args()
    try:
        run_analysis(args.input, args.output, args.data_kind, args.seed, args.history_unit)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Input error: {error}\n")


if __name__ == "__main__":
    main()
