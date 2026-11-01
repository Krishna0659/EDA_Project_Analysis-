import hashlib
import json
from pathlib import Path

import pytest

from credit_analysis.demo import generate_demo
from credit_analysis.pipeline import run_analysis


def test_full_pipeline_writes_safe_aggregate_outputs(tmp_path):
    source = tmp_path / "synthetic.csv"
    generate_demo(90, 4).to_csv(source, index=False)
    result = run_analysis(source, tmp_path / "report", data_kind="synthetic")
    assert result["quality"]["clean_rows"] == 360
    assert result["metrics"]["customer_overlap"] == 0
    assert 0 <= result["metrics"]["test"]["macro_f1"] <= 1
    assert sum(map(sum, result["metrics"]["confusion_matrix"])) == result["metrics"]["split"]["test"]["rows"]
    report = (tmp_path / "report/REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in report
    assert "not findings about real customers" in report
    assert len(list((tmp_path / "report/figures").glob("*.png"))) == 6
    for path in (tmp_path / "report").glob("*.csv"):
        assert "SYNTH_" not in path.read_text(encoding="utf-8")


def test_manifest_blocks_checksum_mismatch_and_mislabeling(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("placeholder", encoding="utf-8")
    manifest = {"synthetic": True, "sha256": "wrong"}
    source.with_suffix(".metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        run_analysis(source, tmp_path / "report")
    manifest["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_suffix(".metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Synthetic input cannot"):
        run_analysis(source, tmp_path / "report", data_kind="user-provided")


def test_unmarked_input_requires_explicit_data_kind(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("placeholder", encoding="utf-8")
    with pytest.raises(ValueError, match="Specify data_kind"):
        run_analysis(source, tmp_path / "report")


def test_committed_demo_matches_manifest():
    source = Path(__file__).resolve().parents[1] / "data/demo/synthetic_credit.csv"
    manifest = json.loads(source.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
