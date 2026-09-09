"""Setup failures and target feasibility; no training or model downloads."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_reports_missing_checkout(tmp_path):
    module = load_script("lab3_preflight")
    missing = module.missing_files(tmp_path)
    assert "tests/test_model_data.py" in missing
    assert "src/bayan/models/training.py" in missing


def test_setup_uses_explicit_project_after_working_directory_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_script("lab3_preflight").missing_files(ROOT) == []


@pytest.mark.parametrize("support,baseline,ceiling,attainable", [
    ({"a": 10, "b": 0}, .5, .5, False),
    ({"a": 10, "b": 10}, 1.0, 1.0, False),
    ({"a": 10, "b": 10}, .71, 1.0, True),
    ({"a": 10, "b": 10}, .92, 1.0, True),
])
def test_target_audit_accounts_for_absent_classes_and_baseline_ceiling(support, baseline, ceiling, attainable):
    result = load_script("topic_target_audit").target_feasibility(support, baseline)
    assert result["maximum_possible_macro_f1"] == ceiling
    assert result["target_attainable_on_this_frozen_test"] is attainable


def test_target_audit_rejects_results_from_different_data(tmp_path):
    data = tmp_path / "changed.csv"
    data.write_text("changed data")
    report = tmp_path / "metrics.json"
    report.write_text('{"data_sha256": "a different data version"}')
    with pytest.raises(ValueError, match="different data"):
        load_script("topic_target_audit").build_report(
            data, report, report,
        )
