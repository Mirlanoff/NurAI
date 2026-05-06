import json
from pathlib import Path

from nurai.evaluation.gate import Thresholds, evaluate_gate, load_thresholds


def test_gate_passes_when_all_metrics_meet_thresholds() -> None:
    thresholds = Thresholds(
        min_retrieval_recall=0.5,
        min_answer_overlap=0.2,
        min_mean_confidence=0.4,
        max_refusal_rate=0.1,
    )

    result = evaluate_gate(
        mean_retrieval_recall=0.9,
        mean_answer_overlap=0.3,
        mean_confidence=0.7,
        refusal_rate=0.0,
        thresholds=thresholds,
    )

    assert result.passed is True
    assert result.failures == ()


def test_gate_collects_all_failures() -> None:
    thresholds = Thresholds(
        min_retrieval_recall=0.9,
        min_answer_overlap=0.5,
        min_mean_confidence=0.8,
        max_refusal_rate=0.0,
    )

    result = evaluate_gate(
        mean_retrieval_recall=0.5,
        mean_answer_overlap=0.1,
        mean_confidence=0.4,
        refusal_rate=0.5,
        thresholds=thresholds,
    )

    assert result.passed is False
    assert len(result.failures) == 4
    failure_text = "\n".join(result.failures)
    assert "retrieval_recall" in failure_text
    assert "answer_overlap" in failure_text
    assert "mean_confidence" in failure_text
    assert "refusal_rate" in failure_text


def test_load_thresholds_from_file(tmp_path: Path) -> None:
    payload = {
        "min_retrieval_recall": 0.7,
        "min_answer_overlap": 0.2,
        "min_mean_confidence": 0.5,
        "max_refusal_rate": 0.0,
    }
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(json.dumps(payload), encoding="utf-8")

    thresholds = load_thresholds(threshold_path)

    assert thresholds.min_retrieval_recall == 0.7
    assert thresholds.min_answer_overlap == 0.2
    assert thresholds.min_mean_confidence == 0.5
    assert thresholds.max_refusal_rate == 0.0


def test_load_thresholds_uses_defaults_for_missing_keys(tmp_path: Path) -> None:
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text("{}", encoding="utf-8")

    thresholds = load_thresholds(threshold_path)

    assert thresholds.min_retrieval_recall == 0.0
    assert thresholds.max_refusal_rate == 1.0
