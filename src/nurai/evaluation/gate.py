from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    min_retrieval_recall: float = 0.0
    min_answer_overlap: float = 0.0
    min_mean_confidence: float = 0.0
    max_refusal_rate: float = 1.0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


def load_thresholds(path: Path) -> Thresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"thresholds file {path} must contain a JSON object"
        raise ValueError(msg)
    return Thresholds(
        min_retrieval_recall=float(payload.get("min_retrieval_recall", 0.0)),
        min_answer_overlap=float(payload.get("min_answer_overlap", 0.0)),
        min_mean_confidence=float(payload.get("min_mean_confidence", 0.0)),
        max_refusal_rate=float(payload.get("max_refusal_rate", 1.0)),
    )


def evaluate_gate(
    *,
    mean_retrieval_recall: float,
    mean_answer_overlap: float,
    mean_confidence: float,
    refusal_rate: float,
    thresholds: Thresholds,
) -> GateResult:
    failures: list[str] = []
    if mean_retrieval_recall < thresholds.min_retrieval_recall:
        failures.append(
            f"retrieval_recall={mean_retrieval_recall:.3f} < "
            f"min_retrieval_recall={thresholds.min_retrieval_recall:.3f}"
        )
    if mean_answer_overlap < thresholds.min_answer_overlap:
        failures.append(
            f"answer_overlap={mean_answer_overlap:.3f} < "
            f"min_answer_overlap={thresholds.min_answer_overlap:.3f}"
        )
    if mean_confidence < thresholds.min_mean_confidence:
        failures.append(
            f"mean_confidence={mean_confidence:.3f} < "
            f"min_mean_confidence={thresholds.min_mean_confidence:.3f}"
        )
    if refusal_rate > thresholds.max_refusal_rate:
        failures.append(
            f"refusal_rate={refusal_rate:.3f} > "
            f"max_refusal_rate={thresholds.max_refusal_rate:.3f}"
        )
    return GateResult(passed=not failures, failures=tuple(failures))
