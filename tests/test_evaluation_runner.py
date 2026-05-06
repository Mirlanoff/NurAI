import json
from pathlib import Path

import pytest

from nurai.evaluation.dataset import load_corpus, load_golden
from nurai.evaluation.gate import Thresholds
from nurai.evaluation.runner import (
    build_default_workflow,
    evaluate_workflow,
    ingest_corpus,
    main,
    run_evaluation,
)


@pytest.fixture()
def golden_corpus(tmp_path: Path) -> tuple[Path, Path]:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "title": "RAG handbook",
                "source": "unit-test",
                "metadata": {"team": "ml"},
                "text": (
                    "Retrieval-Augmented Generation combines retrieval with generation. "
                    "RAG systems need chunking, embeddings, vector search, and reranking."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "rag-components",
                "question": "What does RAG need?",
                "ground_truth": (
                    "RAG systems need chunking, embeddings, vector search, and reranking."
                ),
                "expected_keywords": ["chunking", "embeddings", "vector search"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return corpus_path, dataset_path


def test_load_corpus_and_golden_round_trip(golden_corpus: tuple[Path, Path]) -> None:
    corpus_path, dataset_path = golden_corpus

    corpus = load_corpus(corpus_path)
    samples = load_golden(dataset_path)

    assert len(corpus) == 1
    assert corpus[0].title == "RAG handbook"
    assert corpus[0].metadata == {"team": "ml"}
    assert len(samples) == 1
    assert samples[0].id == "rag-components"
    assert "chunking" in samples[0].expected_keywords


def test_load_golden_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text(
        '{"id": "a", "question": "q", "ground_truth": "g"}\n'
        '{"id": "a", "question": "q", "ground_truth": "g"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample id"):
        load_golden(dataset_path)


def test_evaluate_workflow_produces_summary(golden_corpus: tuple[Path, Path]) -> None:
    corpus_path, dataset_path = golden_corpus
    workflow = build_default_workflow()
    ingest_corpus(workflow, load_corpus(corpus_path))

    report = evaluate_workflow(workflow, load_golden(dataset_path), top_k=3)

    assert len(report.samples) == 1
    sample = report.samples[0]
    assert sample.id == "rag-components"
    assert sample.refused is False
    assert sample.retrieval_recall == 1.0
    assert sample.answer_overlap > 0
    assert report.mean_retrieval_recall == 1.0
    assert report.refusal_rate == 0.0


def test_run_evaluation_passes_strict_thresholds(golden_corpus: tuple[Path, Path]) -> None:
    corpus_path, dataset_path = golden_corpus
    thresholds = Thresholds(
        min_retrieval_recall=0.9,
        min_answer_overlap=0.1,
        min_mean_confidence=0.4,
        max_refusal_rate=0.0,
    )

    report, gate = run_evaluation(
        corpus_path=corpus_path,
        dataset_path=dataset_path,
        thresholds=thresholds,
        top_k=3,
    )

    assert gate.passed is True
    assert report.refusal_rate == 0.0


def test_run_evaluation_fails_when_thresholds_too_high(
    golden_corpus: tuple[Path, Path],
) -> None:
    corpus_path, dataset_path = golden_corpus
    thresholds = Thresholds(
        min_retrieval_recall=1.5,
        min_answer_overlap=1.5,
        min_mean_confidence=1.5,
        max_refusal_rate=0.0,
    )

    _, gate = run_evaluation(
        corpus_path=corpus_path,
        dataset_path=dataset_path,
        thresholds=thresholds,
        top_k=3,
    )

    assert gate.passed is False
    assert any("answer_overlap" in failure for failure in gate.failures)
    assert any("retrieval_recall" in failure for failure in gate.failures)


def test_main_writes_report_and_returns_zero_on_pass(
    golden_corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    corpus_path, dataset_path = golden_corpus
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "min_retrieval_recall": 0.9,
                "min_answer_overlap": 0.1,
                "min_mean_confidence": 0.4,
                "max_refusal_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--corpus",
            str(corpus_path),
            "--dataset",
            str(dataset_path),
            "--thresholds",
            str(thresholds_path),
            "--report",
            str(report_path),
            "--top-k",
            "3",
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["sample_count"] == 1


def test_main_returns_one_on_gate_failure(
    golden_corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    corpus_path, dataset_path = golden_corpus
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({"min_answer_overlap": 1.5}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--corpus",
            str(corpus_path),
            "--dataset",
            str(dataset_path),
            "--thresholds",
            str(thresholds_path),
            "--report",
            str(report_path),
            "--top-k",
            "3",
        ]
    )

    assert exit_code == 1
