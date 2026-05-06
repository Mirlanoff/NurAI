from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from nurai.agents.workflow import AgentWorkflow
from nurai.core.config import Settings
from nurai.embeddings.hashing import HashingEmbedder
from nurai.evaluation.dataset import (
    CorpusDocument,
    GoldenSample,
    load_corpus,
    load_golden,
)
from nurai.evaluation.gate import GateResult, Thresholds, evaluate_gate, load_thresholds
from nurai.evaluation.metrics import answer_overlap, retrieval_recall
from nurai.rerankers.lexical import LexicalReranker
from nurai.retrieval.bm25 import BM25Index
from nurai.services.rag import RagService
from nurai.vectorstores.memory import InMemoryVectorStore


@dataclass(frozen=True)
class SampleResult:
    id: str
    question: str
    ground_truth: str
    predicted_answer: str
    confidence: float
    refused: bool
    retrieval_recall: float
    answer_overlap: float
    retrieved_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    samples: tuple[SampleResult, ...]
    mean_retrieval_recall: float
    mean_answer_overlap: float
    mean_confidence: float
    refusal_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "mean_retrieval_recall": self.mean_retrieval_recall,
                "mean_answer_overlap": self.mean_answer_overlap,
                "mean_confidence": self.mean_confidence,
                "refusal_rate": self.refusal_rate,
                "sample_count": len(self.samples),
            },
            "samples": [asdict(sample) for sample in self.samples],
        }


def build_default_workflow() -> AgentWorkflow:
    settings = Settings(
        retrieval_backend="hybrid",
        reranker_backend="lexical",
        vector_store_backend="memory",
        rate_limit_enabled=False,
        agent_min_confidence=0.0,
    )
    service = RagService(
        settings=settings,
        embedder=HashingEmbedder(dimensions=settings.embedding_dimensions),
        vector_store=InMemoryVectorStore(),
        bm25_index=BM25Index(),
        reranker=LexicalReranker(),
    )
    return AgentWorkflow(rag_service=service)


def ingest_corpus(workflow: AgentWorkflow, corpus: Sequence[CorpusDocument]) -> None:
    for document in corpus:
        workflow.rag_service.ingest_text(
            title=document.title,
            text=document.text,
            source=document.source,
            metadata=document.metadata,
        )


def evaluate_workflow(
    workflow: AgentWorkflow,
    samples: Sequence[GoldenSample],
    top_k: int,
) -> EvaluationReport:
    sample_results: list[SampleResult] = []
    for sample in samples:
        response = workflow.run(question=sample.question, top_k=top_k)
        retrieved_texts = [source.text for source in response.sources]
        retrieved_ids = tuple(source.chunk_id for source in response.sources)
        recall = retrieval_recall(sample.expected_keywords, retrieved_texts)
        overlap = answer_overlap(response.answer, sample.ground_truth)
        sample_results.append(
            SampleResult(
                id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                predicted_answer=response.answer,
                confidence=response.confidence,
                refused=response.refusal_reason is not None,
                retrieval_recall=recall,
                answer_overlap=overlap,
                retrieved_chunk_ids=retrieved_ids,
            )
        )

    if not sample_results:
        return EvaluationReport(
            samples=(),
            mean_retrieval_recall=0.0,
            mean_answer_overlap=0.0,
            mean_confidence=0.0,
            refusal_rate=0.0,
        )

    mean_recall = sum(item.retrieval_recall for item in sample_results) / len(sample_results)
    mean_overlap = sum(item.answer_overlap for item in sample_results) / len(sample_results)
    mean_confidence = sum(item.confidence for item in sample_results) / len(sample_results)
    refusal_rate = sum(1 for item in sample_results if item.refused) / len(sample_results)

    return EvaluationReport(
        samples=tuple(sample_results),
        mean_retrieval_recall=mean_recall,
        mean_answer_overlap=mean_overlap,
        mean_confidence=mean_confidence,
        refusal_rate=refusal_rate,
    )


def run_evaluation(
    *,
    corpus_path: Path,
    dataset_path: Path,
    thresholds: Thresholds,
    top_k: int,
) -> tuple[EvaluationReport, GateResult]:
    workflow = build_default_workflow()
    ingest_corpus(workflow, load_corpus(corpus_path))
    report = evaluate_workflow(workflow, load_golden(dataset_path), top_k=top_k)
    gate = evaluate_gate(
        mean_retrieval_recall=report.mean_retrieval_recall,
        mean_answer_overlap=report.mean_answer_overlap,
        mean_confidence=report.mean_confidence,
        refusal_rate=report.refusal_rate,
        thresholds=thresholds,
    )
    return report, gate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the NurAI agent against a golden dataset "
            "and enforce regression thresholds."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("eval/corpus.jsonl"),
        help="Path to the corpus JSONL fed to the agent.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/golden.jsonl"),
        help="Path to the golden questions JSONL.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("eval/thresholds.json"),
        help="Path to the JSON file with regression thresholds.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("eval/report.json"),
        help="Where to write the JSON evaluation report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="top_k passed to the agent for each question.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    thresholds = load_thresholds(args.thresholds)
    report, gate = run_evaluation(
        corpus_path=args.corpus,
        dataset_path=args.dataset,
        thresholds=thresholds,
        top_k=args.top_k,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    summary = report.to_dict()["summary"]
    print(json.dumps(summary, indent=2))

    if not gate.passed:
        for failure in gate.failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        return 1

    print("GATE PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
