from collections.abc import Sequence
from dataclasses import dataclass

from nurai.models.schemas import ChatResponse


@dataclass(frozen=True)
class RagasEvaluationSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str

    def as_record(self) -> dict[str, str | list[str]]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


def build_ragas_sample(response: ChatResponse, ground_truth: str) -> RagasEvaluationSample:
    return RagasEvaluationSample(
        question=response.question,
        answer=response.answer,
        contexts=[source.text for source in response.sources],
        ground_truth=ground_truth,
    )


def build_ragas_records(
    samples: Sequence[RagasEvaluationSample],
) -> list[dict[str, str | list[str]]]:
    return [sample.as_record() for sample in samples]


def to_ragas_dataset(samples: Sequence[RagasEvaluationSample]) -> object:
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = "Install optional eval dependencies with `pip install -e '.[eval]'`."
        raise RuntimeError(msg) from exc

    return Dataset.from_list(build_ragas_records(samples))  # type: ignore[no-any-return]


def evaluate_with_ragas(
    samples: Sequence[RagasEvaluationSample],
    metrics: Sequence[object],
) -> object:
    try:
        from ragas import evaluate  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = "Install optional eval dependencies with `pip install -e '.[eval]'`."
        raise RuntimeError(msg) from exc

    return evaluate(to_ragas_dataset(samples), metrics=metrics)
