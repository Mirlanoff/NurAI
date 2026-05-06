from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    text: str
    source: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class GoldenSample:
    id: str
    question: str
    ground_truth: str
    expected_keywords: tuple[str, ...]


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                msg = f"line in {path} is not a JSON object: {stripped!r}"
                raise ValueError(msg)
            yield payload


def _required(payload: dict[str, object], key: str, path: Path) -> object:
    if key not in payload:
        msg = f"missing required key {key!r} in {path}"
        raise ValueError(msg)
    return payload[key]


def load_corpus(path: Path) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for payload in _iter_jsonl(path):
        title = str(_required(payload, "title", path))
        text = str(_required(payload, "text", path))
        source = str(_required(payload, "source", path))
        raw_metadata = payload.get("metadata", {}) or {}
        if not isinstance(raw_metadata, dict):
            msg = f"metadata must be an object in {path}"
            raise ValueError(msg)
        metadata = {str(key): str(value) for key, value in raw_metadata.items()}
        documents.append(
            CorpusDocument(title=title, text=text, source=source, metadata=metadata)
        )
    return documents


def load_golden(path: Path) -> list[GoldenSample]:
    samples: list[GoldenSample] = []
    seen: set[str] = set()
    for payload in _iter_jsonl(path):
        sample_id = str(_required(payload, "id", path))
        if sample_id in seen:
            msg = f"duplicate sample id {sample_id!r} in {path}"
            raise ValueError(msg)
        seen.add(sample_id)
        question = str(_required(payload, "question", path))
        ground_truth = str(_required(payload, "ground_truth", path))
        raw_keywords = payload.get("expected_keywords", []) or []
        if not isinstance(raw_keywords, list):
            msg = f"expected_keywords must be a list in {path} (sample {sample_id!r})"
            raise ValueError(msg)
        keywords = tuple(str(item) for item in raw_keywords)
        samples.append(
            GoldenSample(
                id=sample_id,
                question=question,
                ground_truth=ground_truth,
                expected_keywords=keywords,
            )
        )
    return samples
