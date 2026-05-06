from nurai.evaluation.ragas import build_ragas_records, build_ragas_sample
from nurai.models.schemas import ChatResponse, SourceChunk


def test_build_ragas_sample_from_chat_response() -> None:
    response = ChatResponse(
        question="What is RAG?",
        answer="RAG combines retrieval and generation.",
        confidence=0.9,
        sources=[
            SourceChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                title="RAG handbook",
                source="unit-test",
                text="RAG combines retrieval and generation.",
                score=0.9,
                metadata={},
            )
        ],
    )

    sample = build_ragas_sample(response=response, ground_truth="Retrieval plus generation.")
    records = build_ragas_records([sample])

    assert records == [
        {
            "question": "What is RAG?",
            "answer": "RAG combines retrieval and generation.",
            "contexts": ["RAG combines retrieval and generation."],
            "ground_truth": "Retrieval plus generation.",
        }
    ]
