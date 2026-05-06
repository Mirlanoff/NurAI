from nurai.evaluation.metrics import answer_overlap, retrieval_recall


def test_retrieval_recall_returns_one_when_all_keywords_present() -> None:
    chunks = [
        "The pipeline needs chunking, embeddings, retrieval and reranking.",
    ]

    recall = retrieval_recall(["chunking", "embeddings", "retrieval"], chunks)

    assert recall == 1.0


def test_retrieval_recall_is_partial_when_some_keywords_missing() -> None:
    chunks = ["only chunking and retrieval are mentioned here"]

    recall = retrieval_recall(["chunking", "embeddings", "retrieval"], chunks)

    assert 0.6 < recall < 0.7


def test_retrieval_recall_handles_multi_word_keywords() -> None:
    chunks = ["RAG evaluation tracks retrieval recall and answer relevance."]

    recall = retrieval_recall(["retrieval recall", "answer relevance"], chunks)

    assert recall == 1.0


def test_retrieval_recall_returns_zero_when_no_chunks() -> None:
    recall = retrieval_recall(["chunking"], [])

    assert recall == 0.0


def test_retrieval_recall_returns_one_when_no_keywords_required() -> None:
    recall = retrieval_recall([], ["irrelevant chunk text"])

    assert recall == 1.0


def test_answer_overlap_is_one_for_identical_text_modulo_punctuation() -> None:
    overlap = answer_overlap(
        "RAG combines retrieval and generation.",
        "RAG combines retrieval and generation",
    )

    assert overlap == 1.0


def test_answer_overlap_handles_partial_overlap() -> None:
    overlap = answer_overlap(
        "RAG combines retrieval and generation",
        "RAG retrieves documents and generates answers",
    )

    assert 0.0 < overlap < 1.0


def test_answer_overlap_returns_zero_for_disjoint_text() -> None:
    overlap = answer_overlap("blue green red", "alpha beta gamma")

    assert overlap == 0.0


def test_answer_overlap_returns_zero_for_empty_predicted_answer() -> None:
    overlap = answer_overlap("", "RAG combines retrieval and generation")

    assert overlap == 0.0
