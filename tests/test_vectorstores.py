from nurai.models.domain import Chunk
from nurai.vectorstores.memory import InMemoryVectorStore


def test_in_memory_vector_store_healthcheck_and_count() -> None:
    store = InMemoryVectorStore()

    assert store.healthcheck() is True
    assert store.count() == 0


def test_in_memory_vector_store_searches_by_cosine_score() -> None:
    store = InMemoryVectorStore()
    chunks = [
        Chunk(
            id="a",
            document_id="doc-a",
            title="A",
            text="alpha",
            source="unit-test",
            position=0,
        ),
        Chunk(
            id="b",
            document_id="doc-b",
            title="B",
            text="beta",
            source="unit-test",
            position=0,
        ),
    ]

    store.upsert(chunks=chunks, vectors=[[1.0, 0.0], [0.0, 1.0]])
    results = store.search(vector=[1.0, 0.0], top_k=1)

    assert store.count() == 2
    assert results[0].chunk.id == "a"
    assert results[0].score == 1.0
