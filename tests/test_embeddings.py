from nurai.embeddings.hashing import HashingEmbedder


def test_hashing_embedder_returns_normalized_vector() -> None:
    embedder = HashingEmbedder(dimensions=64)

    vector = embedder.embed("RAG uses retrieval and generation")

    assert len(vector) == 64
    norm = sum(value * value for value in vector) ** 0.5
    assert round(norm, 6) == 1.0


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder(dimensions=64)

    assert embedder.embed("same text") == embedder.embed("same text")
