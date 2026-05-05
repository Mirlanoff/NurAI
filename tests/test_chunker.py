from nurai.ingestion.chunker import TextChunker, build_document_id, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello\n\n   world\t ") == "hello world"


def test_build_document_id_is_deterministic() -> None:
    first = build_document_id(title="A", text="text", source="source")
    second = build_document_id(title="A", text="text", source="source")

    assert first == second


def test_chunker_splits_text_with_overlap() -> None:
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    document = chunker.build_document(
        title="Doc",
        text="abcdefghijklmnopqrstuvwxyz",
        source="unit-test",
        metadata={"kind": "test"},
    )

    chunks = chunker.split(document)

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]
    assert chunks[0].metadata == {"kind": "test"}
