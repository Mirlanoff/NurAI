from uuid import UUID

from nurai.ingestion.chunker import (
    TextChunker,
    build_chunk_id,
    build_document_id,
    normalize_text,
)


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


def test_chunk_ids_are_deterministic_uuid_strings() -> None:
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    document = chunker.build_document(
        title="Doc",
        text="abcdefghijklmnopqrstuvwxyz",
        source="unit-test",
        metadata={},
    )

    chunks_first = chunker.split(document)
    chunks_second = chunker.split(document)

    assert [chunk.id for chunk in chunks_first] == [chunk.id for chunk in chunks_second]
    for chunk in chunks_first:
        assert UUID(chunk.id).version == 5


def test_build_chunk_id_is_unique_per_position() -> None:
    document_id = "doc-1"
    first = build_chunk_id(document_id, 0)
    second = build_chunk_id(document_id, 1)

    assert first != second
    assert UUID(first).version == 5
    assert UUID(second).version == 5
