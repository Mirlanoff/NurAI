import re
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from nurai.models.domain import Chunk, Document

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def build_document_id(title: str, text: str, source: str) -> str:
    payload = f"{title}\n{source}\n{normalize_text(text)}".encode()
    return sha256(payload).hexdigest()[:16]


def build_chunk_id(document_id: str, position: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"nurai://chunk/{document_id}/{position}"))


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size"
            raise ValueError(msg)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_document(
        self,
        title: str,
        text: str,
        source: str,
        metadata: dict[str, str],
    ) -> Document:
        normalized = normalize_text(text)
        return Document(
            id=build_document_id(title=title, text=normalized, source=source),
            title=title,
            text=normalized,
            source=source,
            metadata=metadata,
        )

    def split(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []

        chunks: list[Chunk] = []
        start = 0
        position = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(document.text):
            end = min(start + self.chunk_size, len(document.text))
            chunk_text = document.text[start:end].strip()
            if chunk_text:
                chunk_id = build_chunk_id(document.id, position)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        title=document.title,
                        text=chunk_text,
                        source=document.source,
                        position=position,
                        metadata=document.metadata,
                    )
                )
                position += 1
            if end == len(document.text):
                break
            start += step

        return chunks
