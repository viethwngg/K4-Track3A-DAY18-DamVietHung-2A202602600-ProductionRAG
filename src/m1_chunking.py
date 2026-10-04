"""M1: sentence semantics, bounded parent/child chunks and Markdown sections."""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
    model_path,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n\n".join(p.extract_text() or "" for p in PdfReader(path).pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    docs = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix.lower() not in {".md", ".pdf"}:
            continue
        text = (
            path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".md"
            else _extract_pdf_text(str(path))
        )
        if not text.strip():
            warnings.warn(f"Skipped {path.name}: no text layer; OCR required.")
            continue
        title = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        version = re.search(r"Phiên bản:\s*([\d.]+)", text)
        effective = re.search(r"Ngày hiệu lực:\s*([\d/]+)", text)
        docs.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "title": title.group(1) if title else path.stem,
                    "version": version.group(1) if version else "",
                    "effective_date": effective.group(1) if effective else "",
                },
            }
        )
    return docs


def _split_bounded(text: str, size: int) -> list[str]:
    """Prefer paragraph/line/word boundaries; hard split only oversized tokens."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    pieces = []
    remaining = text.strip()
    while len(remaining) > size:
        stop = remaining.rfind("\n\n", 0, size + 1)
        if stop < size // 2:
            stop = max(
                remaining.rfind("\n", 0, size + 1), remaining.rfind(" ", 0, size + 1)
            )
        if stop <= 0:
            stop = size
        pieces.append(remaining[:stop].strip())
        remaining = remaining[stop:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_basic(
    text: str, chunk_size: int = 500, metadata: dict | None = None
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks, current = [], ""
    for para in filter(None, (p.strip() for p in text.split("\n\n"))):
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return [
        Chunk(t, {**(metadata or {}), "chunk_index": i, "strategy": "basic"})
        for i, t in enumerate(chunks)
    ]


@lru_cache(maxsize=1)
def _semantic_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path("sentence-transformers/all-MiniLM-L6-v2"))


def chunk_semantic(
    text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None
) -> list[Chunk]:
    import numpy as np

    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()]
    if not sentences:
        return []
    vectors = _semantic_encoder().encode(sentences, normalize_embeddings=True)
    groups, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if float(np.dot(vectors[i - 1], vectors[i])) < threshold:
            groups.append("\n".join(current))
            current = []
        current.append(sentences[i])
    groups.append("\n".join(current))
    return [
        Chunk(t, {**(metadata or {}), "chunk_index": i, "strategy": "semantic"})
        for i, t in enumerate(groups)
    ]


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    if not 0 < child_size < parent_size:
        raise ValueError("Require 0 < child_size < parent_size")
    meta = metadata or {}
    namespace = hashlib.sha256(
        (meta.get("source", "") + "\0" + text).encode()
    ).hexdigest()[:16]
    parents, children = [], []
    for i, parent_text in enumerate(_split_bounded(text, parent_size)):
        pid = f"{namespace}_parent_{i}"
        parents.append(
            Chunk(
                parent_text,
                {
                    **meta,
                    "parent_id": pid,
                    "chunk_type": "parent",
                    "strategy": "hierarchical",
                },
            )
        )
        for j, child_text in enumerate(_split_bounded(parent_text, child_size)):
            children.append(
                Chunk(
                    child_text,
                    {
                        **meta,
                        "chunk_type": "child",
                        "chunk_index": j,
                        "strategy": "hierarchical",
                        "chunk_id": f"{pid}_child_{j}",
                    },
                    pid,
                )
            )
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    chunks, lines, section, fence = [], [], "", None

    def flush():
        content = "\n".join(lines).strip()
        if content:
            chunks.append(
                Chunk(
                    content,
                    {
                        **(metadata or {}),
                        "section": section,
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )

    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)[0]
            fence = None if fence == token else (token if fence is None else fence)
        header = re.match(r"^#{1,6}\s+.+", line) if fence is None else None
        if header:
            flush()
            lines = []
            section = line.strip()
        lines.append(line)
    flush()
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    strategies = {"basic": [], "semantic": [], "hierarchical": [], "structure": []}
    parent_count = 0
    for doc in documents:
        text, meta = doc["text"], doc.get("metadata", {})
        strategies["basic"].extend(chunk_basic(text, metadata=meta))
        strategies["semantic"].extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        parent_count += len(parents)
        strategies["hierarchical"].extend(children)
        strategies["structure"].extend(chunk_structure_aware(text, meta))
    result = {}
    for name, chunks in strategies.items():
        lengths = [len(c.text) for c in chunks]
        result[name] = {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)) if lengths else 0,
            "min_len": min(lengths, default=0),
            "max_len": max(lengths, default=0),
        }
    result["hierarchical"]["parents"] = parent_count
    return result
