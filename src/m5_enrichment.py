"""M5: combined JSON enrichment, individual techniques and deterministic fallbacks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config import ENRICHMENT_CACHE_DIR, LLM_MODEL, OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY, timeout=45, max_retries=1)


def _request(prompt: str, text: str, json_mode: bool = False) -> str | None:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-...":
        return None
    try:
        options = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = _client().chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=500,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            **options,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 -- API boundary has deterministic fallback
        warnings.warn(
            f"Enrichment API failed ({type(exc).__name__}); using local fallback."
        )
        return None


def _summary(text: str) -> str:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if s.strip()
    ]
    return " ".join(sentences[:2])[:400]


def _questions(text: str, n: int) -> list[str]:
    sentences = [
        s.strip(" #>*-.") for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10
    ]
    return [f"Thông tin về {s[:120]} là gì?" for s in sentences[: max(0, n)]]


def _metadata(text: str) -> dict:
    lower = text.lower()
    category = (
        "it"
        if any(t in lower for t in ("mật khẩu", "vpn", "malware", "mfa"))
        else (
            "finance"
            if any(t in lower for t in ("lương", "chi phí", "tạm ứng", "triệu"))
            else "hr"
        )
    )
    return {
        "topic": text.splitlines()[0].strip(" #")[:100] if text.strip() else "general",
        "entities": [],
        "category": category,
        "language": "vi",
        "date_range": re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text),
    }


def summarize_chunk(text: str) -> str:
    return _request(
        "Tóm tắt nội dung trong 1-2 câu ngắn tiếng Việt. Giữ số liệu và phủ định; không thêm dữ kiện.",
        text,
    ) or _summary(text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0:
        return []
    response = _request(
        f"Tạo {n_questions} câu hỏi tiếng Việt có thể trả lời bằng đoạn văn. Mỗi dòng một câu hỏi, không trả lời.",
        text,
    )
    return (
        [
            q.strip().lstrip("0123456789.-) ")
            for q in response.splitlines()
            if q.strip()
        ][:n_questions]
        if response
        else _questions(text, n_questions)
    )


def contextual_prepend(text: str, document_title: str = "") -> str:
    context = _request(
        "Viết một câu mô tả nguồn và chủ đề đoạn văn; không suy đoán ngoài dữ kiện được cung cấp.",
        f"Tài liệu: {document_title}\n\n{text}",
    )
    context = context or (
        f"Trích từ tài liệu {document_title}."
        if document_title
        else "Trích đoạn tài liệu nội bộ."
    )
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    raw = _request(
        'Trả về JSON metadata: {"topic": "...", "entities": [], "category": "hr|it|finance|policy", "language": "vi", "date_range": []}. Chỉ dùng dữ kiện trong đoạn văn.',
        text,
        True,
    )
    try:
        result = json.loads(raw) if raw else None
        return result if isinstance(result, dict) else _metadata(text)
    except (ValueError, TypeError):
        return _metadata(text)


@lru_cache(maxsize=2048)
def _enrich_single_call(text: str, source: str) -> dict:
    cache_key = hashlib.sha256(
        json.dumps(
            ["combined-v1", LLM_MODEL, text, source], ensure_ascii=False
        ).encode()
    ).hexdigest()
    cache_path = Path(ENRICHMENT_CACHE_DIR) / f"{cache_key}.json"
    if cache_path.exists() and OPENAI_API_KEY and OPENAI_API_KEY != "sk-...":
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("backend") == "openai"
                and isinstance(cached.get("summary"), str)
                and isinstance(cached.get("context"), str)
                and isinstance(cached.get("questions"), list)
                and isinstance(cached.get("metadata"), dict)
            ):
                return cached
        except (OSError, ValueError, AttributeError):
            pass
    fallback = {
        "summary": _summary(text),
        "questions": _questions(text, 3),
        "context": f"Trích từ tài liệu {source}."
        if source
        else "Trích đoạn tài liệu nội bộ.",
        "metadata": _metadata(text),
        "backend": "local_fallback",
    }
    raw = _request(
        "Phân tích đoạn văn, giữ số liệu và phủ định. Trả về JSON với summary (chuỗi tóm tắt), questions (3 câu hỏi), context (1 câu mô tả nguồn/chủ đề), metadata (object chứa topic, entities, category, language, date_range). Không tự tạo dữ kiện.",
        f"Tài liệu: {source}\n\n{text}",
        True,
    )
    try:
        result = json.loads(raw) if raw else None
        if not isinstance(result, dict):
            return fallback
        for key in ("summary", "context"):
            if not isinstance(result.get(key), str) or not result[key].strip():
                result[key] = fallback[key]
        if not isinstance(result.get("questions"), list):
            result["questions"] = fallback["questions"]
        result["questions"] = [q for q in result["questions"] if isinstance(q, str)][:3]
        if not isinstance(result.get("metadata"), dict):
            result["metadata"] = fallback["metadata"]
        result["backend"] = "openai"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result
    except (ValueError, TypeError):
        return fallback


def enrich_chunks(
    chunks: list[dict], methods: list[str] | None = None
) -> list[EnrichedChunk]:
    methods = ["combined"] if methods is None else methods
    if set(methods) - {"combined", "summary", "hyqa", "contextual", "metadata"}:
        raise ValueError("Unknown enrichment method")

    def enrich_one(chunk):
        text, meta = chunk["text"], dict(chunk.get("metadata", {}))
        source = meta.get("title") or meta.get("source", "")
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary, questions = result["summary"], result["questions"]
            enriched_text = f"{result['context']}\n\n{text}"
            auto = {**result["metadata"], "enrichment_backend": result["backend"]}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = (
                contextual_prepend(text, source) if "contextual" in methods else text
            )
            auto = extract_metadata(text) if "metadata" in methods else {}
        # Source, version and parent identifiers come only from ingestion.
        return EnrichedChunk(
            text, enriched_text, summary, questions, {**auto, **meta}, "+".join(methods)
        )

    enriched = []
    with ThreadPoolExecutor(
        max_workers=max(1, int(os.getenv("ENRICHMENT_WORKERS", "4")))
    ) as pool:
        for i, result in enumerate(pool.map(enrich_one, chunks)):
            enriched.append(result)
            if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)}", flush=True)
    return enriched
