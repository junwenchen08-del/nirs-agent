"""Optional local cross-encoder reranking for dense retrieval candidates."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

from nir_core.knowledge.base import SearchResult


class KnowledgeReranker(Protocol):
    """Minimal reranker contract consumed by the vector retriever."""

    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]: ...


def _section_text(value: object) -> str:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return value.strip()
        value = decoded
    if isinstance(value, (list, tuple)):
        return " > ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def build_passage(result: SearchResult) -> str:
    """Build a title-first passage for query-passage cross-encoding."""
    metadata = result.chunk.metadata or {}
    parts = [
        str(metadata.get("title", "")).strip(),
        _section_text(metadata.get("section_path")),
        result.chunk.content.strip(),
    ]
    return "\n".join(part for part in parts if part)


def rerank_results(
    candidates: list[SearchResult],
    scores: list[float],
) -> list[SearchResult]:
    """Attach dense/rerank diagnostics and sort by reranker relevance."""
    if len(candidates) != len(scores):
        raise ValueError("reranker score count must match candidate count")

    reranked: list[SearchResult] = []
    for candidate, raw_score in zip(candidates, scores, strict=True):
        score = max(0.0, min(1.0, float(raw_score)))
        chunk = candidate.chunk.model_copy(deep=True)
        chunk.metadata["dense_score"] = float(candidate.score)
        chunk.metadata["rerank_score"] = score
        reranked.append(
            SearchResult(
                chunk=chunk,
                score=score,
                related_entities=list(candidate.related_entities),
            )
        )
    return sorted(reranked, key=lambda result: result.score, reverse=True)


class CrossEncoderReranker:
    """Lazy, thread-safe Hugging Face sequence-classification reranker."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 768,
    ) -> None:
        if batch_size < 1:
            raise ValueError("reranker batch_size must be at least 1")
        if max_length < 32:
            raise ValueError("reranker max_length must be at least 32")
        normalized_device = device.strip().lower()
        if normalized_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("reranker device must be auto, cpu, or cuda")

        self._model_name = model_name
        self._device_request = normalized_device
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = ""
        self._lock = threading.RLock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device or self._device_request

    def _load(self) -> None:
        if self._model is not None:
            return

        # Keep optional TensorFlow and remote model lookups out of this local
        # PyTorch inference path.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        local_only = Path(self._model_name).is_dir()
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._device_request
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("reranker CUDA device requested but unavailable")

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            local_files_only=local_only,
        )
        model_kwargs = {"local_files_only": local_only}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            **model_kwargs,
        )
        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        if not candidates:
            return []
        with self._lock:
            self._load()
            assert self._torch is not None
            assert self._tokenizer is not None
            assert self._model is not None

            passages = [build_passage(candidate) for candidate in candidates]
            scores: list[float] = []
            with self._torch.inference_mode():
                for start in range(0, len(passages), self._batch_size):
                    batch = passages[start : start + self._batch_size]
                    encoded = self._tokenizer(
                        [query] * len(batch),
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self._max_length,
                        return_tensors="pt",
                    )
                    encoded = {
                        key: value.to(self._device) for key, value in encoded.items()
                    }
                    logits = self._model(**encoded).logits.reshape(-1).float()
                    scores.extend(self._torch.sigmoid(logits).cpu().tolist())
            return rerank_results(candidates, scores)
