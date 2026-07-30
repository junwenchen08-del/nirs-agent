"""NIR knowledge-base semantic search tool.

Wraps ``nir_core.knowledge`` retrieval with an HTTP fallback so the tool
still works when the backend process cannot import the heavy deps
(torch / chromadb) that the in-process retriever needs.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _ok

logger = logging.getLogger(__name__)


# Cache: None = untested, "direct" = in-process import works, "http" = fallback
_knowledge_search_mode: str | None = None

# HTTP search server URL (set via NIR_KNOWLEDGE_URL env var, or auto-detected)
_knowledge_http_url: str | None = None

# Counter for periodic retry of direct import after HTTP fallback.
# Every _RETRY_INTERVAL calls in "http" mode, we retry direct import once.
_knowledge_http_call_count: int = 0
_RETRY_INTERVAL: int = 20
_DEFAULT_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS: float = 60.0


def _get_knowledge_http_url() -> str:
    """Determine the knowledge search server URL.

    Priority:
    1. NIR_KNOWLEDGE_URL env var
    2. http://host.docker.internal:8089 (Docker -> host)
    3. http://localhost:8089 (local dev)
    """
    global _knowledge_http_url
    if _knowledge_http_url is not None:
        return _knowledge_http_url

    url = os.environ.get("NIR_KNOWLEDGE_URL", "")
    if not url:
        # In Docker, host.docker.internal resolves to the host gateway.
        # In local dev, localhost works.
        url = "http://host.docker.internal:8089"
    _knowledge_http_url = url
    return url


def _knowledge_http_headers() -> dict[str, str]:
    """Build headers for the lightweight knowledge HTTP server."""
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("NIR_KNOWLEDGE_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _knowledge_http_timeout_seconds() -> float:
    """Return the validated, configurable HTTP search timeout."""
    raw = os.environ.get(
        "NIR_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS",
        str(_DEFAULT_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        timeout = _DEFAULT_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = _DEFAULT_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS
    return timeout


def _search_knowledge_via_http(
    query: str,
    top_k: int,
    purpose: str,
) -> str:
    """Fallback: search knowledge base via HTTP.

    Used when ``nir_core.knowledge`` cannot be imported in the backend
    process (heavy deps like torch/chromadb not installed). Calls a
    lightweight HTTP server running on the host machine.
    """
    import urllib.error
    import urllib.request

    base_url = _get_knowledge_http_url()

    # Search
    url = base_url + "/search"
    payload = json.dumps({"query": query, "top_k": top_k, "purpose": purpose}).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers=_knowledge_http_headers(),
            method="POST",
        )
        with urllib.request.urlopen(
            req,
            timeout=_knowledge_http_timeout_seconds(),
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("error"):
            return _err(f"Knowledge search error: {data['error']}")
    except urllib.error.HTTPError as exc:
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
            err_msg = err_body.get("error", str(err_body))
        except (ValueError, TypeError):
            err_msg = f"HTTP {exc.code} {exc.reason}"
        return _err(f"Knowledge search HTTP {exc.code}: {err_msg}")
    except urllib.error.URLError as exc:
        return _err(f"Knowledge search server unreachable at {url}: {exc.reason}. Start it with: python nir_core/knowledge/_search_server.py")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Knowledge search HTTP error: {type(exc).__name__}: {exc}")

    return _ok(data)


@tool("nir_search_knowledge", parse_docstring=True)
def nir_search_knowledge_tool(
    runtime: Runtime,
    query: str,
    top_k: int = 5,
    purpose: str = "answer",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Search the NIR knowledge base (papers / docs) for relevant sections.

    **WHEN TO CALL (mandatory triggers — not optional):**
    - ``nir_train_model`` / ``nir_analyze`` / ``nir_reflect`` returned
      ``knowledge_hint != null`` — use ``knowledge_hint.query`` verbatim.
    - ``domain`` is NOT in {food_moisture, food_protein, pharma, feed, soil,
      default} — unknown domain, no inline tips available.
    - ``nir_reflect`` returned ``grade ∈ {C, D, F}`` AND ``attempt >= 2``.
    - ``nir_train_model`` returned ``R2_val < 0.7`` — need domain-typical
      R²/RPD reference to judge whether the result is reasonable.
    - Choosing between SNV vs MSC, airPLS vs asLS, derivative1 vs derivative2
      and the inline docs do not cover this domain.
    - User asks "is this R²/RPD normal?" or "why so low?" — query the
      domain-typical range.

    **WHEN NOT TO CALL:**
    - Same query already searched in this conversation (avoid duplicate tokens).
    - ``grade`` is A/B and user did not ask for optimization.
    - Previous call returned ``count: 0`` for the same query (the knowledge
      base is empty or the retriever intentionally abstained).

    Set ``purpose="decision"`` whenever the retrieved literature will change a
    preprocessing choice, model choice, acceptance threshold, or reported
    recommendation. Decision mode requires at least two independent published
    documents with quality tier A-C. If ``decision_allowed`` is false, abstain
    and state what evidence is missing. For every material claim, copy the
    corresponding ``[KB:evidence_id]`` marker from ``evidence_assessment``;
    clearly label any synthesis beyond the cited passages as an inference and
    compare potentially conflicting findings before recommending an action.

    The knowledge base is populated offline via
    ``python -m nir_core.knowledge.cli import-dir <papers_dir>``. If the
    base is empty or available evidence is too weak/ambiguous, this tool
    returns an empty result list (not an error) plus retrieval diagnostics.

    Args:
        query: Natural-language query. When triggered by a ``knowledge_hint``,
            pass ``hint['query']`` verbatim instead of rewriting it.
            ``"SNV vs MSC for soil organic carbon"`` or
            ``"typical R2 for PLS on wheat protein"``.
        top_k: Maximum number of matching sections to return (default 5).
        purpose: ``answer`` for cited factual answers or ``decision`` for
            evidence used to choose/recommend an action.

    Returns:
        JSON with a list of matching paper sections. Each entry includes
        the chunk content (bounded to 3000 chars), source file, similarity
        stable evidence/chunk/document IDs, citation metadata, source file,
        similarity score, trust marker, and detected NIR entities. Document
        inventory is intentionally served by the separate management API.
    """
    global _knowledge_search_mode, _knowledge_http_call_count
    normalized_purpose = str(purpose).strip().lower()
    if normalized_purpose not in {"answer", "decision"}:
        return _err("purpose must be 'answer' or 'decision'")

    # Periodically retry direct import even after switching to HTTP,
    # so a transient failure doesn't permanently disable the fast path.
    if _knowledge_search_mode == "http":
        _knowledge_http_call_count += 1
        if _knowledge_http_call_count >= _RETRY_INTERVAL:
            _knowledge_search_mode = None  # force retry
            _knowledge_http_call_count = 0

    # Try in-process import first (fast path — works if deps are installed)
    if _knowledge_search_mode in (None, "direct"):
        try:
            from nir_core.knowledge.config import get_retriever

            retriever = get_retriever()
            if hasattr(retriever, "search_with_diagnostics"):
                decision = retriever.search_with_diagnostics(query, top_k=top_k)
                results = list(decision.results)
                retrieval = decision.diagnostics()
            else:
                results = retriever.search(query, top_k=top_k)
                retrieval = None
            from nir_core.knowledge.evidence import assess_evidence

            evidence_assessment = assess_evidence(
                results,
                purpose=normalized_purpose,
            )
            _knowledge_search_mode = "direct"

            if not results:
                abstained = bool(retrieval and retrieval.get("abstained"))
                return _ok(
                    {
                        "results": [],
                        "count": 0,
                        "query": query,
                        "retrieval": retrieval,
                        "evidence_assessment": evidence_assessment,
                        "message": (
                            "No sufficiently reliable knowledge evidence was found; weak or ambiguous vector matches were withheld."
                            if abstained
                            else "Knowledge base is empty or returned no matches. Populate it via `python -m nir_core.knowledge.cli import-dir <papers_dir>`."
                        ),
                    }
                )

            output = []
            for r in results:
                meta = r.chunk.metadata or {}

                def _load_entities(key: str) -> list[str]:
                    val = meta.get(key, "")
                    if isinstance(val, list):
                        return val
                    if isinstance(val, str) and val:
                        try:
                            decoded = json.loads(val)
                            return decoded if isinstance(decoded, list) else [val]
                        except (ValueError, TypeError):
                            return [val]
                    return []

                output.append(
                    {
                        "evidence_id": r.chunk.id,
                        "chunk_id": r.chunk.id,
                        "doc_id": meta.get("doc_id", ""),
                        "content": r.chunk.content[:3000],
                        "source": r.chunk.source,
                        "score": round(r.score, 4),
                        "title": meta.get("title", ""),
                        "authors": _load_entities("authors"),
                        "year": meta.get("year"),
                        "doi": meta.get("doi", ""),
                        "section_path": _load_entities("section_path"),
                        "page_start": meta.get("page_start"),
                        "page_end": meta.get("page_end"),
                        "quality_tier": meta.get("quality_tier", ""),
                        "review_status": meta.get("review_status", ""),
                        "content_trust": meta.get("content_trust", "untrusted_evidence"),
                        "entities": {
                            "methods": _load_entities("methods"),
                            "models": _load_entities("models"),
                            "datasets": _load_entities("datasets"),
                            "metrics": _load_entities("metrics"),
                        },
                        "related_entities": r.related_entities,
                    }
                )

            return _ok(
                {
                    "results": output,
                    "count": len(output),
                    "query": query,
                    "retrieval": retrieval,
                    "evidence_assessment": evidence_assessment,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Direct knowledge import failed, falling back to HTTP: %s: %s",
                type(exc).__name__,
                exc,
            )
            _knowledge_search_mode = "http"

    # Fallback: HTTP search server (runs on host with Anaconda Python)
    return _search_knowledge_via_http(
        query,
        top_k,
        normalized_purpose,
    )
