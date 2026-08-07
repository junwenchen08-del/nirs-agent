"""NIR spectroscopy community tools for DeerFlow — facade module.

Historical note: this module used to contain ~2000 lines of tool
implementations. It has been split into focused sub-modules:

- ``_common``         — path resolution, JSON helpers, pipeline-step parsing
- ``_knowledge_hint`` — structured knowledge-base retrieval hint builder
- ``_report``         — Markdown report generator
- ``io_tools``        — nir_load_data / nir_inspect / nir_predict
- ``preprocess``      — nir_preprocess
- ``classification``  — nir_train_classifier
- ``modeling``        — nir_train_model / nir_analyze / nir_compare /
                        nir_register_model
- ``reflect``         — nir_reflect
- ``knowledge``       — nir_search_knowledge (+ HTTP fallback)
- ``workflow``        — durable NIR workflow state and approval gates

This file remains as a re-export facade so existing imports of the form
``from deerflow.community.nir.tools import nir_reflect_tool`` and the
config.yaml entry ``use: deerflow.community.nir.tools:nir_reflect_tool``
continue to work without modification.

If you are writing new code, prefer importing directly from the relevant
sub-module.
"""

from __future__ import annotations

# Re-export the shared helpers (tests and downstream code may patch /
# import these from ``deerflow.community.nir.tools``).
from ._common import (
    _err,
    _json_default,
    _ok,
    _parse_pipeline_step,
    _resolve,
    _resolve_writable_dir,
    logger,
)
from ._knowledge_hint import _KNOWN_DOMAINS, _build_knowledge_hint
from ._report import _build_report
from .classification import nir_train_classifier_tool

# Re-export every @tool function so ``deerflow.community.nir.tools:<name>``
# paths in config.yaml keep working.
from .io_tools import nir_inspect_tool, nir_load_data_tool, nir_predict_tool
from .knowledge import (
    _RETRY_INTERVAL,
    _get_knowledge_http_url,
    _knowledge_http_call_count,
    _knowledge_http_url,
    _knowledge_search_mode,
    _search_knowledge_via_http,
    nir_search_knowledge_tool,
)
from .modeling import (
    nir_analyze_collection_tool,
    nir_analyze_tool,
    nir_compare_tool,
    nir_register_model_tool,
    nir_train_auto_split_model_tool,
    nir_train_model_tool,
    nir_train_multi_model_tool,
    nir_train_partitioned_model_tool,
)
from .preprocess import nir_preprocess_tool
from .reflect import nir_reflect_tool
from .workflow import nir_workflow_tool

__all__ = [
    # Tools
    "nir_load_data_tool",
    "nir_inspect_tool",
    "nir_preprocess_tool",
    "nir_train_auto_split_model_tool",
    "nir_train_model_tool",
    "nir_train_classifier_tool",
    "nir_train_partitioned_model_tool",
    "nir_train_multi_model_tool",
    "nir_predict_tool",
    "nir_analyze_collection_tool",
    "nir_analyze_tool",
    "nir_reflect_tool",
    "nir_compare_tool",
    "nir_register_model_tool",
    "nir_search_knowledge_tool",
    "nir_workflow_tool",
    # Helpers
    "_resolve",
    "_resolve_writable_dir",
    "_err",
    "_ok",
    "_json_default",
    "_parse_pipeline_step",
    "_build_knowledge_hint",
    "_KNOWN_DOMAINS",
    "_build_report",
    "logger",
    # Legacy knowledge helpers retained for downstream monkeypatching.
    "_RETRY_INTERVAL",
    "_get_knowledge_http_url",
    "_knowledge_http_call_count",
    "_knowledge_http_url",
    "_knowledge_search_mode",
    "_search_knowledge_via_http",
]
