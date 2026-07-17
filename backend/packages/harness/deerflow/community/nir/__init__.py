"""NIR community package — @tool functions bridging DeerFlow to nir_core.

The actual implementations live in focused sub-modules (io_tools,
preprocess, modeling, reflect, knowledge). This package init re-exports
the public tool functions for convenience so callers can write::

    from deerflow.community.nir import nir_train_model_tool

For legacy reasons ``from deerflow.community.nir.tools import ...`` also
continues to work (tools.py is a re-export facade).
"""

from .io_tools import nir_inspect_tool, nir_load_data_tool, nir_predict_tool
from .knowledge import nir_search_knowledge_tool
from .modeling import (
    nir_analyze_tool,
    nir_compare_tool,
    nir_register_model_tool,
    nir_train_model_tool,
)
from .preprocess import nir_preprocess_tool
from .reflect import nir_reflect_tool
from .workflow import nir_workflow_tool

__all__ = [
    "nir_load_data_tool",
    "nir_inspect_tool",
    "nir_preprocess_tool",
    "nir_train_model_tool",
    "nir_predict_tool",
    "nir_analyze_tool",
    "nir_reflect_tool",
    "nir_compare_tool",
    "nir_register_model_tool",
    "nir_search_knowledge_tool",
    "nir_workflow_tool",
]
