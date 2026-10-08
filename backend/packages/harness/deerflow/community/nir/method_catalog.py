"""Application-owned method references, shared by management and trainers."""

from nir_core.knowledge.method_store import MethodCatalogStore
from nir_core.knowledge.methods import MethodKnowledgeBase

from deerflow.config.paths import get_paths


def get_method_store() -> MethodCatalogStore:
    return MethodCatalogStore(get_paths().base_dir / "knowledge" / "methods.sqlite3")


class ManagedMethodKnowledgeBase:
    """Resolve edits inside retrieval so training retains its failure fallback."""

    def search(self, query: str, **kwargs) -> dict:
        return MethodKnowledgeBase(payload=get_method_store().snapshot()["payload"]).search(query, **kwargs)


def get_method_knowledge_base() -> ManagedMethodKnowledgeBase:
    return ManagedMethodKnowledgeBase()
