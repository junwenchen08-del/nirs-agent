"""Temporary script to import papers into the knowledge base."""

import os
import sys
from pathlib import Path

# Set env vars BEFORE any heavy imports.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nir_core.knowledge.config import get_config
from nir_core.knowledge.governance import KnowledgeCatalog
from nir_core.knowledge.ingestion import ingest_document_bytes
from nir_core.knowledge.vectorstore import ChromaDBRetriever


def _resolve_papers_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if os.environ.get("NIR_PAPERS_DIR"):
        return Path(os.environ["NIR_PAPERS_DIR"])
    raise SystemExit(
        "Usage: python -m nir_core.knowledge._import_papers <papers_dir> or set NIR_PAPERS_DIR"
    )


papers_dir = _resolve_papers_dir()
files = sorted(f for f in papers_dir.iterdir() if f.suffix.lower() == ".pdf")
print(f"Found {len(files)} PDF files")

cfg = get_config()
retriever = ChromaDBRetriever(
    db_path=cfg.chroma_path,
    embedding_model=cfg.embedding_model,
    collection_name=cfg.collection_name,
)
catalog = KnowledgeCatalog(cfg.catalog_path)
review_status = os.environ.get("NIR_IMPORT_REVIEW_STATUS", "draft")

total = 0
for i, f in enumerate(files, 1):
    try:
        result = ingest_document_bytes(
            filename=f.name,
            content=f.read_bytes(),
            retriever=retriever,
            catalog=catalog,
            title=f.stem,
            review_status=review_status,
            chunk_strategy=cfg.chunk_strategy,
            max_tokens=cfg.chunk_max_tokens,
            overlap_percent=cfg.chunk_overlap,
            index_version=cfg.index_version,
        )
        print(
            f"  [{i}/{len(files)}] {f.name}: {result.action}, "
            f"{result.chunks_added} chunks, status={result.record.review_status}"
        )
        total += result.chunks_added
    except Exception as e:  # noqa: BLE001
        print(f"  [{i}/{len(files)}] {f.name}: ERROR {type(e).__name__}: {e}")

print(f"\nDone: {len(files)} files, {total} chunks indexed")
docs = retriever.list_documents()
print(f"Documents in KB: {len(docs)}")
for d in docs:
    print(
        f"  doc_id={d['doc_id']}, title={d.get('title', '')}, chunks={d['chunk_count']}"
    )
