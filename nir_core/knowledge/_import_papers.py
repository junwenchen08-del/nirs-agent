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

from nir_core.knowledge.chunker import chunk_document
from nir_core.knowledge.config import get_config
from nir_core.knowledge.entity_extractor import extract_entities
from nir_core.knowledge.parser import parse_document
from nir_core.knowledge.vectorstore import ChromaDBRetriever


def _resolve_papers_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if os.environ.get("NIR_PAPERS_DIR"):
        return Path(os.environ["NIR_PAPERS_DIR"])
    raise SystemExit("Usage: python -m nir_core.knowledge._import_papers <papers_dir> or set NIR_PAPERS_DIR")


papers_dir = _resolve_papers_dir()
files = sorted(f for f in papers_dir.iterdir() if f.suffix.lower() == ".pdf")
print(f"Found {len(files)} PDF files")

cfg = get_config()
model_path = str(Path(__file__).parent / "all-MiniLM-L6-v2")
retriever = ChromaDBRetriever(
    db_path=cfg.chroma_path,
    embedding_model=model_path,
    collection_name=cfg.collection_name,
)

total = 0
for i, f in enumerate(files, 1):
    try:
        md = parse_document(str(f))
        print(f"  [{i}/{len(files)}] {f.name}: parsed {len(md)} chars")
        chunks = chunk_document(md, source=f.name, strategy="section", max_tokens=512)
        print(f"       -> {len(chunks)} chunks")
        entities = extract_entities(md)
        print(
            f"       entities: methods={entities['methods']}, "
            f"models={entities['models']}, datasets={entities['datasets']}"
        )
        for c in chunks:
            c.metadata.update(entities)
            c.metadata["title"] = f.stem
        retriever.add_documents(chunks)
        total += len(chunks)
    except Exception as e:  # noqa: BLE001
        print(f"  [{i}/{len(files)}] {f.name}: ERROR {type(e).__name__}: {e}")

print(f"\nDone: {len(files)} files, {total} chunks indexed")
docs = retriever.list_documents()
print(f"Documents in KB: {len(docs)}")
for d in docs:
    print(
        f"  doc_id={d['doc_id']}, title={d.get('title', '')}, "
        f"chunks={d['chunk_count']}"
    )
