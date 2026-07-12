"""NIR knowledge base CLI.

Usage::

    python -m nir_core.knowledge.cli add paper.pdf --title "..." --year 2023
    python -m nir_core.knowledge.cli import-dir ./papers/
    python -m nir_core.knowledge.cli list
    python -m nir_core.knowledge.cli search "SNV scatter correction"
    python -m nir_core.knowledge.cli delete <doc_id>
    python -m nir_core.knowledge.cli rebuild
    python -m nir_core.knowledge.cli stats

The CLI is intentionally thin — it's a convenience wrapper around the
``parser → chunker → entity_extractor → retriever`` pipeline so users can
populate the knowledge base without writing code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nir_core.knowledge.chunker import chunk_document
from nir_core.knowledge.config import get_config, get_retriever
from nir_core.knowledge.entity_extractor import extract_entities
from nir_core.knowledge.parser import SUPPORTED_EXTENSIONS, parse_document


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_add(args: argparse.Namespace) -> int:
    """Add a single document to the knowledge base."""
    retriever = get_retriever()
    cfg = get_config()

    markdown = parse_document(args.file_path)
    doc_id = Path(args.file_path).stem

    chunks = chunk_document(
        markdown,
        source=args.file_path,
        strategy=cfg.chunk_strategy,
        max_tokens=cfg.chunk_max_tokens,
        doc_id=doc_id,
    )

    entities = extract_entities(markdown)
    for chunk in chunks:
        chunk.metadata.update(
            {
                "title": args.title or doc_id,
                "year": args.year,
                **entities,
            }
        )

    count = retriever.add_documents(chunks)
    print(f"Added {count} chunks from {args.file_path} (doc_id={doc_id})")
    if entities["methods"]:
        print(f"  Methods detected: {', '.join(entities['methods'])}")
    if entities["models"]:
        print(f"  Models detected: {', '.join(entities['models'])}")
    return 0


def cmd_import_dir(args: argparse.Namespace) -> int:
    """Recursively import all supported documents from a directory."""
    retriever = get_retriever()
    cfg = get_config()

    dir_path = Path(args.directory)
    files = [
        f
        for f in dir_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"No supported files found in {args.directory}")
        return 0

    print(f"Found {len(files)} files to import...")

    total_chunks = 0
    errors = 0
    for i, file_path in enumerate(files, 1):
        try:
            markdown = parse_document(str(file_path))
            doc_id = file_path.stem

            chunks = chunk_document(
                markdown,
                source=str(file_path),
                strategy=cfg.chunk_strategy,
                max_tokens=cfg.chunk_max_tokens,
                doc_id=doc_id,
            )

            entities = extract_entities(markdown)
            for chunk in chunks:
                chunk.metadata.update({"title": doc_id, **entities})

            retriever.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  [{i}/{len(files)}] {file_path.name} -> {len(chunks)} chunks")
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [{i}/{len(files)}] {file_path.name} -> ERROR: {exc}")

    print(
        f"\nDone: {len(files) - errors}/{len(files)} files, "
        f"{total_chunks} chunks total"
    )
    return 0 if errors == 0 else 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all documents in the knowledge base."""
    retriever = get_retriever()
    docs = retriever.list_documents()

    if not docs:
        print("Knowledge base is empty.")
        return 0

    print(f"{'doc_id':<40} {'title':<40} {'year':>6} {'chunks':>7}")
    print("-" * 95)
    for doc in docs:
        print(
            f"{doc.get('doc_id', ''):<40} "
            f"{doc.get('title', ''):<40} "
            f"{str(doc.get('year', '')):>6} "
            f"{doc.get('chunk_count', 0):>7}"
        )
    print(f"\nTotal: {len(docs)} documents")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search the knowledge base and print results."""
    retriever = get_retriever()
    results = retriever.search(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return 0

    for i, r in enumerate(results, 1):
        print(f"\n{'=' * 70}")
        print(f"Result {i} (score: {r.score:.4f})")
        print(f"Source: {r.chunk.source}")
        methods = r.chunk.metadata.get("methods", "")
        if isinstance(methods, str):
            try:
                methods = json.loads(methods)
            except (json.JSONDecodeError, TypeError):
                methods = [methods] if methods else []
        if methods:
            print(f"Methods: {methods}")
        print(f"{'=' * 70}")
        print(r.chunk.content[:500])
        if len(r.chunk.content) > 500:
            print("...")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a document by doc_id."""
    retriever = get_retriever()
    success = retriever.delete_document(args.doc_id)
    if success:
        print(f"Deleted document: {args.doc_id}")
    else:
        print(f"Not found or delete failed: {args.doc_id}")
        return 1
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Clear the vector DB (documents must be re-imported afterwards)."""
    cfg = get_config()
    db_path = Path(cfg.chroma_path)

    if db_path.exists():
        import shutil

        shutil.rmtree(db_path)
        print(f"Cleared {db_path}")
    else:
        print(f"DB path does not exist: {db_path}")

    print("Rebuild complete. Use 'import-dir' to re-import documents.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Print knowledge base statistics."""
    retriever = get_retriever()
    cfg = get_config()
    docs = retriever.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    print(f"Documents: {len(docs)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Database path: {cfg.chroma_path}")
    print(f"Embedding model: {cfg.embedding_model}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nir_core.knowledge.cli",
        description="NIR knowledge base CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a single document")
    p_add.add_argument("file_path", help="Path to document")
    p_add.add_argument("--title", help="Document title")
    p_add.add_argument("--year", type=int, help="Publication year")
    p_add.add_argument("--authors", nargs="*", help="Author names")
    p_add.set_defaults(func=cmd_add)

    p_import = sub.add_parser("import-dir", help="Import all documents from a directory")
    p_import.add_argument("directory", help="Directory path")
    p_import.set_defaults(func=cmd_import_dir)

    p_list = sub.add_parser("list", help="List all documents")
    p_list.set_defaults(func=cmd_list)

    p_search = sub.add_parser("search", help="Search knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    p_del = sub.add_parser("delete", help="Delete a document")
    p_del.add_argument("doc_id", help="Document ID")
    p_del.set_defaults(func=cmd_delete)

    p_rebuild = sub.add_parser("rebuild", help="Rebuild index (clears all data)")
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_stats = sub.add_parser("stats", help="Show statistics")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
