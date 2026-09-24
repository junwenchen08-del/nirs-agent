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

from nir_core.knowledge.config import get_config, get_retriever
from nir_core.knowledge.governance import KnowledgeCatalog, require_publication_ready
from nir_core.knowledge.ingestion import ingest_document_bytes
from nir_core.knowledge.migration import (
    ChromaMetadataStore,
    apply_metadata_migration,
    audit_catalog,
    backup_knowledge_store,
    load_migration_manifest,
    migration_results_json,
    preflight_migration,
)
from nir_core.knowledge.parser import SUPPORTED_EXTENSIONS

# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_add(args: argparse.Namespace) -> int:
    """Add a single document to the knowledge base."""
    retriever = get_retriever()
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)
    path = Path(args.file_path)
    result = ingest_document_bytes(
        filename=path.name,
        content=path.read_bytes(),
        retriever=retriever,
        catalog=catalog,
        title=args.title,
        authors=args.authors,
        year=args.year,
        doi=args.doi,
        source_type=args.source_type,
        domains=args.domains,
        quality_tier=args.quality_tier,
        review_status=args.review_status,
        chunk_strategy=cfg.chunk_strategy,
        max_tokens=cfg.chunk_max_tokens,
        overlap_percent=cfg.chunk_overlap,
        index_version=cfg.index_version,
    )
    print(
        f"{result.action}: {result.chunks_added} chunks from {args.file_path} "
        f"(doc_id={result.record.doc_id}, version={result.record.version}, "
        f"status={result.record.review_status})"
    )
    return 0


def cmd_import_dir(args: argparse.Namespace) -> int:
    """Recursively import all supported documents from a directory."""
    retriever = get_retriever()
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)

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
            result = ingest_document_bytes(
                filename=file_path.name,
                content=file_path.read_bytes(),
                retriever=retriever,
                catalog=catalog,
                title=file_path.stem,
                quality_tier=args.quality_tier,
                review_status=args.review_status,
                chunk_strategy=cfg.chunk_strategy,
                max_tokens=cfg.chunk_max_tokens,
                overlap_percent=cfg.chunk_overlap,
                index_version=cfg.index_version,
            )
            total_chunks += result.chunks_added
            print(
                f"  [{i}/{len(files)}] {file_path.name} -> {result.action}, "
                f"{result.chunks_added} chunks"
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [{i}/{len(files)}] {file_path.name} -> ERROR: {exc}")

    print(
        f"\nDone: {len(files) - errors}/{len(files)} files, {total_chunks} chunks total"
    )
    return 0 if errors == 0 else 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all documents in the knowledge base."""
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)
    docs = [record.model_dump() for record in catalog.list()]

    if not docs:
        print("Knowledge base is empty.")
        return 0

    print(f"{'doc_id':<48} {'title':<32} {'status':<12} {'ver':>3}")
    print("-" * 100)
    for doc in docs:
        print(
            f"{doc.get('doc_id', ''):<48} "
            f"{doc.get('title', ''):<32} "
            f"{doc.get('review_status', ''):<12} "
            f"{doc.get('version', 1):>3}"
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
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)
    vector_deleted = retriever.delete_document(args.doc_id)
    catalog_deleted = catalog.delete(args.doc_id)
    if vector_deleted or catalog_deleted:
        print(f"Deleted document: {args.doc_id}")
    else:
        print(f"Not found or delete failed: {args.doc_id}")
        return 1
    return 0


def cmd_set_status(args: argparse.Namespace) -> int:
    """Set draft/review/publication state and mirror it into vector metadata."""
    retriever = get_retriever()
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)
    record = catalog.get(args.doc_id)
    if record is None:
        print(f"Document not found: {args.doc_id}")
        return 1
    if args.review_status == "published":
        require_publication_ready(record)
    if not retriever.update_document_metadata(
        args.doc_id, {"review_status": args.review_status}
    ):
        print(f"Vector chunks not found: {args.doc_id}")
        return 1
    updated = catalog.set_review_status(args.doc_id, args.review_status)
    print(f"Document status: {args.doc_id} -> {updated.review_status}")
    return 0


def cmd_audit_metadata(args: argparse.Namespace) -> int:
    """Report publication readiness without mutating the knowledge base."""
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)
    audit = audit_catalog(catalog)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def cmd_migrate_metadata(args: argparse.Namespace) -> int:
    """Back up and apply a hash-pinned legacy metadata migration."""
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)
    manifest = load_migration_manifest(args.manifest)
    preflight_migration(catalog=catalog, manifest=manifest)
    if not args.dry_run:
        if not args.backup_dir:
            raise ValueError("--backup-dir is required unless --dry-run is used")
        backup = backup_knowledge_store(
            catalog_path=cfg.catalog_path,
            chroma_path=cfg.chroma_path,
            destination=args.backup_dir,
        )

    metadata_store = ChromaMetadataStore(
        db_path=cfg.chroma_path,
        collection_name=cfg.collection_name,
    )

    preview = apply_metadata_migration(
        catalog=catalog,
        metadata_store=metadata_store,
        manifest=manifest,
        dry_run=True,
    )
    if args.dry_run:
        print(json.dumps(migration_results_json(preview), ensure_ascii=False, indent=2))
        return 0
    results = apply_metadata_migration(
        catalog=catalog,
        metadata_store=metadata_store,
        manifest=manifest,
    )
    print(
        json.dumps(
            {
                "migration_id": manifest.migration_id,
                "backup": str(backup),
                "results": migration_results_json(results),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
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

    catalog_path = Path(cfg.catalog_path)
    if catalog_path.exists() and not catalog_path.is_relative_to(db_path):
        catalog_path.unlink()
        print(f"Cleared {catalog_path}")

    print("Rebuild complete. Use 'import-dir' to re-import documents.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Print knowledge base statistics."""
    retriever = get_retriever()
    cfg = get_config()
    catalog = KnowledgeCatalog(cfg.catalog_path)
    docs = catalog.list()
    vector_docs = retriever.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in vector_docs)
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
    p_add.add_argument("--doi", help="Digital Object Identifier")
    p_add.add_argument(
        "--source-type", help="journal, standard, internal_document, ..."
    )
    p_add.add_argument("--domains", nargs="*", help="Domain tags")
    p_add.add_argument("--quality-tier", choices=list("ABCDE"), default="C")
    p_add.add_argument(
        "--review-status",
        choices=("draft", "needs_review", "published", "retired"),
        default="draft",
    )
    p_add.set_defaults(func=cmd_add)

    p_import = sub.add_parser(
        "import-dir", help="Import all documents from a directory"
    )
    p_import.add_argument("directory", help="Directory path")
    p_import.add_argument("--quality-tier", choices=list("ABCDE"), default="C")
    p_import.add_argument(
        "--review-status",
        choices=("draft", "needs_review", "published", "retired"),
        default="draft",
    )
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

    p_status = sub.add_parser("set-status", help="Set document review status")
    p_status.add_argument("doc_id", help="Stable document ID")
    p_status.add_argument(
        "review_status",
        choices=("draft", "needs_review", "published", "retired"),
    )
    p_status.set_defaults(func=cmd_set_status)

    p_audit = sub.add_parser(
        "audit-metadata",
        help="Report publication-readiness problems for existing documents",
    )
    p_audit.set_defaults(func=cmd_audit_metadata)

    p_migrate = sub.add_parser(
        "migrate-metadata",
        help="Apply a hash-pinned metadata migration without re-embedding",
    )
    p_migrate.add_argument("manifest", help="Migration manifest JSON path")
    p_migrate.add_argument(
        "--backup-dir",
        help="New directory that will receive catalog and Chroma backups",
    )
    p_migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview changes without writing data or a backup",
    )
    p_migrate.set_defaults(func=cmd_migrate_metadata)

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
