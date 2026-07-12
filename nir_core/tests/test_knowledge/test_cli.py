"""Tests for the CLI argument wiring and non-DB commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from nir_core.knowledge.cli import build_parser


def test_build_parser_has_all_subcommands() -> None:
    """The parser exposes all expected subcommands."""
    parser = build_parser()
    # Each subcommand should be parseable without errors.
    for cmd in ("add", "import-dir", "list", "search", "delete", "rebuild", "stats"):
        # parse_args returns Namespace; we just check it doesn't raise.
        if cmd == "add":
            args = parser.parse_args([cmd, "paper.pdf", "--year", "2023"])
            assert args.command == "add"
            assert args.year == 2023
        elif cmd == "import-dir":
            args = parser.parse_args([cmd, "./papers/"])
            assert args.command == "import-dir"
            assert args.directory == "./papers/"
        elif cmd == "list":
            args = parser.parse_args([cmd])
            assert args.command == "list"
        elif cmd == "search":
            args = parser.parse_args([cmd, "SNV soil", "--top-k", "3"])
            assert args.command == "search"
            assert args.top_k == 3
        elif cmd == "delete":
            args = parser.parse_args([cmd, "doc123"])
            assert args.command == "delete"
            assert args.doc_id == "doc123"
        elif cmd == "rebuild":
            args = parser.parse_args([cmd])
            assert args.command == "rebuild"
        elif cmd == "stats":
            args = parser.parse_args([cmd])
            assert args.command == "stats"


def test_parser_no_command_errors() -> None:
    """No subcommand raises SystemExit."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_add_with_all_options() -> None:
    """The add command accepts all optional metadata."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "add",
            "paper.pdf",
            "--title",
            "My Paper",
            "--year",
            "2023",
            "--authors",
            "Alice",
            "Bob",
        ]
    )
    assert args.title == "My Paper"
    assert args.authors == ["Alice", "Bob"]


def test_cmd_rebuild_nonexistent_path(tmp_path: Path) -> None:
    """rebuild handles a non-existent DB path gracefully."""
    from nir_core.knowledge.config import KnowledgeConfig, set_config

    cfg = KnowledgeConfig(chroma_path=str(tmp_path / "nonexistent_db"))
    set_config(cfg)

    from nir_core.knowledge.cli import cmd_rebuild
    import argparse

    args = argparse.Namespace()
    # Should not raise even when the path doesn't exist.
    result = cmd_rebuild(args)
    assert result == 0


def test_cmd_rebuild_existing_path(tmp_path: Path) -> None:
    """rebuild clears an existing DB directory."""
    from nir_core.knowledge.config import KnowledgeConfig, set_config

    db_path = tmp_path / "test_db"
    db_path.mkdir()
    (db_path / "dummy.txt").write_text("dummy")

    cfg = KnowledgeConfig(chroma_path=str(db_path))
    set_config(cfg)

    from nir_core.knowledge.cli import cmd_rebuild
    import argparse

    args = argparse.Namespace()
    result = cmd_rebuild(args)
    assert result == 0
    assert not db_path.exists()  # directory was removed
