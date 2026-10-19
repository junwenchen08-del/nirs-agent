"""Durable reviewed overrides of the packaged method reference snapshot."""

from __future__ import annotations

import copy
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .method_registry import (
    MODEL_REFERENCES,
    runtime_reference,
    validate_reference_params,
)

from .methods import (
    DEFAULT_CARDS_PATH,
    PROBLEM_LABELS,
    MethodKnowledgeBase,
    method_eligibility,
)

EDITABLE = {
    "title",
    "source_url",
    "source_section",
    "summary_zh",
    "keywords",
    "problem_tags",
    "avoid_tags",
    "candidate_params",
    "planning_notes_zh",
    "priority",
    "review_status",
    "applicability_zh",
    "limitations_zh",
    "parameter_guidance_zh",
}
TEXT_LIMITS = {
    "title": 150,
    "source_url": 500,
    "source_section": 250,
    "summary_zh": 2000,
    "keywords": 1000,
    "planning_notes_zh": 2000,
    "applicability_zh": 2000,
    "limitations_zh": 2000,
    "parameter_guidance_zh": 2000,
}


class CatalogConflict(ValueError):
    """An edit refers to a revision that is no longer current."""


def provider_binding(method_id: str) -> dict:
    reference = runtime_reference(method_id)
    return {
        key: reference[key]
        for key in (
            "method_kind",
            "provider",
            "provider_class",
            "runtime_provider_version",
            "regular_axis_required",
        )
    }


def validate_card(card: dict) -> None:
    provider_binding(card["method_id"])
    for field, limit in TEXT_LIMITS.items():
        value = card.get(field, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{field} must be text of at most {limit} characters")
    if not card.get("title", "").strip():
        raise ValueError("title is required")
    if card.get("review_status") not in {"draft", "published", "retired"}:
        raise ValueError("review_status must be draft, published or retired")
    for field in ("problem_tags", "avoid_tags"):
        value = card.get(field, [])
        if (
            not isinstance(value, list)
            or len(value) > len(PROBLEM_LABELS)
            or any(tag not in PROBLEM_LABELS for tag in value)
        ):
            raise ValueError(f"{field} contains an unsupported diagnostic tag")
    priority = card.get("priority", 99)
    if type(priority) is not int or not 0 <= priority <= 99:
        raise ValueError("priority must be an integer from 0 to 99")
    url = urlsplit(card.get("source_url", ""))
    allowed = {
        "chemotools.org": "/methods/",
        "scikit-learn.org": "/stable/modules/",
        "docs.scipy.org": "/doc/scipy/reference/generated/",
    }
    if (
        url.scheme != "https"
        or url.netloc not in allowed
        or not url.path.startswith(allowed.get(url.netloc, "!"))
        or url.query
    ):
        raise ValueError(
            "source_url must be an HTTPS official Chemotools, scikit-learn or SciPy reference page"
        )
    variants = card.get("candidate_params")
    if not isinstance(variants, list) or not 1 <= len(variants) <= 8:
        raise ValueError("candidate_params requires 1 to 8 parameter combinations")
    for params in variants:
        if not isinstance(params, dict):
            raise ValueError("each candidate must be a parameter object")
        json.dumps(params, allow_nan=False)
        valid, reason = validate_reference_params(card["method_id"], params)
        if not valid:
            raise ValueError(reason)
    if card["review_status"] == "published":
        if not all(
            card.get(field)
            for field in ("summary_zh", "source_section", "problem_tags", "keywords")
        ):
            raise ValueError(
                "publication requires summary, source section, problem tags and keywords"
            )
        reason = method_eligibility(card, None)
        if reason in {"method_not_available", "runtime_version_mismatch"}:
            raise ValueError(reason)


class MethodCatalogStore:
    """Whole-catalog SQLite transactions with optimistic revision checks."""

    def __init__(self, path: Path, *, defaults_path: Path = DEFAULT_CARDS_PATH):
        self.path = Path(path)
        self.defaults_path = Path(defaults_path)

    def defaults(self) -> dict:
        return json.loads(self.defaults_path.read_text(encoding="utf-8"))

    def _merge_defaults(self, payload: dict, revision: int) -> dict:
        """Add newly packaged methods/fields while preserving all managed edits."""
        payload = copy.deepcopy(payload)
        defaults = self.defaults()
        existing = {card["method_id"]: card for card in payload["cards"]}
        for default in defaults["cards"]:
            if default["method_id"] not in existing:
                payload["cards"].append(copy.deepcopy(default))
            else:
                for key, value in default.items():
                    existing[default["method_id"]].setdefault(key, copy.deepcopy(value))
        payload["index_version"] = f"{defaults['index_version']}:managed-{revision}"
        return payload

    def snapshot(self) -> dict:
        if self.path.exists():
            with sqlite3.connect(self.path) as connection:
                row = connection.execute(
                    "SELECT revision,payload FROM catalog WHERE id=1"
                ).fetchone()
            if row:
                return {
                    "revision": row[0],
                    "payload": self._merge_defaults(json.loads(row[1]), row[0]),
                }
        return {"revision": 0, "payload": self.defaults()}

    def history(self) -> list[dict]:
        if not self.path.exists():
            return []
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT revision,method_id,action,actor,changed_at FROM history ORDER BY revision DESC LIMIT 30"
            ).fetchall()
        return [
            dict(zip(("revision", "method_id", "action", "actor", "changed_at"), row))
            for row in rows
        ]

    def update(
        self, method_id: str, changes: dict, *, expected_revision: int, actor: str
    ) -> dict:
        if not isinstance(changes, dict) or not changes or set(changes) - EDITABLE:
            raise ValueError("changes contains unsupported or immutable fields")
        return self._write(method_id, changes, expected_revision, actor, reset=False)

    def reset(self, method_id: str, *, expected_revision: int, actor: str) -> dict:
        return self._write(method_id, {}, expected_revision, actor, reset=True)

    def _write(
        self,
        method_id: str,
        changes: dict,
        expected_revision: int,
        actor: str,
        *,
        reset: bool,
    ) -> dict:
        binding = provider_binding(method_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS catalog (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS history (revision INTEGER PRIMARY KEY, method_id TEXT, action TEXT, actor TEXT, changed_at TEXT)"
            )
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision,payload FROM catalog WHERE id=1"
            ).fetchone()
            revision, payload = (
                (row[0], json.loads(row[1])) if row else (0, self.defaults())
            )
            payload = self._merge_defaults(payload, revision)
            if type(expected_revision) is not int or expected_revision != revision:
                raise CatalogConflict(
                    "Catalog changed; refresh before saving / 知识库已更新，请刷新后再保存"
                )
            cards = payload["cards"]
            index = next(
                (i for i, item in enumerate(cards) if item["method_id"] == method_id),
                None,
            )
            if reset:
                default = next(
                    (
                        item
                        for item in self.defaults()["cards"]
                        if item["method_id"] == method_id
                    ),
                    None,
                )
                if default is None:
                    raise ValueError("method has no packaged default to restore")
                card = copy.deepcopy(default)
            else:
                card = (
                    copy.deepcopy(cards[index])
                    if index is not None
                    else {
                        "method_id": method_id,
                        **binding,
                        "documentation_version": "unversioned_live_site",
                        "documented_parameters": {},
                        "parameter_mapping": {},
                        "avoid_tags": [],
                        "candidate_params": [{}],
                        "priority": 99,
                        "review_status": "draft",
                        "planning_notes_zh": "",
                        "source_section": "",
                        "summary_zh": "",
                        "keywords": "",
                        "problem_tags": [],
                    }
                )
                card.update(copy.deepcopy(changes))
                if index is None:
                    card["retrieved_on"] = datetime.now(timezone.utc).date().isoformat()
            # Packaged cards inherit version metadata; persist it explicitly.
            for key in ("runtime_provider_version", "documentation_version"):
                card.setdefault(key, payload[key])
            validate_card(card)
            if index is None:
                cards.append(card)
            else:
                cards[index] = card
            revision += 1
            payload["index_version"] = (
                f"{self.defaults()['index_version']}:managed-{revision}"
            )
            serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            connection.execute(
                "INSERT OR REPLACE INTO catalog VALUES (1,?,?)", (revision, serialized)
            )
            connection.execute(
                "INSERT INTO history VALUES (?,?,?,?,?)",
                (
                    revision,
                    method_id,
                    "reset" if reset else "update",
                    actor,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.execute("COMMIT")
            return {"revision": revision, "payload": payload}
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def view(self) -> dict:
        from nir_core.preprocess.registry import METHOD_REGISTRY
        from .method_inventory import CATEGORY_LABELS, coverage, enrich_card, inventory

        snapshot = self.snapshot()
        cards = MethodKnowledgeBase(payload=snapshot["payload"]).cards
        defaults = {card["method_id"] for card in self.defaults()["cards"]}
        cards = [enrich_card(card) for card in cards]
        for card in cards:
            binding = runtime_reference(card["method_id"])
            reason = method_eligibility(card, None)
            card.update(
                runtime_parameters=binding["runtime_parameters"],
                auto_eligible=reason is None,
                eligibility_reason=reason,
                has_default=card["method_id"] in defaults,
            )
        available = []
        covered_mcp = {card.get("mcp_capability_id") for card in cards}
        for method_id in [*METHOD_REGISTRY, *MODEL_REFERENCES, *inventory()["entries"]]:
            if method_id in covered_mcp:
                continue
            try:
                reference = runtime_reference(method_id)
            except ValueError:
                continue
            available.append(reference)
        return {
            "revision": snapshot["revision"],
            "index_version": snapshot["payload"]["index_version"],
            "cards": cards,
            "count": len(cards),
            "available_methods": available,
            "problem_tags": PROBLEM_LABELS,
            "category_labels": CATEGORY_LABELS,
            "coverage": coverage(cards),
            "history": self.history(),
        }
