"""Desktop method-reference management without an external paper RAG server."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from nir_core.knowledge.method_store import CatalogConflict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.gateway.deps import require_admin_user
from deerflow.community.nir import method_catalog

router = APIRouter(prefix="/api/method-knowledge", tags=["method-knowledge"])


class MethodEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    changes: dict


class MethodReset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)


class MethodSearch(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=12)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


@router.get("/cards")
async def list_cards(request: Request):
    result = await asyncio.to_thread(lambda: method_catalog.get_method_store().view())
    result["can_edit"] = getattr(getattr(request.state, "user", None), "system_role", None) == "admin"
    return result


@router.post("/search")
async def search_cards(body: MethodSearch):
    return await asyncio.to_thread(lambda: method_catalog.get_method_knowledge_base().search(body.query, top_k=body.top_k))


async def _change(request: Request, method_id: str, body: MethodEdit | MethodReset, *, reset: bool):
    await require_admin_user(request, detail="Admin privileges required to edit shared method references / 管理员可修改共享方法知识库")
    user = getattr(request.state, "user", None)
    actor = str(getattr(user, "id", getattr(user, "user_id", "admin")))

    def write():
        store = method_catalog.get_method_store()
        if reset:
            store.reset(method_id, expected_revision=body.expected_revision, actor=actor)
        else:
            store.update(method_id, body.changes, expected_revision=body.expected_revision, actor=actor)
        return {**store.view(), "can_edit": True}

    try:
        return await asyncio.to_thread(write)
    except CatalogConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.put("/cards/{method_id}")
async def save_card(method_id: str, body: MethodEdit, request: Request):
    return await _change(request, method_id, body, reset=False)


@router.post("/cards/{method_id}/reset")
async def reset_card(method_id: str, body: MethodReset, request: Request):
    return await _change(request, method_id, body, reset=True)
