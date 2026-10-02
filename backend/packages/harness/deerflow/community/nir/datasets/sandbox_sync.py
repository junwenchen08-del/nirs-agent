"""Synchronize a host-side dataset attachment into non-mounted sandboxes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from deerflow.sandbox.sandbox_provider import get_sandbox_provider


async def sync_attachment_to_sandbox(owner: str, thread_id: str, virtual_path: str, local_path: Path) -> None:
    provider = await asyncio.to_thread(get_sandbox_provider)
    if provider.uses_thread_data_mounts:
        return
    sandbox_id = await provider.acquire_async(thread_id, user_id=owner)
    sandbox = await asyncio.to_thread(provider.get, sandbox_id)
    if sandbox is None:
        raise RuntimeError("Sandbox acquisition returned no instance")
    content = await asyncio.to_thread(local_path.read_bytes)
    await asyncio.to_thread(sandbox.update_file, virtual_path, content)
