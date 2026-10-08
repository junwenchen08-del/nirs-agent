"""Disabled-by-default capacity policy for the user-scoped NIR library.

Batch B validates and exposes configuration only. Dataset/Model services in
later batches enforce these admission limits before writing bytes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NIRLibraryConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable the persistent, owner-scoped NIR dataset/model library. Requires sqlite or postgres.")
    auto_save_uploads: bool = Field(default=True, description="Automatically save successful CSV/TXT/MAT HTTP uploads when the library is enabled; duplicate content reuses an existing asset.")
    auto_save_registered_models: bool = Field(default=True, description="Automatically persist an approved model after successful agent registration when the library is enabled; explicit user refusal skips storage.")
    max_dataset_file_size: int = Field(default=50 * 1024 * 1024, gt=0, description="Maximum source file size in bytes accepted for library saving.")
    max_user_dataset_bytes: int = Field(default=5 * 1024**3, gt=0, description="Owner-scoped long-term dataset bytes admitted by the library.")
    max_user_model_bytes: int = Field(default=5 * 1024**3, gt=0, description="Owner-scoped long-term model-package bytes admitted by the library.")
    max_total_bytes_for_library_writes: int = Field(
        default=15 * 1024**3,
        gt=0,
        description="Admission ceiling for new library writes, accounting for current thread bytes; not a strict cap on uploads or sandbox writes.",
    )
    max_model_versions_per_id: int = Field(default=20, gt=0, description="Maximum ready versions of one logical model before archiving or deletion.")
    min_free_disk_bytes: int = Field(default=2 * 1024**3, ge=0, description="Disk free-space floor required before new library writes.")
    staging_ttl_hours: int = Field(default=24, gt=0, description="Age after which only service-created abandoned staging directories may be cleaned.")
