# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Library manager with a staging inbox and JSON sidecar metadata."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

from .context import LibraryContext


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LibraryManager:
    """Manage a library tree plus a hidden inbox for staged files."""

    _context: LibraryContext
    _root: Path
    _inbox: Path

    def __init__(self, root_path: str, inbox_name: str = ".inbox") -> None:
        self._context = LibraryContext.from_root(root_path)
        self._root = self._context.root
        self._inbox = self._root / inbox_name
        self._inbox.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def inbox(self) -> Path:
        return self._inbox

    def stage_copy(self, source_path: str) -> Path:
        """Copy an external file into the inbox and create metadata."""
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        # TODO: detect duplicate content by hash before staging.
        dest = self._unique_path(self._inbox / source.name)
        shutil.copy2(source, dest)
        metadata = {
            "id": str(uuid.uuid4()),
            "path": str(dest),
            "hash": _hash_file(dest),
            "added_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._write_metadata(dest, metadata)
        return dest

    def move_file(self, source_path: str, dest_path: str) -> Path:
        """Move a file within the library; destination cannot be in the inbox."""
        source = self._context.resolve_path(source_path)
        dest = self._context.resolve_path(dest_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        if self._is_in_inbox(dest):
            raise ValueError(f"Destination cannot be inside inbox: {dest}")
        if dest.exists():
            raise FileExistsError(f"Destination already exists: {dest}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        source_meta = self._metadata_path(source)
        dest_meta = self._metadata_path(dest)
        source.rename(dest)
        if source_meta.exists():
            source_meta.rename(dest_meta)
            metadata = self._read_metadata(dest_meta)
            metadata["path"] = str(dest)
            metadata["updated_at"] = _utc_now()
            self._write_metadata(dest, metadata)
        self._cleanup_empty_dirs(source.parent)
        return dest

    def read_metadata(self, file_path: str) -> Dict[str, Any]:
        path = self._context.resolve_path(file_path)
        meta_path = self._metadata_path(path)
        return self._read_metadata(meta_path)

    def update_metadata(
        self, file_path: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        path = self._context.resolve_path(file_path)
        meta_path = self._metadata_path(path)
        metadata = self._read_metadata(meta_path)
        metadata.update(updates)
        metadata["path"] = str(path)
        metadata["updated_at"] = _utc_now()
        self._write_metadata(path, metadata)
        return metadata

    def _metadata_path(self, file_path: Path) -> Path:
        return file_path.with_name(f"{file_path.name}.meta.json")

    def _read_metadata(self, meta_path: Path) -> Dict[str, Any]:
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_metadata(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        meta_path = self._metadata_path(file_path)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _cleanup_empty_dirs(self, start_dir: Path) -> None:
        current = start_dir
        while True:
            if current == self._root or current == self._inbox:
                return
            if not current.is_dir():
                return
            try:
                next(current.iterdir())
                return
            except StopIteration:
                current.rmdir()
                current = current.parent

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for idx in range(1, 1000):
            candidate = path.with_name(f"{stem}-{idx}{suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Failed to find free name for: {path}")

    def _is_in_inbox(self, path: Path) -> bool:
        try:
            path.relative_to(self._inbox)
            return True
        except ValueError:
            return False
