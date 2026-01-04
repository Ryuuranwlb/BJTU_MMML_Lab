"""Library manager with a staging inbox and JSON sidecar metadata."""

from __future__ import annotations

import base64
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional
from pypdf import PdfReader

from .context import LibraryContext

logger = logging.getLogger(__name__)


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
    _llm_client: Any

    def __init__(
        self,
        root_path: str,
        inbox_name: str = ".inbox",
        llm_client: Optional[Any] = None
    ) -> None:
        self._context = LibraryContext.from_root(root_path)
        self._root = self._context.root
        self._inbox = self._root / inbox_name
        self._inbox.mkdir(parents=True, exist_ok=True)

        if llm_client is None:
            from pangu_agent.llm_client import LLMClient
            self._llm_client = LLMClient()
        else:
            self._llm_client = llm_client

    @property
    def root(self) -> Path:
        return self._root

    @property
    def inbox(self) -> Path:
        return self._inbox

    def stage_copy(self, source_path: str) -> Path:
        """Copy an external file into the inbox and create metadata.

        Args:
            source_path: Path to the source file to stage

        Returns:
            Path to the staged file in the inbox
        """
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

        # Generate description via LLM
        try:
            description = self._generate_description(dest)
            if description:
                metadata["description"] = description
                logger.info(f"Generated description for {dest.name}")
        except Exception as exc:
            logger.warning(f"Failed to generate description for {dest.name}: {exc}")

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

    def list_children(self, path: Path | str) -> List[Path]:
        if isinstance(path, str):
            start = self._context.resolve_path(path)
        else:
            start = path.expanduser().resolve()
            try:
                start.relative_to(self._root)
            except ValueError as exc:
                raise ValueError(f"Path escapes library root: {start}") from exc
        if not start.exists():
            raise FileNotFoundError(f"Path not found: {start}")
        if not start.is_dir():
            raise NotADirectoryError(f"Not a directory: {start}")
        try:
            return sorted(start.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as exc:
            raise OSError(f"Failed to list directory '{start}': {exc}") from exc

    def get_entry(self, path: str) -> Path:
        entry = self._context.resolve_path(path)
        if not entry.exists():
            raise FileNotFoundError(f"Path not found: {entry}")
        return entry

    def read_file(self, path: str, include_content: bool = True, include_meta: bool = True) -> Dict[str, Any]:
        file_path = self._context.resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        result: Dict[str, Any] = {"path": path}

        # Read metadata if requested
        if include_meta:
            meta_path = self._metadata_path(file_path)
            meta_data = self._read_metadata(meta_path)
            result["meta_data"] = meta_data

        # Read content if requested
        if include_content:
            file_type = self._detect_file_type(file_path)
            if file_type == "pdf":
                text = self._read_pdf(file_path)
                result["kind"] = "text"
                result["text"] = text
            elif file_type == "image":
                data = file_path.read_bytes()
                data_url = f"data:image/{file_path.suffix[1:]};base64,{base64.b64encode(data).decode('ascii')}"
                result["kind"] = "image"
                result["image_url"] = {"url": data_url}
            else:
                raise ValueError("Only image or PDF files are supported")

        return result

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

    def _detect_file_type(self, path: Path) -> Optional[str]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        image_types = [".jpg", ".jpeg", ".png"]
        if suffix in image_types:
            return "image"
        return None

    def _read_pdf(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to read PDF: {path}") from exc

        chunks: list[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                chunks.append(text)
        return "\n".join(chunks).strip()

    def _generate_description(self, file_path: Path) -> Optional[str]:
        """Generate a description for a file using LLM.

        Args:
            file_path: Path to the file (must be within library root)

        Returns:
            Generated description string, or None if generation fails
        """
        from pangu_agent.llm_client import Memory

        try:
            # Get relative path for read_file
            relative_path = file_path.relative_to(self._root)

            # Read file content using existing read_file method
            file_data = self.read_file(str(relative_path), include_content=True, include_meta=False)

            # Build LLM prompt based on file type
            memory = Memory()
            memory.add(
                "system",
                "You are a helpful assistant that generates concise descriptions for academic literature files. "
                "Provide a brief summary (2-3 sentences) highlighting the main topic, key findings, or content focus."
            )

            kind = file_data.get("kind")
            if kind == "text":
                # PDF content
                text = file_data.get("text", "")
                if not text:
                    logger.warning(f"Empty text content: {file_path}")
                    return None
                # Limit content length for LLM
                content_preview = text[:4000] if len(text) > 4000 else text
                memory.add("user", f"Please describe this PDF file:\n\n{content_preview}")
            elif kind == "image":
                # Image content (already in data URL format)
                image_url = file_data.get("image_url")
                if not image_url:
                    logger.warning(f"No image URL found: {file_path}")
                    return None
                memory.add_raw({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe this image briefly."},
                        {"type": "image_url", "image_url": image_url}
                    ]
                })
            else:
                logger.warning(f"Unsupported file kind: {kind}")
                return None

            # Call LLM
            response = self._llm_client.completion(memory)
            if response:
                return response.strip()
            else:
                logger.warning(f"LLM returned empty response for {file_path}")
                return None

        except Exception as exc:
            logger.exception(f"Error generating description for {file_path}: {exc}")
            return None
