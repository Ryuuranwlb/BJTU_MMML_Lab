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
    _encoder: Any
    _vector_store: Any

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

        # Initialize embedding system
        logger.info("Initializing embedding system...")
        from pangu_agent.library.embeddings import OpenCLIPEncoder, VectorStore

        self._encoder = OpenCLIPEncoder()
        self._vector_store = VectorStore(
            storage_path=self._root / ".vector_store"
        )
        logger.info("Embedding system ready")

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
            "path": str(dest.relative_to(self._root)),
            "hash": _hash_file(dest),
            "added_at": _utc_now(),
            "updated_at": _utc_now(),
        }

        # Generate description via LLM
        # TODO: generate authors, title, year
        try:
            description = self._generate_description(dest)
            if description:
                metadata["description"] = description
                logger.info(f"Generated description for {dest.name}")
        except Exception as exc:
            logger.warning(f"Failed to generate description for {dest.name}: {exc}")

        self._write_metadata(dest, metadata)

        # Generate and store embedding
        try:
            relative_path = dest.relative_to(self._root)
            embedding = self._encoder.encode_file(dest)

            # Determine file type from extension
            file_type = self._detect_file_type(dest)
            self._vector_store.add(
                file_id=metadata["id"],
                embedding=embedding,
                metadata={
                    "path": str(relative_path),
                    "file_type": file_type
                }
            )
            logger.info(f"Generated embedding for {dest.name}")
        except Exception as exc:
            logger.warning(f"Failed to generate embedding for {dest.name}: {exc}")

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
            metadata["path"] = str(dest.relative_to(self._root))
            metadata["updated_at"] = _utc_now()
            self._write_metadata(dest, metadata)

            # Update vector store path
            try:
                file_id = metadata.get("id")
                if file_id:
                    relative_path = dest.relative_to(self._root)
                    self._vector_store.update_metadata(
                        file_id=file_id,
                        metadata={"path": str(relative_path)}
                    )
                    logger.debug(f"Updated vector store path for {file_id}")
            except Exception as exc:
                logger.warning(f"Failed to update vector store: {exc}")

        self._cleanup_empty_dirs(source.parent)
        return dest

    def read_metadata(self, file_path: str) -> Dict[str, Any]:
        path = self._context.resolve_path(file_path)
        meta_path = self._metadata_path(path)
        return self._read_metadata(meta_path)

    def list_children(self, path: Path | str) -> List[Path]:
        start = self._context.resolve_path(path)
        if not start.exists():
            raise FileNotFoundError(f"Path not found: {start}")
        if not start.is_dir():
            raise NotADirectoryError(f"Not a directory: {start}")
        
        def is_inside(start: Path, inbox: Path) -> bool:
            try:
                start.resolve().relative_to(inbox.resolve())
                return True
            except ValueError:
                return False
        if is_inside(start, self._inbox):
            raise ValueError(f"Cannot list children inside inbox: {start}")
        
        try:
            all_entries = start.iterdir()
            # Filter out metadata files (starting with .) and only keep PDFs, images, or directories
            filtered = [
                p for p in all_entries
                if not p.name.startswith('.') and (
                    p.is_dir() or 
                    p.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']
                )
                # if p.is_dir() or (
                #     not p.name.startswith('.') and
                #     p.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']
                # )
            ]
            return sorted(filtered, key=lambda p: (p.is_file(), p.name.lower()))
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
        metadata["path"] = str(path.relative_to(self._root))
        metadata["updated_at"] = _utc_now()
        self._write_metadata(path, metadata)
        return metadata

    def _metadata_path(self, file_path: Path) -> Path:
        return file_path.with_name(f".{file_path.name}.meta.json")

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
        """Generate a description for a file using LLM."""
        from pangu_agent.llm_client import Memory

        try:
            relative_path = file_path.relative_to(self._root)

            file_data = self.read_file(str(relative_path), include_content=True, include_meta=False)

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
                memory.add("user", f"Please describe this PDF file:\n\n{text}")
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

            response = self._llm_client.completion(memory)
            if response:
                return response.strip()
            else:
                logger.warning(f"LLM returned empty response for {file_path}")
                return None

        except Exception as exc:
            logger.exception(f"Error generating description for {file_path}: {exc}")
            return None

    def search_library(
        self, query: str, top_k: int = 5, file_types: List[str] = ["pdf", "image"]
    ) -> List[Dict[str, Any]]:
        """Search library files by semantic similarity.

        Args:
            query: Natural language search query
            top_k: Total number of results to return across all file types
            file_types: List of file types to search (e.g., ["pdf", "image"])

        Returns:
            List of search results, each containing path, score, description, metadata, and file_type.
            Results are sorted by similarity score in descending order.
        """
        query_embedding = self._encoder.encode_text(query)

        # Use ChromaDB's where clause to filter by file_type
        where_filter = {"file_type": {"$in": file_types}}

        # Search with filter
        raw_results = self._vector_store.search(
            query_embedding,
            top_k=top_k,
            where=where_filter
        )

        # Build final results with metadata
        results = []
        for result in raw_results:
            path = result.metadata["path"]
            file_type = result.metadata.get("file_type")
            file_path = self._context.resolve_path(path)

            # Try to read metadata from .meta.json
            try:
                meta_path = self._metadata_path(file_path)
                file_metadata = self._read_metadata(meta_path)
            except Exception:
                file_metadata = {}

            results.append({
                "path": path,
                "score": result.score,
                "description": file_metadata.get("description", ""),
                "metadata": file_metadata,
                "file_type": file_type,
            })

        return results

    def reset(self) -> Dict[str, Any]:
        """Reset the library by removing all files and metadata.

        Returns:
            Dict with success status and count of removed items
        """
        if not self._root.exists():
            return {
                "success": False,
                "error": "Library directory does not exist",
                "removed_count": 0
            }

        # Count items before deletion
        total_items = 0
        for item in self._root.iterdir():
            total_items += 1

        if total_items == 0:
            return {
                "success": True,
                "removed_count": 0,
                "message": "Library is already empty"
            }

        # Delete all contents
        try:
            for item in self._root.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            # Reinitialize inbox and vector store
            self._inbox.mkdir(parents=True, exist_ok=True)
            self._vector_store = None

            # Reinitialize vector store
            from pangu_agent.library.embeddings import VectorStore
            self._vector_store = VectorStore(
                storage_path=self._root / ".vector_store"
            )

            logger.info(f"Library reset complete: removed {total_items} items")
            return {
                "success": True,
                "removed_count": total_items,
                "message": f"Successfully reset library"
            }
        except Exception as exc:
            logger.error(f"Failed to reset library: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "removed_count": 0
            }
