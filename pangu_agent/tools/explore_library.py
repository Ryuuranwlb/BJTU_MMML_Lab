"""Tool for exploring the library with a tree view."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pangu_agent.library.manager import LibraryManager
from pangu_agent.tools.base import Tool, ToolCallArguments, ToolParameter


class ExploreLibraryTool(Tool):
    _manager: LibraryManager
    _default_depth: int

    def __init__(self, manager: LibraryManager, default_depth: int = 2) -> None:
        super().__init__(
            name="explore_library",
            description=(
                "Explore the library from a relative path and return a tree view. "
                "Supports optional depth and whether to include file metadata."
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description=(
                        "Library-relative path to start from. "
                        "Example: 'nlp/transformers/attention'."
                    ),
                ),
                ToolParameter(
                    name="depth",
                    type="integer",
                    description=f"How many levels to expand (default: {default_depth}).",
                    required=False,
                ),
                ToolParameter(
                    name="include_meta",
                    type="boolean",
                    description="Whether to include file metadata (default: false).",
                    required=False,
                ),
            ],
        )
        self._manager = manager
        self._default_depth = default_depth

    def run(self, arguments: ToolCallArguments) -> str:
        raw_path = str(arguments.get("path", "")).strip()
        if not raw_path:
            raise ValueError("path is required")

        depth = arguments.get("depth", self._default_depth)
        include_meta = bool(arguments.get("include_meta", False))

        if isinstance(depth, str):
            if not depth.isdigit():
                raise ValueError("depth must be an integer")
            depth = int(depth)

        start = self._manager.get_entry(raw_path)
        if start.is_file():
            return self._format_file(start, include_meta)

        lines = list(self._walk_tree(start, int(depth), include_meta))
        return "\n".join(lines) if lines else "(empty)"

    def _walk_tree(
        self, path: Path, depth: int, include_meta: bool, indent: str = ""
    ) -> Iterable[str]:
        if depth < 0:
            return

        try:
            entries = self._manager.list_children(path)
        except OSError as exc:
            yield f"{indent}[error] {exc}"
            return

        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            meta = self._format_meta(entry, include_meta)
            yield f"{indent}- {entry.name}{suffix}{meta}"
            if entry.is_dir() and depth > 0:
                yield from self._walk_tree(entry, depth - 1, include_meta, indent + "  ")

    def _format_meta(self, path: Path, include_meta: bool) -> str:
        if not include_meta or path.is_dir():
            return ""
        try:
            return f" ({path.stat().st_size} bytes)"
        except OSError:
            return " (stat failed)"

    def _format_file(self, path: Path, include_meta: bool) -> str:
        suffix = self._format_meta(path, include_meta)
        return f"- {path.name}{suffix}"
