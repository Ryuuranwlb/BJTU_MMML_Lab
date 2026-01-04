"""Tool for reading image or PDF content from the library."""

from __future__ import annotations

import json

from pangu_agent.library.manager import LibraryManager
from pangu_agent.tools.base import Tool, ToolCallArguments, ToolParameter


class ViewFileTool(Tool):
    _manager: LibraryManager

    def __init__(self, manager: LibraryManager) -> None:
        super().__init__(
            name="view_file",
            description=(
                "Read a single image or PDF file from the library and return base64 content "
                "with metadata. Paths must be library-relative."
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description=(
                        "Library-relative file path. "
                        "Example: 'nlp/architecture/transformers/attention.pdf'."
                    ),
                ),
            ],
        )
        self._manager = manager

    def run(self, arguments: ToolCallArguments) -> dict[str, str]:
        raw_path = str(arguments.get("path", "")).strip()
        if not raw_path:
            raise ValueError("path is required")
        payload = self._manager.read_file(raw_path)
        return payload
