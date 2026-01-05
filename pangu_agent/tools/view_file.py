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
                "Read a single image or PDF file from the library. "
                "Paths must be library-relative."
            ),
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description=(
                        "Library-relative file path. "
                        "Example: 'nlp/architecture/transformers/attention.pdf'."
                    ),
                ),
                ToolParameter(
                    name="info_type",
                    type="string",
                    description=(
                        "Type of information to return: "
                        "'content' (file content only), "
                        "'overview' (metadata and description only), "
                        "'both' (content, metadata, and description)."
                    ),
                    required=False,
                    enum=["content", "overview", "both"],
                ),
            ],
        )
        self._manager = manager

    def run(self, arguments: ToolCallArguments) -> dict[str, str]:
        raw_path = str(arguments.get("file_path", "")).strip()
        if not raw_path:
            raise ValueError("file_path is required")

        info_type = str(arguments.get("info_type", "both")).strip().lower()
        if info_type not in ["content", "overview", "both"]:
            raise ValueError(
                f"info_type must be 'content', 'overview', or 'both', got '{info_type}'"
            )

        # Map info_type to manager parameters
        if info_type == "overview":
            payload = self._manager.read_file(raw_path, include_content=False, include_meta=True)
        elif info_type == "content":
            payload = self._manager.read_file(raw_path, include_content=True, include_meta=False)
        else:
            payload = self._manager.read_file(raw_path, include_content=True, include_meta=True)

        return payload
