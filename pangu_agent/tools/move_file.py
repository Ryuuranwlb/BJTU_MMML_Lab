"""Tool for moving a file inside the library."""

from __future__ import annotations

from pangu_agent.library.manager import LibraryManager
from pangu_agent.tools.base import Tool, ToolCallArguments, ToolParameter


class MoveFileTool(Tool):
    _manager: LibraryManager

    def __init__(self, manager: LibraryManager) -> None:
        super().__init__(
            name="move_file",
            description=(
                "Move a file within the library to a new path. Paths must be library-relative. "
                f"Source may be inside the hidden inbox (e.g. '{manager.inbox.name}'), but the "
                "destination must not be inside the inbox."
            ),
            parameters=[
                ToolParameter(
                    name="source_path",
                    type="string",
                    description=(
                        "Library-relative path of the file to move. "
                        "Example: 'nlp/surveys/transformers/attention.pdf'."
                    ),
                ),
                ToolParameter(
                    name="dest_path",
                    type="string",
                    description=(
                        "Library-relative destination path. "
                        "Example: 'nlp/architecture/transformers/attention.pdf'. "
                        f"Destination must not be inside '{manager.inbox.name}'."
                    ),
                ),
            ],
        )
        self._manager = manager

    def run(self, arguments: ToolCallArguments) -> str:
        source_path = str(arguments.get("source_path", "")).strip()
        dest_path = str(arguments.get("dest_path", "")).strip()
        if not source_path or not dest_path:
            raise ValueError("source_path and dest_path are required")
        moved = self._manager.move_file(source_path, dest_path)
        return f"moved {source_path} -> {moved}"
