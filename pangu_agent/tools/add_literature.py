"""Tool for adding literature files to the library."""

from __future__ import annotations

from typing import Any, Dict

from pangu_agent.tools.base import Tool, ToolCallArguments, ToolParameter


class AddLiteratureTool(Tool):
    """Add literature files to the library with LLM-guided organization."""
    
    _service: Any

    def __init__(self, service) -> None:
        super().__init__(
            name="add_literature",
            description=(
                "Add one or more literature files (PDFs or images) to the library. "
                "This tool will launch a separate agent process for each file to analyze content "
                "and automatically organize it into the appropriate library location. "
                "Can accept a single file path or a directory path to add multiple files recursively."
            ),
            parameters=[
                ToolParameter(
                    name="source_path",
                    type="string",
                    description=(
                        "Path to the file or directory to add. "
                        "For a single file: '/path/to/paper.pdf'. "
                        "For a directory: '/path/to/papers/' (all PDFs and images will be added recursively)."
                    ),
                ),
                ToolParameter(
                    name="additional_context",
                    type="string",
                    description=(
                        "Optional context or instructions for the organization agent. "
                        "Use this to provide hints about the content, desired categorization, "
                        "or any special handling needed. "
                    ),
                    required=False,
                ),
            ],
        )
        self._service = service

    def run(self, arguments: ToolCallArguments) -> Dict[str, Any]:
        """Execute the add literature operation and return structured results."""
        source_path = str(arguments.get("source_path", "")).strip()
        if not source_path:
            raise ValueError("source_path is required")

        additional_context = arguments.get("additional_context")
        if additional_context is not None:
            additional_context = str(additional_context).strip() or None

        # Call the service to add files
        results = self._service.add_path(source_path, additional_context)

        if not results:
            return {
                "success": False,
                "total_files": 0,
                "succeeded": 0,
                "failed": 0,
                "message": f"No addable files found in: {source_path}",
                "files": [],
            }

        # Aggregate results
        success_count = sum(1 for r in results if r["success"])
        failure_count = len(results) - success_count

        return {
            "success": success_count > 0,
            "total_files": len(results),
            "succeeded": success_count,
            "failed": failure_count,
            "files": results,
        }
