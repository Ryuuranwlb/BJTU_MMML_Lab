"""Tool for semantic search in the library using embeddings."""

from __future__ import annotations

from pangu_agent.library.manager import LibraryManager
from pangu_agent.tools.base import Tool, ToolCallArguments, ToolParameter


class SearchLibraryTool(Tool):
    """Search library files using semantic similarity."""

    _manager: LibraryManager

    def __init__(self, manager: LibraryManager) -> None:
        super().__init__(
            name="search_library",
            description=(
                "Search for files in the library using semantic similarity. "
                "Provide a natural language query describing what you're looking for. "
                "Returns the most relevant files with their descriptions and similarity scores. "
                "Results are grouped by file type (PDF/image) with separate ranking for each type."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Natural language search query. "
                        "Examples: 'machine learning papers', 'attention mechanism', "
                        "'neural network architecture diagrams'."
                    ),
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Number of results to return per file type (default: 3).",
                    required=False,
                ),
                ToolParameter(
                    name="file_types",
                    type="array",
                    description=(
                        "File types to search. Options: 'pdf', 'image'. "
                        "If not specified, searches all types."
                    ),
                    required=False,
                    items={"type": "string", "enum": ["pdf", "image"]},
                ),
            ],
        )
        self._manager = manager

    def run(self, arguments: ToolCallArguments) -> str:
        """Execute semantic search and return formatted results."""
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")

        top_k = arguments.get("top_k", 3)  # Default to 3 per type
        if isinstance(top_k, str):
            if not top_k.isdigit():
                raise ValueError("top_k must be an integer")
            top_k = int(top_k)

        # Parse file_types
        file_types = arguments.get("file_types")
        if file_types is not None and not isinstance(file_types, list):
            file_types = [file_types]

        pdf_results = self._manager.search_library(query, top_k=int(top_k), file_types=["pdf"])
        image_results = self._manager.search_library(query, top_k=int(top_k), file_types=["image"])

        output_lines = [f"Found {len(pdf_results)} PDF(s) and {len(image_results)} image(s) matching\n"]

        # Format PDF results
        if pdf_results:
            output_lines.append("📄 PDF Documents:")
            for i, result in enumerate(pdf_results, 1):
                path = result["path"]
                score = result["score"]
                description = result.get("description", "No description available")

                output_lines.append(f"  {i}. {path}")
                output_lines.append(f"     Similarity: {score:.3f}")
                output_lines.append(f"     {description}")
                output_lines.append("")

        # Format image results
        if image_results:
            output_lines.append("🖼️  Images:")
            for i, result in enumerate(image_results, 1):
                path = result["path"]
                score = result["score"]
                description = result.get("description", "No description available")

                output_lines.append(f"  {i}. {path}")
                output_lines.append(f"     Similarity: {score:.3f}")
                output_lines.append(f"     {description}")
                output_lines.append("")

        return "\n".join(output_lines)
