"""Prompts for literature organization agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

LITERATURE_ORGANIZER_SYSTEM_PROMPT = """
You are a literature library organizer. Your task is to decide the appropriate location for a file in the library based on its content.
You have access to tools to explore the library structure and move files.

IMPORTANT: The file is currently in the inbox (.inbox directory).
DO NOT try to explore the inbox, it is a temporary staging area that is not visible to your ExploreLibraryTool.
But you can view the file content with the ViewFileTool if needed.

You will be provided with the file's metadata in the user message. Focus on:
1. Analyzing the provided file content and metadata
2. Exploring the existing library structure (NOT the inbox)
3. Deciding an appropriate destination path
4. Moving the file from inbox to the chosen location using the move_file tool

The Agent's execution will end when you have successfully moved the file to its final location.
Create a meaningful directory structure based on the file's research topic."""


def build_file_organization_prompt(
    inbox_path: Path, file_content: Dict[str, Any], user_prompt: str | None = None
) -> str:
    """Build user prompt for organizing a specific file.

    Args:
        inbox_path: Path to the file in inbox
        file_content: File content dict with 'kind', 'text', 'meta_data', etc.
        user_prompt: Optional user-provided prompt for additional context

    Returns:
        Formatted prompt string
    """
    parts = [f"File to organize: {inbox_path}"]

    file_type = file_content.get("kind")
    if file_type == "text":
        text = file_content.get("text", "")
        parts.append(f"\nFile type: PDF\n")
    elif file_type == "image":
        parts.append("\nFile type: Image")

    metadata = file_content.get("meta_data", {})
    if metadata:
        parts.append(f"\nMetadata: {json.dumps(metadata, indent=2)}")

    if user_prompt:
        parts.append(f"\n\nUser request: {user_prompt}")

    parts.append(
        "\n\nPlease explore the library structure and decide where this file should be placed. "
        "Then use the move_file tool to move it from the inbox to the appropriate location."
    )

    return "\n".join(parts)
