"""Prompts for literature organization agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

LITERATURE_ORGANIZER_SYSTEM_PROMPT = """You are a literature library organizer. Your task is to decide the appropriate \
location for a file in the library based on its content. \
You have access to tools to explore the library structure and move files. \
The file is currently in the inbox. Please analyze its content and move it to \
an appropriate location in the library with a meaningful directory structure."""


def build_file_organization_prompt(
    inbox_path: Path, file_content: Dict[str, Any]
) -> str:
    """Build user prompt for organizing a specific file.

    Args:
        inbox_path: Path to the file in inbox
        file_content: File content dict with 'kind', 'text', 'meta_data', etc.

    Returns:
        Formatted prompt string
    """
    parts = [f"File in inbox: {inbox_path}"]

    file_type = file_content.get("kind")
    if file_type == "text":
        text = file_content.get("text", "")
        preview = text[:2000] if len(text) > 2000 else text
        parts.append(f"\nFile type: PDF\nContent preview:\n{preview}")
    elif file_type == "image":
        parts.append("\nFile type: Image")

    metadata = file_content.get("meta_data", {})
    if metadata:
        parts.append(f"\nMetadata: {json.dumps(metadata, indent=2)}")

    parts.append(
        "\n\nPlease explore the library structure and decide where this file should be placed. "
        "Then use the move_file tool to move it from the inbox to the appropriate location."
    )

    return "\n".join(parts)
