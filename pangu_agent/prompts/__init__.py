"""Prompt templates for various agents."""

from pangu_agent.prompts.literature_organizer import (
    LITERATURE_ORGANIZER_SYSTEM_PROMPT,
    build_file_organization_prompt,
)
from pangu_agent.prompts.file_searcher import (
    FILE_SEARCHER_SYSTEM_PROMPT,
    build_file_search_prompt,
)

__all__ = [
    "LITERATURE_ORGANIZER_SYSTEM_PROMPT",
    "build_file_organization_prompt",
    "FILE_SEARCHER_SYSTEM_PROMPT",
    "build_file_search_prompt",
]
