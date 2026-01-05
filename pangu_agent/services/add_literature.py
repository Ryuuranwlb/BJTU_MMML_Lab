"""Service for adding literature files to the library with LLM-guided organization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from pangu_agent.agent import Agent
from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.prompts import (
    LITERATURE_ORGANIZER_SYSTEM_PROMPT,
    build_file_organization_prompt,
)
from pangu_agent.tools import Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class AddLiteratureService:
    """Orchestrates the process of adding files to the library with LLM organization."""

    _manager: LibraryManager
    _llm_client: LLMClient
    _tools: List[Tool]

    def __init__(
        self,
        manager: LibraryManager,
        llm_client: LLMClient,
        tools: List[Tool],
    ) -> None:
        self._manager = manager
        self._llm_client = llm_client
        self._tools = tools

    def scan_addable_files(self, source_path: str) -> List[Path]:
        """Recursively scan for addable files (PDFs and images)."""
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source}")

        addable_files: List[Path] = []

        if source.is_file():
            if self._is_addable(source):
                addable_files.append(source)
        else:
            for item in source.rglob("*"):
                if item.is_file() and self._is_addable(item):
                    addable_files.append(item)

        return sorted(addable_files)

    def add_file_with_llm(
        self, file_path: str, user_prompt: str | None = None
    ) -> Dict[str, Any]:
        """Add a single file: stage to inbox, let LLM decide location, then move."""
        try:
            # Stage file to inbox
            staged_path = self._manager.stage_copy(file_path)
            inbox_relative = staged_path.relative_to(self._manager.root)
            logger.debug(f"Staged file to: {inbox_relative}")

            # Read file content for LLM context
            file_content = self._manager.read_file(str(inbox_relative))

            # Ask LLM to move the file (LLM will call move_file tool directly)
            moved_info = self._ask_llm_to_move_file(
                inbox_relative, file_content, user_prompt
            )

            if not moved_info:
                return {
                    "success": False,
                    "source": file_path,
                    "staged": str(inbox_relative),
                    "error": "LLM failed to move file",
                }

            return {
                "success": True,
                "source": file_path,
                "destination": moved_info["destination"],
            }

        except Exception as exc:
            logger.exception(f"Failed to add file {file_path}: {exc}")
            return {
                "success": False,
                "source": file_path,
                "error": str(exc),
            }

    def add_path(
        self, source_path: str, user_prompt: str | None = None
    ) -> List[Dict[str, Any]]:
        """Add a file or recursively add all files in a directory."""
        files = self.scan_addable_files(source_path)

        if not files:
            logger.warning(f"No addable files found in: {source_path}")
            return []

        logger.info(f"PangGu🍄 Found {len(files)} file(s) to add")

        results: List[Dict[str, Any]] = []
        for file_path in files:
            result = self.add_file_with_llm(str(file_path), user_prompt)
            results.append(result)

        return results

    def _is_addable(self, path: Path) -> bool:
        """Check if a file is addable (PDF or image)."""
        suffix = path.suffix.lower()
        return suffix in [".pdf", ".jpg", ".jpeg", ".png"]

    def _ask_llm_to_move_file(
        self,
        inbox_path: Path,
        file_content: Dict[str, Any],
        user_prompt: str | None = None,
    ) -> Dict[str, str] | None:
        """Use LLM agent to move the file from inbox to appropriate location."""
        agent = Agent(
            llm_client=self._llm_client,
            tools=self._tools,
            max_iterations=5,
        )

        agent.add_system_prompt(LITERATURE_ORGANIZER_SYSTEM_PROMPT)
        agent.add_user_message(
            build_file_organization_prompt(inbox_path, file_content, user_prompt)
        )

        def stop_when_file_moved(tool_call: ToolCall, tool_result: ToolResult) -> bool:
            """Stop when move_file tool is successfully executed."""
            return tool_call.name == "move_file" and tool_result.success

        result = agent.run(stop_condition=stop_when_file_moved)

        if result["success"] and result.get("result"):
            dest_path = result["result"]["tool_args"].get("dest_path")
            return {"destination": dest_path}

        logger.warning(f"Agent failed to move file: {result}")
        return None
