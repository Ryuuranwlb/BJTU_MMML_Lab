"""Service for adding literature files to the library with LLM-guided organization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.llm_client.memory import Memory
from pangu_agent.tools.base import ToolExecutor, ToolCall

logger = logging.getLogger(__name__)


class AddLiteratureService:
    """Orchestrates the process of adding files to the library with LLM organization."""

    _manager: LibraryManager
    _llm_client: LLMClient
    _tool_executor: ToolExecutor

    def __init__(
        self,
        manager: LibraryManager,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
    ) -> None:
        self._manager = manager
        self._llm_client = llm_client
        self._tool_executor = tool_executor

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

    def add_file_with_llm(self, file_path: str) -> Dict[str, Any]:
        """Add a single file: stage to inbox, let LLM decide location, then move."""
        try:
            # Stage file to inbox
            staged_path = self._manager.stage_copy(file_path)
            inbox_relative = staged_path.relative_to(self._manager.root)
            logger.info(f"Staged file to: {inbox_relative}")

            # Read file content for LLM context
            file_content = self._manager.read_file(str(inbox_relative))

            # Ask LLM to move the file (LLM will call move_file tool directly)
            moved_info = self._ask_llm_to_move_file(inbox_relative, file_content)

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

    def add_path(self, source_path: str) -> List[Dict[str, Any]]:
        """Add a file or recursively add all files in a directory."""
        files = self.scan_addable_files(source_path)

        if not files:
            logger.warning(f"No addable files found in: {source_path}")
            return []

        logger.info(f"Found {len(files)} file(s) to add")

        results: List[Dict[str, Any]] = []
        for file_path in files:
            result = self.add_file_with_llm(str(file_path))
            results.append(result)

        return results

    def _is_addable(self, path: Path) -> bool:
        """Check if a file is addable (PDF or image)."""
        suffix = path.suffix.lower()
        return suffix in [".pdf", ".jpg", ".jpeg", ".png"]

    def _ask_llm_to_move_file(
        self, inbox_path: Path, file_content: Dict[str, Any]
    ) -> Dict[str, str] | None:
        """Use LLM to move the file from inbox to appropriate location."""
        memory = Memory()

        # System prompt
        memory.add(
            "system",
            "You are a literature library organizer. Your task is to decide the appropriate "
            "location for a file in the library based on its content. "
            "You have access to tools to explore the library structure and move files. "
            "The file is currently in the inbox. Please analyze its content and move it to "
            "an appropriate location in the library with a meaningful directory structure.",
        )

        # Construct user message with file info
        user_content = self._build_file_context(inbox_path, file_content)
        memory.add("user", user_content)

        # Get tool schemas
        tool_schemas = self._tool_executor.schema()

        # Call LLM with tools
        max_iterations = 5
        for iteration in range(max_iterations):
            response = self._llm_client.completion(
                memory, tools=tool_schemas, raw=True
            )

            if response is None:
                logger.error("LLM returned no response")
                return None

            message = response.choices[0].message
            memory.add_raw(message.model_dump(exclude_unset=True))

            logger.info(f"LLM response (iteration {iteration + 1}): tool_calls={bool(message.tool_calls)}, content={message.content[:100] if message.content else None}")

            # Check if LLM wants to call tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"LLM calling tool: {tool_name} with args: {tool_args}")

                    # Execute tool
                    tc = ToolCall(
                        name=tool_name, arguments=tool_args, call_id=tool_call.id
                    )
                    result = self._tool_executor.execute(tc)

                    # Add tool result to memory
                    memory.add(
                        "tool",
                        content=str(result.output) if result.success else result.error,
                        tool_call_id=tool_call.id,
                    )

                    # If move_file was successful, return the destination
                    if tool_name == "move_file" and result.success:
                        dest_path = tool_args.get("dest_path")
                        return {"destination": dest_path}

            elif message.content:
                # LLM finished without tool calls
                logger.warning(f"LLM finished without moving file: {message.content}")
                return None

        logger.error("Max iterations reached without moving file")
        return None

    def _build_file_context(
        self, inbox_path: Path, file_content: Dict[str, Any]
    ) -> str:
        """Build context message about the file for the LLM."""
        parts: List[str] = []
        parts.append(f"File in inbox: {inbox_path}")

        file_type = file_content.get("kind")
        if file_type == "text":
            text = file_content.get("text", "")
            preview = text[:2000] if len(text) > 2000 else text
            parts.append(f"\nFile type: PDF\nContent preview:\n{preview}")
        elif file_type == "image":
            parts.append(f"\nFile type: Image")

        metadata = file_content.get("meta_data", {})
        if metadata:
            parts.append(f"\nMetadata: {json.dumps(metadata, indent=2)}")

        parts.append(
            "\n\nPlease explore the library structure and decide where this file should be placed. "
            "Then use the move_file tool to move it from the inbox to the appropriate location."
        )

        return "\n".join(parts)
