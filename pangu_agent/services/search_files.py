"""Service for searching and retrieving relevant files from the library."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pangu_agent.agent import Agent
from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.prompts import (
    FILE_SEARCHER_SYSTEM_PROMPT,
    build_file_search_prompt,
)
from pangu_agent.tools import Tool
from pangu_agent.agent.utils import finish_tool_success_stop_condition

logger = logging.getLogger(__name__)


class SearchFilesService:
    """Orchestrates the process of searching files based on user query with LLM agent."""

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

    def search(self, query: str) -> Dict[str, Any]:
        """Search for relevant files based on user query.

        Args:
            query: User's search query describing what files they're looking for.

        Returns:
            Dict containing:
                - success: Whether the search completed successfully
                - files: List of relevant file paths
                - observation: Additional helpful information for the user
                - iterations: Number of agent iterations used
        """
        try:
            result = self._ask_llm_to_search(query)

            if not result:
                return {
                    "success": False,
                    "files": [],
                    "observation": "Failed to complete the search.",
                    "error": "Agent did not return a result",
                }

            return result

        except Exception as exc:
            logger.exception(f"Failed to search files: {exc}")
            return {
                "success": False,
                "files": [],
                "observation": f"An error occurred during search: {str(exc)}",
                "error": str(exc),
            }

    def _ask_llm_to_search(self, query: str) -> Dict[str, Any] | None:
        """Use LLM agent to search for relevant files and provide observations.

        The agent will use search_library, view_file, and explore_library tools
        to find relevant files, then use the finish tool to return results.
        """
        agent = Agent(
            llm_client=self._llm_client,
            tools=self._tools,
            max_iterations=10,
        )

        agent.add_system_prompt(FILE_SEARCHER_SYSTEM_PROMPT)
        agent.add_user_message(build_file_search_prompt(query))

        # Run agent until finish tool is successfully called
        result = agent.run(stop_condition=finish_tool_success_stop_condition)

        if not result["success"]:
            logger.warning(f"Agent failed to complete search: {result}")
            return None

        # Extract the finish tool output
        if result.get("result"):
            tool_output = result["result"]["output"]

            # The finish tool returns a dict with 'finished', 'status', 'result'
            # The 'result' field contains the JSON string with files and observation
            finish_result = tool_output.get("result", "{}")

            # Parse the JSON result
            import json
            try:
                parsed_result = json.loads(finish_result)
                return {
                    "success": True,
                    "files": parsed_result.get("files", []),
                    "observation": parsed_result.get("observation", ""),
                    "iterations": result["iterations"],
                }
            except json.JSONDecodeError:
                logger.error(f"Failed to parse finish tool result: {finish_result}")
                return {
                    "success": False,
                    "files": [],
                    "observation": "Failed to parse search results.",
                }

        return None
