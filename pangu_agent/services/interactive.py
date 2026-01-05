"""Service for interactive chat session with the literature library assistant."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pangu_agent.agent import Agent
from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.prompts import INTERACTIVE_ASSISTANT_SYSTEM_PROMPT
from pangu_agent.tools import Tool

logger = logging.getLogger(__name__)


class InteractiveService:
    """Manages an interactive chat session with the LLM-powered library assistant.

    This service provides a conversational interface where users can interact with
    the agent to perform various library management tasks.
    """

    _manager: LibraryManager
    _llm_client: LLMClient
    _agent: Agent
    _max_iterations: int

    def __init__(
        self,
        manager: LibraryManager,
        llm_client: LLMClient,
        tools: List[Tool],
        max_iterations: int = 15,
    ) -> None:
        """Initialize the interactive service.

        Args:
            manager: Library manager instance
            llm_client: LLM client for completions
            tools: List of tools available to the agent
            max_iterations: Maximum iterations per user message (default: 15)
        """
        self._manager = manager
        self._llm_client = llm_client
        self._max_iterations = max_iterations

        # Create a persistent agent for the session
        self._agent = Agent(
            llm_client=self._llm_client,
            tools=tools,
            max_iterations=max_iterations,
        )

        # Add system prompt once at initialization
        self._agent.add_system_prompt(INTERACTIVE_ASSISTANT_SYSTEM_PROMPT)

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process a user message and return the agent's response.

        Args:
            user_message: The user's input message

        Returns:
            Dict containing:
                - success: Whether the interaction completed successfully
                - response: The agent's response text
                - iterations: Number of iterations used
                - stop_reason: Why the agent stopped
        """
        try:
            # Add user message to agent's memory
            self._agent.add_user_message(user_message)

            # Run the agent without a stop condition (let it finish naturally)
            result = self._agent.run(stop_condition=None)

            if not result["success"]:
                error_msg = f"Agent failed: {result.get('stop_reason', 'unknown')}"
                logger.warning(error_msg)
                return {
                    "success": False,
                    "response": "I encountered an issue processing your request. Please try again.",
                    "iterations": result.get("iterations", 0),
                    "stop_reason": result.get("stop_reason"),
                }

            # Extract the final message from the agent
            response_text = result.get("final_message", "I've completed your request.")

            return {
                "success": True,
                "response": response_text,
                "iterations": result["iterations"],
                "stop_reason": result["stop_reason"],
            }

        except Exception as exc:
            logger.exception(f"Failed to process message: {exc}")
            return {
                "success": False,
                "response": f"An error occurred: {str(exc)}",
                "iterations": 0,
                "stop_reason": "exception",
            }

    def reset(self) -> None:
        """Reset the conversation history while keeping the system prompt."""
        self._agent.clear_memory()
        self._agent.add_system_prompt(INTERACTIVE_ASSISTANT_SYSTEM_PROMPT)
        logger.info("Interactive session reset")
