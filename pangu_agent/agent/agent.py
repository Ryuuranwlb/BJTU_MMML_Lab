"""Generic agent with tool-calling loop."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from pangu_agent.llm_client.client import LLMClient
from pangu_agent.llm_client.memory import Memory
from pangu_agent.tools.base import Tool, ToolCall, ToolExecutor

logger = logging.getLogger(__name__)


class Agent:
    """Generic agent that executes tool-calling loops.

    The agent maintains its own memory and can run multiple iterations
    of LLM calls with tool execution until completion.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: List[Tool],
        max_iterations: int = 10,
    ) -> None:
        self._llm_client = llm_client
        self._tool_executor = ToolExecutor(tools)
        self._max_iterations = max_iterations
        self._memory = Memory()

    def add_system_prompt(self, prompt: str) -> None:
        self._memory.add("system", prompt)

    def add_user_message(self, message: str) -> None:
        self._memory.add("user", message)

    def run(
        self,
        stop_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Run agent tool-calling loop until completion.

        Args:
            stop_condition: Optional function to check if agent should stop.
                           Receives dict with 'tool_name', 'tool_args', 'result'.
                           Returns True to stop, False to continue.

        Returns:
            Dict with success, iterations, stop_reason, and optional result/final_message.
        """
        tool_schemas = self._tool_executor.schema()

        for iteration in range(self._max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{self._max_iterations}")

            response = self._llm_client.completion(
                self._memory, tools=tool_schemas, raw=True
            )

            if response is None:
                logger.error("LLM returned no response")
                return {
                    "success": False,
                    "iterations": iteration + 1,
                    "stop_reason": "llm_no_response",
                }

            message = response.choices[0].message
            self._memory.add_raw(message.model_dump(exclude_unset=True))

            logger.info(
                f"LLM response: tool_calls={bool(message.tool_calls)}, "
                f"content={'...' if message.content else None}"
            )

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Executing tool: {tool_name}({tool_args})")

                    tc = ToolCall(
                        name=tool_name, arguments=tool_args, call_id=tool_call.id
                    )
                    result = self._tool_executor.execute(tc)
                    logging.info(f"Tool result: {result.output or result.error}")

                    self._memory.add(
                        "tool",
                        content=str(result.output) if result.success else result.error,
                        tool_call_id=tool_call.id,
                    )

                    if stop_condition:
                        should_stop = stop_condition(
                            {
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "result": result,
                            }
                        )
                        if should_stop:
                            return {
                                "success": result.success,
                                "iterations": iteration + 1,
                                "stop_reason": "stop_condition_met",
                                "result": {
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "output": result.output,
                                },
                            }

            elif message.content:
                logger.info(f"Agent finished: {message.content}")
                return {
                    "success": True,
                    "iterations": iteration + 1,
                    "stop_reason": "llm_finished",
                    "final_message": message.content,
                }

        logger.warning(f"Max iterations ({self._max_iterations}) reached")
        return {
            "success": False,
            "iterations": self._max_iterations,
            "stop_reason": "max_iterations",
        }

    def clear_memory(self) -> None:
        self._memory = Memory()

    @property
    def memory(self) -> Memory:
        return self._memory
