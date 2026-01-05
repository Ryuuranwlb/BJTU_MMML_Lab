"""Tool for terminating execution and returning final results."""

from __future__ import annotations

from typing import Any

from pangu_agent.tools.base import Tool, ToolCallArguments, ToolParameter


class FinishTool(Tool):
    """Terminate execution and return the final result to the user."""

    def __init__(self) -> None:
        super().__init__(
            name="finish",
            description=(
                "Call this tool when you have completed the task and want to return "
                "the final result to the user. This will terminate the agent's execution. "
                "Use this only when you are confident you have fulfilled the user's request."
            ),
            parameters=[
                ToolParameter(
                    name="result",
                    type="string",
                    description=(
                        "The final result or answer to return to the user. "
                        "This should be a complete, well-formatted response that "
                        "addresses the user's original request."
                    ),
                ),
                ToolParameter(
                    name="status",
                    type="string",
                    description=(
                        "The completion status of the task. "
                        "Options: 'success' (task completed successfully), "
                        "'partial' (task partially completed), "
                        "'failed' (task could not be completed)."
                    ),
                    required=False,
                    enum=["success", "partial", "failed"],
                ),
            ],
        )

    def run(self, arguments: ToolCallArguments) -> Any:
        """Execute the finish operation and return the final result.

        Returns:
            A dictionary containing the result and metadata.

        Raises:
            ValueError: If required arguments are missing or invalid.
        """
        result = arguments.get("result")
        if result is None or (isinstance(result, str) and not result.strip()):
            raise ValueError("result is required and cannot be empty")

        status = arguments.get("status", "success")
        if status not in ["success", "partial", "failed"]:
            raise ValueError(
                f"Invalid status: {status}. Must be one of: success, partial, failed"
            )

        # Return a structured result that can be easily parsed by the agent
        return {
            "finished": True,
            "status": status,
            "result": str(result),
        }
