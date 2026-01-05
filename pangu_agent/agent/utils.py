"""Utility functions for agents."""

from __future__ import annotations

from pangu_agent.tools.base import ToolCall, ToolResult


def finish_tool_stop_condition(tool_call: ToolCall, tool_result: ToolResult) -> bool:
    """Default stop condition that checks if the finish tool was called."""
    return tool_call.name == "finish"


def finish_tool_success_stop_condition(tool_call: ToolCall, tool_result: ToolResult) -> bool:
    """Stop condition that checks if the finish tool was called successfully."""
    return tool_call.name == "finish" and tool_result.success
