"""Minimal synchronous tool template for AzureOpenAI function calling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

ToolCallArguments = Dict[str, Any]


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str | List[str]
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: ToolCallArguments
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    call_id: Optional[str] = None


class Tool(ABC):
    """Base class for synchronous tools."""

    _name: str
    _description: str
    _parameters: List[ToolParameter]

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> List[ToolParameter]:
        return list(self._parameters)

    @abstractmethod
    def run(self, arguments: ToolCallArguments) -> Any:
        """Execute the tool synchronously and return a result."""

    def schema(self) -> Dict[str, Any]:
        """OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": _build_schema(self._parameters),
            },
        }


class ToolExecutor:
    """Simple tool registry and executor."""

    _tools: Dict[str, Tool]

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

    @property
    def tools(self) -> List[Tool]:
        return list(self._tools.values())

    def schema(self) -> List[Dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall | str, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        if isinstance(tool_call, ToolCall):
            name = tool_call.name
            args = tool_call.arguments
            call_id = tool_call.call_id
        else:
            name = tool_call
            args = dict(arguments or {})
            call_id = None

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                name=name,
                success=False,
                error=f"Tool '{name}' not found. Available: {sorted(self._tools)}",
                call_id=call_id,
            )

        try:
            output = tool.run(args)
            return ToolResult(name=name, success=True, output=output, call_id=call_id)
        except Exception as exc:
            return ToolResult(name=name, success=False, error=str(exc), call_id=call_id)


def _build_schema(parameters: List[ToolParameter]) -> Dict[str, Any]:
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    for param in parameters:
        schema: Dict[str, Any] = {
            "type": param.type,
            "description": param.description,
        }
        if param.enum:
            schema["enum"] = param.enum
        if param.items:
            schema["items"] = param.items
        properties[param.name] = schema
        if param.required:
            required.append(param.name)

    payload: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        payload["required"] = required
    return payload
