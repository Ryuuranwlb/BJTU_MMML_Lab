import json
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy


MessageDict = Dict[str, Any]

class Memory:
    """
    General-purpose conversational memory for LLM agents.
    -----------------------------------------------------
    - Keeps a full `_history` of `Message` objects.
    - Exposes a context-limited `.history` property (using `keep_message_window()`).
    - Retains system prompts and last user input automatically.
    """

    def __init__(
        self,
        messages: Optional[List[MessageDict]] = None,
        context_window: int = 20,
    ):
        """
        Args:
            context_window: number of message pairs (user+assistant) to include
                            in active context. If <= 0, no limit is applied.
        """
        self.context_window = context_window
        self._history: List[MessageDict] = []

        if messages:
            for m in messages:
                # ensure role is present
                role = m.get("role", "user")
                msg = {"role": role}
                msg.update(m)
                self._history.append(msg)


    def add_raw(self, message: MessageDict):
        if not isinstance(message, dict):
            raise TypeError("message must be a dict")
        if "role" not in message:
            raise ValueError("message dict must contain a 'role' field")
        self._history.append(message)

    def add(self, role: str, content: Optional[str] = None, **kwargs: Any):
        msg: MessageDict = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)
        self._history.append(msg)


    def last(self, role: Optional[str] = None) -> Optional[MessageDict]:
        """
        Return the most recent message, optionally filtered by role.
        """
        if not self._history:
            return None

        if role is None:
            return self._history[-1]

        for m in reversed(self._history):
            if m.get("role") == role:
                return m
        return None

    # ============================================================
    # Context Handling
    # ============================================================
    def keep_message_window(
        self,
        messages: List[MessageDict],
    ) -> List[MessageDict]:
        """
        Return a context-trimmed view of messages.
        Keeps:
        - the first system message (if any)
        - the most recent N * 2 dialogue messages (user/assistant)
        - the last user message (if exists)
        """
        if not messages:
            return []

        has_system = messages[0].get("role") == "system"
        context_limit = 2 * self.context_window if self.context_window > 0 else 0

        last_is_user = messages[-1].get("role") == "user"
        last_message = messages[-1] if last_is_user else None

        start_index = 1 if has_system else 0

        if last_message:
            context_messages = messages[start_index:-1]
        else:
            context_messages = messages[start_index:]

        if context_limit > 0:
            context_messages = context_messages[-context_limit:]

        result: List[MessageDict] = []
        if has_system:
            result.append(messages[0])
        result.extend(context_messages)
        if last_message:
            result.append(last_message)
        return result

    @property
    def history(self) -> List[MessageDict]:
        history = self.keep_message_window(self._history)
        # deepcopy to prevent external mutation
        return [deepcopy(m) for m in history]

    # ============================================================
    # Persistence
    # ============================================================
    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of memory."""
        return {"history": deepcopy(self._history)}

    def load_snapshot(self, data: Dict[str, Any]):
        """Restore memory from snapshot data."""
        history = data.get("history", [])
        if not isinstance(history, list):
            raise TypeError("snapshot['history'] must be a list")
        self._history = [deepcopy(m) for m in history]

    def save_to_file(self, path: str):
        """Save full memory to disk."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)

    def load_from_file(self, path: str):
        """Load full memory from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_snapshot(data)

    # ============================================================
    # Maintenance
    # ============================================================
    def clear_memory(self):
        """Completely clear the stored conversation history."""
        self._history.clear()

    # ============================================================
    # Display / Debug
    # ============================================================
    def show(self, n: int = 10):
        """Print the latest messages (untrimmed)."""
        print("Memory Snapshot:")
        for m in self._history[-n:]:
            ts = m.get("timestamp", "-")
            role = m.get("role", "?")
            content = m.get("content", "")
            print(f"[{ts}] {role}: {content}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the entire memory as a serializable Python dict.
        Equivalent to `snapshot()` but with context_window included.
        """
        return {
            "context_window": self.context_window,
            "history": deepcopy(self._history),
        }