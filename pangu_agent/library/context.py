"""Library path helpers and directory traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LibraryContext:
    root: Path

    @classmethod
    def from_root(cls, root_path: str) -> "LibraryContext":
        root = Path(root_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.expanduser().resolve()
        if not self._is_within_root(candidate):
            raise ValueError(f"Path escapes library root: {candidate}")
        return candidate

    def _is_within_root(self, candidate: Path) -> bool:
        try:
            candidate.relative_to(self.root)
            return True
        except ValueError:
            return False
