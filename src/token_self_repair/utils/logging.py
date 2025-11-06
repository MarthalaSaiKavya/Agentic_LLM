"""Lightweight logging and tracing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from rich.console import Console


@dataclass(slots=True)
class Logger:
    """Structured logger that coordinates with messaging layer."""

    name: str
    console: Console = Console(width=100, soft_wrap=True)

    def info(self, message: str, *, status: Optional[str] = None) -> None:
        prefix = f"[{self.name}]"
        if status:
            prefix = f"{prefix}[{status}]"
        self.console.print(f"{prefix} {message}")

    def table(self, headers: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
        from rich.table import Table

        table = Table(*headers, show_header=True, box=None, pad_edge=False)
        for row in rows:
            table.add_row(*row)
        self.console.print(table)
