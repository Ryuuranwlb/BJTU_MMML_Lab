from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


@click.group()
def pangu():
    """Literature library agent (framework-only, no embeddings or external APIs)."""


DEFAULT_LIBRARY_ROOT = Path(__file__).resolve().parents[1] / "library"


@pangu.command()
@click.option(
    "--library",
    "library_root",
    default=str(DEFAULT_LIBRARY_ROOT),
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--action",
    required=True,
    type=click.Choice(["search", "add"], case_sensitive=False),
    help="Fixed task to execute in run mode.",
)
@click.option("--path", help="Path to add (for add).")
@click.option("--query", help="Query text")
def run(
    library_root: str,
    action: str,
    path: str | None,
    query: str | None,
):
    """Run a fixed task (no embeddings/LLM yet)."""

    action = action.lower()
    if action == "add":
        if not path:
            raise click.UsageError("--path is required for add.")
        # result = executor.execute(
        #     "add_literature", {"src_path": path}
        # )
        # _print_result(result)
        return

    if action == "search":
        if not query:
            raise click.UsageError("--query is required for search.")
        # result = executor.execute("search", {"query": query})
        # _print_result(result)
        return


@pangu.command()
@click.option(
    "--library",
    "library_root",
    default=str(DEFAULT_LIBRARY_ROOT),
    show_default=True,
    type=click.Path(file_okay=False),
)
def interactive(library_root: str):
    """Start a simple interactive session (LLM not wired yet)."""
    pass


def _print_result(result: Any):
    if result.success:
        click.echo(result.output or "OK")
    else:
        click.echo(f"[error] {result.error}")


def main():
    pangu()


if __name__ == "__main__":
    main()
