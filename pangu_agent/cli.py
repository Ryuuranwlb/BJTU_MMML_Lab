from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.services.add_literature import AddLiteratureService
from pangu_agent.tools.explore_library import ExploreLibraryTool
from pangu_agent.tools.move_file import MoveFileTool
from pangu_agent.tools.view_file import ViewFileTool


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

        # Initialize components
        manager = LibraryManager(library_root)
        llm_client = LLMClient()

        # Setup tools
        tools = [
            ExploreLibraryTool(manager),
            ViewFileTool(manager),
            MoveFileTool(manager),
        ]

        # Create service and execute add
        service = AddLiteratureService(manager, llm_client, tools)
        results = service.add_path(path)

        # Print results
        _print_add_results(results)
        return

    if action == "search":
        if not query:
            raise click.UsageError("--query is required for search.")
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


def _print_add_results(results: list[dict[str, Any]]):
    """Print results from add operation."""
    if not results:
        click.echo("No files were added.")
        return

    success_count = sum(1 for r in results if r.get("success"))
    total_count = len(results)

    click.echo(f"\nAdded {success_count}/{total_count} file(s):\n")

    for result in results:
        source = result.get("source", "unknown")
        if result.get("success"):
            dest = result.get("destination", "unknown")
            click.echo(f"  ✓ {source} -> {dest}")
        else:
            error = result.get("error", "unknown error")
            click.echo(f"  ✗ {source}: {error}")



def main():
    pangu()


if __name__ == "__main__":
    main()
