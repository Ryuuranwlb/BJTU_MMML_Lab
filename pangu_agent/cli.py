from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from pangu_agent.library.manager import LibraryManager
from pangu_agent.llm_client.client import LLMClient
from pangu_agent.services.add_literature import AddLiteratureService
from pangu_agent.services.search_files import SearchFilesService
from pangu_agent.tools import (
    ExploreLibraryTool,
    FinishTool,
    MoveFileTool,
    SearchLibraryTool,
    ViewFileTool,
)


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
    type=click.Choice(["search", "add", "reset"], case_sensitive=False),
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
            SearchLibraryTool(manager),
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

        # Initialize components
        manager = LibraryManager(library_root)
        llm_client = LLMClient()

        # Setup tools (no MoveFileTool for search)
        tools = [
            SearchLibraryTool(manager),
            ViewFileTool(manager),
            ExploreLibraryTool(manager),
            FinishTool(),
        ]

        # Create service and execute search
        service = SearchFilesService(manager, llm_client, tools)
        result = service.search(query)

        # Print results
        _print_search_results(result)
        return

    if action == "reset":
        # Reset the library by removing all files and metadata
        _reset_library(library_root)
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


def _print_search_results(result: dict[str, Any]):
    """Print results from search operation."""
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        click.echo(f"\n✗ Search failed: {error}")
        return

    files = result.get("files", [])
    observation = result.get("observation", "")
    iterations = result.get("iterations", "N/A")

    click.echo(f"\n✓ Search completed in {iterations} iteration(s)\n")

    if files:
        click.echo(f"Found {len(files)} relevant file(s):\n")
        for i, file_path in enumerate(files, 1):
            click.echo(f"  {i}. {file_path}")
    else:
        click.echo("No files found.")

    if observation:
        click.echo(f"\nObservation:\n  {observation}")


def _reset_library(library_root: str):
    """Reset the library by removing all contents."""
    # Confirm with user
    click.echo(f"WARNING: This will delete all contents in: {library_root}")
    if not click.confirm("Are you sure you want to reset the library?"):
        click.echo("Reset cancelled.")
        return

    # Initialize manager and perform reset
    try:
        manager = LibraryManager(library_root)
        result = manager.reset()

        if result["success"]:
            removed_count = result["removed_count"]
            message = result.get("message", "")

            if removed_count == 0:
                click.echo(f"\n{message}")
            else:
                click.echo(f"\n✓ Successfully reset library: {library_root}")
                click.echo(f"  Removed {removed_count} item(s)")
        else:
            error = result.get("error", "Unknown error")
            click.echo(f"\n✗ Failed to reset library: {error}")
    except Exception as e:
        click.echo(f"\n✗ Failed to reset library: {e}")



def main():
    pangu()


if __name__ == "__main__":
    main()
