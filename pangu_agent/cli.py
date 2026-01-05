from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

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

console = Console()


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
@click.option("--prompt", help="Optional prompt for add, required query for search")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (shows agent iterations, tool calls, etc.)",
)
def run(
    library_root: str,
    action: str,
    path: str | None,
    prompt: str | None,
    verbose: bool,
):
    """Run a fixed task with LLM-powered organization."""

    # Configure logging based on verbose flag
    _configure_logging(verbose)

    action = action.lower()
    if action == "add":
        if not path:
            raise click.UsageError("--path is required for add.")

        with console.status("[bold blue]Initializing...", spinner="dots"):
            manager = LibraryManager(library_root)
            llm_client = LLMClient(log=verbose)
            tools = [
                ExploreLibraryTool(manager),
                ViewFileTool(manager),
                MoveFileTool(manager),
                SearchLibraryTool(manager),
            ]
            service = AddLiteratureService(manager, llm_client, tools)

        # Scan for files first
        console.print(f"[cyan]Scanning:[/cyan] {path}")
        files = service.scan_addable_files(path)

        if not files:
            console.print("[yellow]No addable files found (looking for PDFs and images)[/yellow]")
            return

        console.print(f"[green]Found {len(files)} file(s) to add[/green]\n")

        # Process files with progress bar
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Adding files...", total=len(files))

            for file_path in files:
                file_name = Path(file_path).name
                progress.update(task, description=f"[cyan]Processing: {file_name}")

                result = service.add_file_with_llm(str(file_path), user_prompt=prompt)
                results.append(result)

                progress.advance(task)

        # Print results
        _print_add_results(results)
        return

    if action == "search":
        if not prompt:
            raise click.UsageError("--prompt is required for search.")

        with console.status("[bold blue]Initializing...", spinner="dots"):
            manager = LibraryManager(library_root)
            llm_client = LLMClient(log=verbose)
            tools = [
                SearchLibraryTool(manager),
                ViewFileTool(manager),
                ExploreLibraryTool(manager),
                FinishTool(),
            ]
            service = SearchFilesService(manager, llm_client, tools)

        console.print(Panel(f"[bold]Query:[/bold] {prompt}", border_style="blue"))

        with console.status("[bold green]Searching library...", spinner="dots"):
            result = service.search(prompt)

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


def _configure_logging(verbose: bool):
    """Configure logging based on verbose flag."""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    else:
        # Disable all logging except errors
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            handlers=[RichHandler(console=console, show_time=False, show_path=False)],
        )


def _print_add_results(results: list[dict[str, Any]]):
    """Print results from add operation with rich formatting."""
    if not results:
        console.print("[yellow]No files were added.[/yellow]")
        return

    success_count = sum(1 for r in results if r.get("success"))
    total_count = len(results)

    # Create a table for results
    table = Table(title=f"\nAdded {success_count}/{total_count} file(s)", show_header=True, header_style="bold")
    table.add_column("Status", style="dim", width=8)
    table.add_column("Source", style="cyan")
    table.add_column("Destination", style="green")

    for result in results:
        source = Path(result.get("source", "unknown")).name
        if result.get("success"):
            dest = result.get("destination", "unknown")
            table.add_row("✓", source, dest)
        else:
            error = result.get("error", "unknown error")
            table.add_row("✗", source, f"[red]{error}[/red]")

    console.print(table)


def _print_search_results(result: dict[str, Any]):
    """Print results from search operation with rich formatting."""
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        console.print(Panel(f"[red]✗ Search failed:[/red] {error}", border_style="red"))
        return

    files = result.get("files", [])
    observation = result.get("observation", "")
    iterations = result.get("iterations", "N/A")

    # Success header
    console.print(f"\n[green]✓ Search completed in {iterations} iteration(s)[/green]\n")

    # Files table
    if files:
        table = Table(title=f"Found {len(files)} relevant file(s)", show_header=True, header_style="bold cyan")
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("File Path", style="green")

        for i, file_path in enumerate(files, 1):
            table.add_row(str(i), file_path)

        console.print(table)
    else:
        console.print("[yellow]No files found.[/yellow]")

    # Observation panel
    if observation:
        console.print("\n")
        console.print(Panel(observation, title="[bold]Observation[/bold]", border_style="blue"))


def _reset_library(library_root: str):
    """Reset the library by removing all contents."""
    # Confirm with user
    console.print(f"[yellow]WARNING:[/yellow] This will delete all contents in: [cyan]{library_root}[/cyan]")
    if not click.confirm("Are you sure you want to reset the library?"):
        console.print("[yellow]Reset cancelled.[/yellow]")
        return

    # Initialize manager and perform reset
    try:
        with console.status("[bold red]Resetting library...", spinner="dots"):
            manager = LibraryManager(library_root)
            result = manager.reset()

        if result["success"]:
            removed_count = result["removed_count"]
            message = result.get("message", "")

            if removed_count == 0:
                console.print(f"\n[yellow]{message}[/yellow]")
            else:
                console.print(f"\n[green]✓ Successfully reset library:[/green] {library_root}")
                console.print(f"  [dim]Removed {removed_count} item(s)[/dim]")
        else:
            error = result.get("error", "Unknown error")
            console.print(Panel(f"[red]✗ Failed to reset library:[/red] {error}", border_style="red"))
    except Exception as e:
        console.print(Panel(f"[red]✗ Failed to reset library:[/red] {e}", border_style="red"))



def main():
    pangu()


if __name__ == "__main__":
    main()
