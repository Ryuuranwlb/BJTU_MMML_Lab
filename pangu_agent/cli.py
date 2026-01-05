from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logging.getLogger().setLevel(logging.ERROR)

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
from pangu_agent.services.interactive import InteractiveService
from pangu_agent.tools import (
    AddLiteratureTool,
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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (shows agent iterations, tool calls, etc.)",
)
def interactive(library_root: str, verbose: bool):
    """Start an interactive chat session with PangGu🍄 assistant."""
    # Configure logging
    _configure_logging(verbose)

    # Initialize components
    console.print("[bold cyan]Initializing PangGu🍄 Interactive Assistant...[/bold cyan]")

    with console.status("[bold blue]Loading...", spinner="dots"):
        manager = LibraryManager(library_root)
        llm_client = LLMClient(log=verbose)

        # Create AddLiteratureService for AddLiteratureTool
        add_lit_service = AddLiteratureService(
            manager=manager,
            llm_client=llm_client,
            tools=[
                ExploreLibraryTool(manager),
                ViewFileTool(manager),
                MoveFileTool(manager),
                SearchLibraryTool(manager),
            ]
        )

        # Create all available tools for the interactive agent
        tools = [
            SearchLibraryTool(manager),
            ExploreLibraryTool(manager),
            ViewFileTool(manager),
            AddLiteratureTool(add_lit_service),
            MoveFileTool(manager),
        ]

        # Create interactive service
        service = InteractiveService(
            manager=manager,
            llm_client=llm_client,
            tools=tools,
            max_iterations=15,
        )

    # Display welcome message
    console.print()
    console.print(Panel.fit(
        "[bold green]Welcome to PangGu🍄 Interactive Assistant![/bold green]\n\n"
        f"[dim]Library: {library_root}[/dim]\n\n"
        "I can help you:\n"
        "  • Search for papers and files\n"
        "  • Add and organize new literature\n"
        "  • Explore the library structure\n"
        "  • View and analyze file contents\n"
        "  • Move and reorganize files\n\n"
        "[dim]Type your message or 'exit' to quit. Type 'reset' to clear conversation history.[/dim]",
        border_style="cyan",
        title="🍄 PangGu Assistant"
    ))
    console.print()

    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break

            if user_input.lower() == "reset":
                service.reset()
                console.print("[green]✓ Conversation history reset[/green]\n")
                continue

            # Process the message with a spinner
            with console.status("[bold green]PangGu🍄 is thinking...", spinner="dots"):
                result = service.chat(user_input)

            # Display the response
            if result["success"]:
                console.print(f"\n[bold green]PangGu🍄:[/bold green] {result['response']}")
                if verbose:
                    console.print(f"[dim]({result['iterations']} iterations)[/dim]")
            else:
                console.print(f"\n[bold red]PangGu🍄:[/bold red] {result['response']}")
                if verbose:
                    console.print(f"[dim]Reason: {result['stop_reason']}[/dim]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            if verbose:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")



def _configure_logging(verbose: bool):
    """Configure logging based on verbose flag."""
    # Keep root logger at ERROR to silence third-party libraries
    root = logging.getLogger()
    root.setLevel(logging.ERROR)

    # Create a dedicated handler for pangu_agent loggers
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=verbose,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    if verbose:
        handler.setLevel(logging.INFO)
        level = logging.INFO
    else:
        handler.setLevel(logging.ERROR)
        level = logging.ERROR

    # Configure all pangu_agent module loggers with their own handler
    for logger_name in ["pangu_agent", "pangu_agent.agent", "pangu_agent.llm_client",
                        "pangu_agent.services", "pangu_agent.library"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False  # Don't propagate to root logger

    # Suppress warnings from third-party libraries
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def _print_add_results(results: list[dict[str, Any]]):
    """Print results from add operation with rich formatting."""
    if not results:
        console.print("[yellow]No files were added.[/yellow]")
        return

    success_count = sum(1 for r in results if r.get("success"))
    total_count = len(results)

    # Create a table for results
    table = Table(title=f"\nPangGu🍄 added {success_count}/{total_count} file(s)", show_header=True, header_style="bold")
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
    console.print(f"\n[green]✓ PangGu🍄 completed search in {iterations} iteration(s)[/green]\n")

    # Files table
    if files:
        table = Table(title=f"PangGu🍄 found {len(files)} relevant file(s)", show_header=True, header_style="bold cyan")
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
                console.print(f"\n[green]✓ PangGu🍄 successfully reset library:[/green] {library_root}")
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
