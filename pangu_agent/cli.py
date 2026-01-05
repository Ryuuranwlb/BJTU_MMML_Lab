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

        # Display task header
        console.print()
        console.print("[bold cyan]┌─ 🍄 PangGu Add Literature ─────────────────┐[/bold cyan]")
        console.print(f"[bold cyan]│[/bold cyan] [dim]Path:[/dim] [cyan]{path}[/cyan]")
        if prompt:
            console.print(f"[bold cyan]│[/bold cyan] [dim]Context:[/dim] {prompt}")
        console.print("[bold cyan]└────────────────────────────────────────────┘[/bold cyan]")
        console.print()

        with console.status("[bold blue]│ Initializing...", spinner="dots"):
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
        console.print("[bold blue]│[/bold blue] Scanning for files...")
        files = service.scan_addable_files(path)

        if not files:
            console.print("[bold yellow]│[/bold yellow] No addable files found (PDFs and images)")
            console.print()
            return

        console.print(f"[bold green]│[/bold green] Found [bold]{len(files)}[/bold] file(s) to add")
        console.print()

        # Process files with progress bar
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[bold blue]│[/bold blue] [cyan]Processing files...", total=len(files))

            for file_path in files:
                file_name = Path(file_path).name
                progress.update(task, description=f"[bold blue]│[/bold blue] [cyan]{file_name}")

                result = service.add_file_with_llm(str(file_path), user_prompt=prompt)
                results.append(result)

                progress.advance(task)

        # Print results
        console.print()
        _print_add_results(results)
        return

    if action == "search":
        if not prompt:
            raise click.UsageError("--prompt is required for search.")

        # Display task header
        console.print()
        console.print("[bold cyan]┌─ 🍄 PangGu Search Library ─────────────────┐[/bold cyan]")
        console.print(f"[bold cyan]│[/bold cyan] [bold]Query:[/bold] {prompt}")
        console.print("[bold cyan]└────────────────────────────────────────────┘[/bold cyan]")
        console.print()

        with console.status("[bold blue]│ Initializing...", spinner="dots"):
            manager = LibraryManager(library_root)
            llm_client = LLMClient(log=verbose)
            tools = [
                SearchLibraryTool(manager),
                ViewFileTool(manager),
                ExploreLibraryTool(manager),
                FinishTool(),
            ]
            service = SearchFilesService(manager, llm_client, tools)

        with console.status("[bold green]│ Searching library...", spinner="dots"):
            result = service.search(prompt)

        # Print results
        console.print()
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
    # Lazy imports to avoid circular dependency
    from pangu_agent.services.interactive import InteractiveService
    from pangu_agent.tools.add_literature import AddLiteratureTool

    # Configure logging
    _configure_logging(verbose)

    # Clear screen for a fresh start
    console.clear()

    # Display banner
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]    🍄 PangGu Literature Assistant 🍄      [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════╝[/bold cyan]")
    console.print()

    # Initialize components
    with console.status("[bold blue]Loading components...", spinner="dots"):
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
    console.print(Panel(
        f"[dim]Library:[/dim] [cyan]{library_root}[/cyan]\n\n"
        "[bold]I can help you with:[/bold]\n"
        "  [green]•[/green] Search for papers and files\n"
        "  [green]•[/green] Add and organize new literature\n"
        "  [green]•[/green] Explore the library structure\n"
        "  [green]•[/green] View and analyze file contents\n"
        "  [green]•[/green] Move and reorganize files\n\n"
        "[dim]Commands: [cyan]exit[/cyan] to quit, [cyan]reset[/cyan] to clear history[/dim]",
        border_style="green",
        title="[bold green]Ready to assist![/bold green]",
        title_align="left",
    ))
    console.print()

    # Main interaction loop
    conversation_count = 0
    while True:
        try:
            # Get user input with a clean prompt
            user_input = console.input("[bold blue]│[/bold blue] [bold white]You[/bold white] [bold blue]›[/bold blue] ").strip()

            if not user_input:
                continue

            # Handle exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print()
                console.print("[dim]─[/dim]" * 50)
                console.print("[bold yellow]Thanks for using PangGu! Goodbye! 👋[/bold yellow]")
                console.print()
                break

            # Handle reset command
            if user_input.lower() == "reset":
                service.reset()
                conversation_count = 0
                console.print("[bold green]│[/bold green] [dim]✓ Conversation history cleared[/dim]\n")
                continue

            # Process the message with a spinner
            console.print()
            with console.status("[bold green]│[/bold green] [dim]PangGu🍄 is thinking...[/dim]", spinner="dots"):
                result = service.chat(user_input)

            # Display the response with visual separator
            conversation_count += 1
            console.print("[bold green]│[/bold green] [bold green]PangGu🍄[/bold green] [bold green]›[/bold green]", end=" ")

            if result["success"]:
                console.print(result['response'])
                if verbose:
                    console.print(f"[bold green]│[/bold green] [dim]↳ {result['iterations']} iterations[/dim]")
            else:
                console.print(f"[red]{result['response']}[/red]")
                if verbose:
                    console.print(f"[bold green]│[/bold green] [dim]↳ Error: {result['stop_reason']}[/dim]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n")
            console.print("[dim]─[/dim]" * 50)
            console.print("[bold yellow]Interrupted. Goodbye! 👋[/bold yellow]")
            console.print()
            break
        except Exception as e:
            console.print(f"\n[bold red]│[/bold red] [red]Error: {e}[/red]\n")
            if verbose:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]\n")



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
        console.print("[bold yellow]│[/bold yellow] No files were added.")
        console.print()
        return

    success_count = sum(1 for r in results if r.get("success"))
    total_count = len(results)

    # Summary header
    if success_count == total_count:
        status_text = "[bold green]✓ Success[/bold green]"
        status_color = "green"
    elif success_count > 0:
        status_text = "[bold yellow]⚠ Partial Success[/bold yellow]"
        status_color = "yellow"
    else:
        status_text = "[bold red]✗ Failed[/bold red]"
        status_color = "red"

    console.print(f"{status_text} - Added [bold]{success_count}[/bold] of [bold]{total_count}[/bold] file(s)")
    console.print()

    # Create a table for results
    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style=status_color,
        title="📚 Results",
        title_style=f"bold {status_color}"
    )
    table.add_column("", style="dim", width=3, justify="center")
    table.add_column("Source File", style="cyan", no_wrap=False)
    table.add_column("Destination", style="green", no_wrap=False)

    for i, result in enumerate(results, 1):
        source = Path(result.get("source", "unknown")).name
        if result.get("success"):
            dest = result.get("destination", "unknown")
            table.add_row("✓", source, dest)
        else:
            error = result.get("error", "unknown error")
            # Truncate long error messages
            error_short = error if len(error) < 50 else error[:47] + "..."
            table.add_row("✗", source, f"[red]{error_short}[/red]")

    console.print(table)
    console.print()


def _print_search_results(result: dict[str, Any]):
    """Print results from search operation with rich formatting."""
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        console.print()
        console.print(Panel(
            f"[bold red]✗ Search Failed[/bold red]\n\n{error}",
            border_style="red",
            title="Error"
        ))
        console.print()
        return

    files = result.get("files", [])
    observation = result.get("observation", "")
    iterations = result.get("iterations", "N/A")

    # Success header
    console.print(f"[bold green]✓ Search completed[/bold green] [dim]({iterations} iterations)[/dim]")
    console.print()

    # Files table
    if files:
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="green",
            title=f"📄 Found {len(files)} relevant file(s)",
            title_style="bold green"
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("File Path", style="cyan", no_wrap=False)

        for i, file_path in enumerate(files, 1):
            table.add_row(str(i), file_path)

        console.print(table)
        console.print()
    else:
        console.print("[bold yellow]│[/bold yellow] No relevant files found")
        console.print()

    # Observation panel
    if observation:
        console.print(Panel(
            observation,
            title="[bold cyan]💡 Observation[/bold cyan]",
            title_align="left",
            border_style="blue",
            padding=(1, 2)
        ))
        console.print()


def _reset_library(library_root: str):
    """Reset the library by removing all contents."""
    # Display warning header
    console.print()
    console.print("[bold red]┌─ ⚠️  WARNING ──────────────────────────────┐[/bold red]")
    console.print(f"[bold red]│[/bold red] This will delete all contents in:")
    console.print(f"[bold red]│[/bold red] [cyan]{library_root}[/cyan]")
    console.print("[bold red]└────────────────────────────────────────────┘[/bold red]")
    console.print()

    if not click.confirm("Are you sure you want to reset the library?"):
        console.print("[bold yellow]│[/bold yellow] Reset cancelled")
        console.print()
        return

    # Initialize manager and perform reset
    try:
        console.print()
        with console.status("[bold red]│ Resetting library...", spinner="dots"):
            manager = LibraryManager(library_root)
            result = manager.reset()

        console.print()
        if result["success"]:
            removed_count = result["removed_count"]
            message = result.get("message", "")

            if removed_count == 0:
                console.print(f"[bold yellow]│[/bold yellow] {message}")
            else:
                console.print(f"[bold green]│[/bold green] ✓ Library reset successfully")
                console.print(f"[bold green]│[/bold green] [dim]Removed {removed_count} item(s)[/dim]")
        else:
            error = result.get("error", "Unknown error")
            console.print(Panel(
                f"[bold red]Reset Failed[/bold red]\n\n{error}",
                border_style="red",
                title="Error"
            ))
    except Exception as e:
        console.print(Panel(f"[red]✗ Failed to reset library:[/red] {e}", border_style="red"))



def main():
    pangu()


if __name__ == "__main__":
    main()
