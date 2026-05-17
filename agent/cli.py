"""
agent/cli.py

CLI interface for the GPU benchmark agent.

Usage:
    python -m agent.cli "Which GPU has best LLM performance under $2000?"
    python -m agent.cli --interactive
"""
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

app = typer.Typer()
console = Console()


@app.command()
def chat(
    query: str = typer.Argument(None, help="Single question to ask the agent"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive chat session"),
):
    """PerfBot CLI — Ask questions about GPU benchmark data."""
    from agent.graph import run_agent

    if interactive or query is None:
        _interactive_session(run_agent)
    else:
        _single_query(query, run_agent)


def _single_query(query: str, run_agent):
    console.print(Panel(f"[bold cyan]Query:[/] {query}", border_style="cyan"))
    with console.status("[bold green]PerfBot is thinking..."):
        response = run_agent(query)
    console.print(Panel(Markdown(response), title="[bold green]PerfBot", border_style="green"))


def _interactive_session(run_agent):
    console.print(Panel(
        "[bold green]PerfBot Interactive Mode[/]\n"
        "Ask anything about GPU benchmarks. Type [bold red]exit[/] to quit.",
        border_style="green"
    ))

    history = []
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/]")
            break
        if not query:
            continue

        with console.status("[bold green]PerfBot is thinking..."):
            response = run_agent(query, history=history)

        console.print("\n[bold green]PerfBot:[/]")
        console.print(Markdown(response))

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    app()
