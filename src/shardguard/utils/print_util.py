from __future__ import annotations

import json
from typing import Any

from rich.console import Console

console = Console()


def print_json(data: Any, pretty: bool = True) -> None:
    if pretty:
        console.print_json(data=data)
    else:
        print_json(json.dumps(data, separators=(",", ":"), ensure_ascii=False))


def _print_tools_info(tools_description: str, verbose: bool = False) -> None:
    if "No MCP tools available." in tools_description:
        console.print("[dim]No MCP tools available.[/dim]")
        return

    if verbose:
        console.print("[bold blue]MCP Servers & Tools:[/bold blue]")
        for line in tools_description.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Server:"):
                server_name = stripped.replace("Server:", "").strip()
                console.print(f"[bold cyan]{server_name}[/bold cyan]")
            elif stripped.startswith("•"):
                tool_name = stripped.replace("•", "").strip()
                if ":" in tool_name:
                    tool_name = tool_name.split(":")[0]
                console.print(f"  └── [green]{tool_name}[/green]")
        console.print()
    else:
        tool_count, server_count = _count_tools_and_servers(tools_description)
        console.print(
            f"[dim]Available tools: {tool_count} tools from {server_count} servers[/dim]"
        )
        console.print("[bold blue]Available MCP Tools:[/bold blue]")
        console.print(tools_description)


def _count_tools_and_servers(tools_description: str) -> tuple[int, int]:
    """Count tools and servers from tools description."""
    lines = tools_description.split("\n")
    tool_count = len([line for line in lines if line.strip().startswith("•")])
    server_count = len([line for line in lines if line.strip().startswith("Server:")])
    return tool_count, server_count


def _print_verbose_tools_info(tools_description: str) -> None:
    """Print detailed server and tool information."""
    for line in tools_description.split("\n"):
        stripped_line = line.strip()
        if stripped_line.startswith("Server:"):
            server_name = stripped_line.replace("Server:", "").strip()
            console.print(f"[bold cyan]{server_name}[/bold cyan]")
        elif stripped_line.startswith("•"):
            tool_name = stripped_line.replace("•", "").strip()
            if ":" in tool_name:
                tool_name = tool_name.split(":")[0]
            console.print(f"  └── [green]{tool_name}[/green]")
    console.print()


def log_success(msg: str):
    console.print(f"[green]{msg}[/green]")


def log_err(msg: str):
    console.print(f"[red]{msg}[/red]")
