#!/usr/bin/env python3
"""MCP server exposing Igor commands for development/testing with Claude Code."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
import server.commands as commands

mcp = FastMCP("igor")


@mcp.tool()
def list_commands() -> str:
    """List all available Igor commands and their parameter schemas."""
    lines = []
    for name, cmd in commands.get_all_commands().items():
        params = list(cmd.parameters.keys())
        required = cmd.required_parameters
        lines.append(f"{name}: {cmd.description}")
        if params:
            lines.append(f"  params: {params} (required: {required})")
    return "\n".join(lines)


@mcp.tool()
def run_command(name: str, args: str = "{}") -> str:
    """Execute an Igor command by name with JSON-encoded args.

    Args:
        name: Command name (e.g. 'set_timer', 'calculate', 'get_time')
        args: JSON object of arguments (e.g. '{"name": "pasta", "duration": "5 minutes"}')
    """
    try:
        kwargs = json.loads(args)
    except json.JSONDecodeError as e:
        return f"Invalid JSON args: {e}"
    return commands.execute(name, **kwargs)


@mcp.tool()
def get_command_schema(name: str) -> str:
    """Get the full tool schema for a specific command.

    Args:
        name: Command name
    """
    all_commands = commands.get_all_commands()
    if name not in all_commands:
        return f"Unknown command: {name}. Use list_commands() to see available commands."
    return json.dumps(all_commands[name].to_tool(), indent=2)


if __name__ == "__main__":
    mcp.run()
