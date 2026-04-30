#!/usr/bin/env python3
"""MCP server exposing Igor commands and pipeline testing tools for Claude Code.

Three groups of tools:
  1. Command tools — list, run, and inspect Igor voice commands.
  2. Pipeline testing — probe individual pipeline stages (quality gate, intent
     router, STT, TTS) or run the full pipeline end-to-end without a Pi.
  3. Diagnostics — benchmark runner, log tail.

The MCP server runs independently of the main FastAPI server.  Heavy models
(Whisper, Kokoro) are loaded lazily on first use to keep startup fast.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
import server.commands as commands

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp_server")

mcp = FastMCP("igor")

def _ensure_brain():
    """Initialize BrainStore if not already done (MCP server runs independently)."""
    from server.brain import get_brain, init_brain
    try:
        return get_brain()
    except RuntimeError:
        from server.config import BRAIN_FILE
        brain = init_brain(BRAIN_FILE)
        brain.migrate_legacy_files(BRAIN_FILE.parent)
        logger.info("BrainStore initialized (lazy init)")
        return brain


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


# ---------------------------------------------------------------------------
# Pipeline testing tools — probe individual stages without a Pi or full server.
# ---------------------------------------------------------------------------

@mcp.tool()
def test_intent_router(phrase: str) -> str:
    """Test the Tier 1 intent router with a phrase.

    Returns the routing result: matched command + params + response text,
    or 'no match' if the phrase would fall through to the LLM.

    Args:
        phrase: The transcription text to route (e.g. "can you pause", "lights off")
    """
    from server.intent_router import route
    match = route(phrase)
    if match is None:
        return json.dumps({
            "tier": "LLM (no Tier 1 match)",
            "phrase": phrase,
            "match": None,
        }, indent=2)
    return json.dumps({
        "tier": "Tier 1",
        "phrase": phrase,
        "match": {
            "command": match.command,
            "params": match.params,
            "response": match.response,
        },
    }, indent=2)


@mcp.tool()
def test_quality_gate(text: str, tv_playing: bool = False) -> str:
    """Test the quality gate filter on a transcription.

    Returns whether the text would be accepted or rejected, and the cleaned
    output if accepted.  Set tv_playing=True to simulate TV-playing context.

    Args:
        text: Raw transcription text to filter
        tv_playing: Whether to simulate TV playing (enables long-text rejection)
    """
    from server.quality_gate import filter_transcription
    gate = filter_transcription(text, tv_playing=tv_playing)
    if gate.text is None:
        return json.dumps({
            "verdict": "REJECTED",
            "reason": gate.reason,
            "input": text,
            "tv_playing": tv_playing,
            "output": None,
        }, indent=2)
    return json.dumps({
        "verdict": "ACCEPTED",
        "input": text,
        "tv_playing": tv_playing,
        "output": result,
    }, indent=2)


@mcp.tool()
def test_pipeline(text: str, tv_playing: bool = False) -> str:
    """Run text through the full pipeline: quality gate -> intent router -> LLM -> TTS.

    Simulates process_interaction() without needing audio input or a Pi.
    Skips STT (uses provided text directly).  Returns results and timings
    from each stage.

    Args:
        text: The transcription text to process (as if Whisper produced it)
        tv_playing: Simulate TV-playing context
    """
    timings = {}
    stages = []

    # Stage 1: Quality Gate
    t0 = time.perf_counter()
    from server.quality_gate import filter_transcription
    gate = filter_transcription(text, tv_playing=tv_playing)
    timings["quality_gate_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    cleaned = gate.text

    if cleaned is None:
        stages.append({"stage": "quality_gate", "result": "REJECTED", "reason": gate.reason})
        return json.dumps({
            "text": text,
            "tv_playing": tv_playing,
            "final_result": "Rejected by quality gate",
            "stages": stages,
            "timings": timings,
        }, indent=2)
    stages.append({"stage": "quality_gate", "result": "ACCEPTED", "cleaned": cleaned})

    # Stage 2: Intent Router
    t0 = time.perf_counter()
    from server.intent_router import route as route_intent
    tier1 = route_intent(cleaned)
    timings["intent_router_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if tier1 is not None:
        stages.append({
            "stage": "intent_router",
            "result": "Tier 1 match",
            "command": tier1.command,
            "params": tier1.params,
            "response": tier1.response,
        })

        # Stage 3 (Tier 1): Execute command
        t0 = time.perf_counter()
        try:
            cmd_result = commands.execute(tier1.command, **tier1.params)
        except Exception as e:
            cmd_result = f"Error: {e}"
        timings["command_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        stages.append({"stage": "command", "result": cmd_result})

        return json.dumps({
            "text": text,
            "tv_playing": tv_playing,
            "tier": "Tier 1",
            "response_text": tier1.response,
            "final_result": cmd_result,
            "stages": stages,
            "timings": timings,
        }, indent=2)

    stages.append({"stage": "intent_router", "result": "No Tier 1 match -> LLM"})

    # Stage 3 (Tier 2): LLM
    t0 = time.perf_counter()
    try:
        from server.llm import LLM
        import server.commands as commands

        llm = LLM()
        tools = commands.get_tools()
        brain = _ensure_brain()
        behavior_rules = brain.get_behavior_rules()
        relevant = brain.retrieve_relevant(cleaned)
        relevant_memories = brain.format_relevant(relevant)
        llm_commands = []

        def tool_executor(command_name: str, **kwargs) -> str:
            llm_commands.append(command_name)
            return commands.execute(command_name, **kwargs)

        patterns = ""
        if tv_playing:
            patterns = (
                "TV is currently playing. Extract only clear, direct commands "
                "and ignore everything else."
            )

        chat_result = llm.chat(
            user_text=cleaned,
            tools=tools,
            tool_executor=tool_executor,
            behavior_rules=behavior_rules,
            patterns=patterns,
            relevant_memories=relevant_memories,
        )
        llm_text = chat_result.text if chat_result else "LLM returned no result"
    except Exception as e:
        llm_text = f"LLM error: {e}"
        llm_commands = []
    timings["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    stages.append({
        "stage": "llm",
        "response": llm_text[:500] if llm_text else None,
        "commands_executed": llm_commands,
    })

    return json.dumps({
        "text": text,
        "tv_playing": tv_playing,
        "tier": "Tier 2 (LLM)",
        "response_text": llm_text[:500] if llm_text else None,
        "commands_executed": llm_commands,
        "stages": stages,
        "timings": timings,
    }, indent=2)


@mcp.tool()
def run_benchmark(phrases: str = "") -> str:
    """Run a batch of test phrases through quality gate + intent router + TTS.

    Tests routing correctness and TTS latency for common commands.  Results
    are appended to data/benchmark.csv.  Provide custom phrases as a
    comma-separated string, or leave empty to use the built-in test suite.

    Args:
        phrases: Comma-separated test phrases (empty = use defaults)
    """
    # Default test suite covers exact matches, pattern matches, and LLM fallthrough
    default_phrases = [
        # Exact matches (should all be Tier 1)
        "pause", "play", "resume", "stop", "next", "lights off", "lights on", "mute", "unmute",
        # Pattern matches (should all be Tier 1)
        "can you pause", "please resume", "turn the lights off", "volume to 50",
        "mute the tv", "go ahead and play",
        # LLM fallthrough (should NOT be Tier 1)
        "what time is it", "set a timer for 5 minutes", "make the lights blue",
        "what's the weather", "turn on the living room lights",
        # Quality gate rejections
        "thank you.", "um", "the",
        # Edge cases
        "lights on warm", "don't mute",
    ]

    if phrases.strip():
        test_phrases = [p.strip() for p in phrases.split(",") if p.strip()]
    else:
        test_phrases = default_phrases

    from server.quality_gate import filter_transcription
    from server.intent_router import route as route_intent

    results = []
    for phrase in test_phrases:
        row = {"phrase": phrase}

        # Quality gate
        gate = filter_transcription(phrase)
        cleaned = gate.text
        if cleaned is None:
            row["gate"] = f"REJECTED ({gate.reason})"
            row["tier"] = "-"
            row["command"] = "-"
            row["response"] = "-"
        else:
            row["gate"] = "OK"

            # Intent router
            match = route_intent(cleaned)
            if match:
                row["tier"] = "Tier 1"
                row["command"] = f"{match.command}({match.params})"
                row["response"] = match.response
            else:
                row["tier"] = "Tier 2 (LLM)"
                row["command"] = "-"
                row["response"] = "-"

        results.append(row)

    # Append to benchmark CSV
    csv_path = Path(__file__).parent / "data" / "benchmark.csv"
    try:
        import csv
        write_header = not csv_path.exists()
        fieldnames = ["timestamp", "phrase", "gate", "tier", "command", "response"]
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            for row in results:
                row["timestamp"] = ts
                writer.writerow(row)
        csv_note = f"Appended {len(results)} rows to {csv_path}"
    except Exception as e:
        csv_note = f"CSV write failed: {e}"

    # Format summary table
    lines = [f"{'PHRASE':<35} {'GATE':<10} {'TIER':<15}"]
    lines.append("-" * 60)
    for r in results:
        lines.append(f"{r['phrase']:<35} {r['gate']:<10} {r['tier']:<15}")
    lines.append("")
    lines.append(csv_note)

    return "\n".join(lines)


@mcp.tool()
def tail_logs(lines: int = 50, level: str = "all") -> str:
    """Return recent server log lines from the log file.

    Reads the last N lines from data/server.log.  Filter by level to show
    only errors, warnings, etc.

    Args:
        lines: Number of recent lines to return (default 50, max 500)
        level: Filter by log level: 'all', 'error', 'warning', 'info' (case-insensitive)
    """
    import os
    lines = max(1, min(int(lines), 500))
    level = level.upper().strip()

    # Check common log file locations
    log_paths = [
        Path(__file__).parent / "data" / "server.log",
        Path(__file__).parent / "server.log",
    ]
    log_file = None
    for p in log_paths:
        if p.exists():
            log_file = p
            break

    if log_file is None:
        # No log file — try reading from systemd journal on Linux
        if sys.platform != "win32":
            import subprocess
            try:
                result = subprocess.run(
                    ["journalctl", "-u", "igor-server", "-n", str(lines), "--no-pager"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    output = result.stdout
                    if level != "ALL":
                        output = "\n".join(
                            l for l in output.splitlines()
                            if level in l.upper()
                        )
                    return output or f"No {level} entries in journal"
            except Exception:
                pass
        return (
            "No log file found. Checked:\n"
            + "\n".join(f"  - {p}" for p in log_paths)
            + "\n\nTo enable file logging, add a FileHandler in server/main.py."
        )

    # Read last N lines from log file (efficient tail)
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:]

        if level != "ALL":
            tail = [l for l in tail if level in l.upper()]

        if not tail:
            return f"No {level} entries in last {lines} lines"

        return "".join(tail)
    except Exception as e:
        return f"Error reading {log_file}: {e}"


if __name__ == "__main__":
    mcp.run()
