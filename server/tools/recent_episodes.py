"""Dump the last N episodes from brain.db with the full back-and-forth.

Run inside the container:
    docker exec igor python -m server.tools.recent_episodes
    docker exec igor python -m server.tools.recent_episodes 20
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

from server.external.sqlite_persistence import SqlitePersistence


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    db_path = Path(os.environ.get("BRAIN_DIR", "/app/data")) / "brain.db"
    if not db_path.exists():
        print(f"No brain.db at {db_path}", file=sys.stderr)
        sys.exit(1)
    sp = SqlitePersistence(db_path)
    episodes = sp.list_recent_episodes(n)
    if not episodes:
        print("(no episodes yet)")
        return
    for ep in reversed(episodes):
        print(f"--- {ep.occurred_at.isoformat()}  [{ep.intent}]  {ep.episode_id}")
        print(f"  user:  {ep.raw_utterance[:300]}")
        response = ep.response_text or "(not stored)"
        print(f"  igor:  {response[:300]}")
        if ep.tool_calls:
            print(f"  tools: {', '.join(tc.name for tc in ep.tool_calls)}")
        if ep.summary:
            print(f"  sum:   {ep.summary}")
        print()


if __name__ == "__main__":
    main()
