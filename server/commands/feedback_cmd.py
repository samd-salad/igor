"""Feedback / change-request logging commands."""
import json
import logging
from datetime import datetime

from server.commands.base import Command
from server.config import DATA_DIR

logger = logging.getLogger(__name__)

FEEDBACK_FILE = DATA_DIR / "feedback.json"


def _load() -> list:
    if not FEEDBACK_FILE.exists():
        return []
    try:
        return json.loads(FEEDBACK_FILE.read_text())
    except Exception:
        return []


def _save(items: list):
    FEEDBACK_FILE.write_text(json.dumps(items, indent=2))


class LogFeedbackCommand(Command):
    name = "log_feedback"
    description = (
        "Log a change request or complaint about something the assistant just did. "
        "Call this after gathering what went wrong and what the user wants instead. "
        "Use [AWAIT] before calling this if you need more detail from the user."
    )

    @property
    def parameters(self) -> dict:
        return {
            "issue": {
                "type": "string",
                "description": "What went wrong or what needs to change",
            },
            "suggestion": {
                "type": "string",
                "description": "What the user wants instead (optional)",
            },
            "context": {
                "type": "string",
                "description": "Brief summary of what the assistant did (e.g. the command run or response given)",
            },
        }

    @property
    def required_parameters(self) -> list:
        return ["issue"]

    def execute(self, issue: str, suggestion: str = "", context: str = "") -> str:
        items = _load()
        new_id = max((i["id"] for i in items), default=0) + 1
        items.append({
            "id": new_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "open",
            "issue": issue,
            "suggestion": suggestion,
            "context": context,
        })
        _save(items)
        logger.info(f"Feedback logged #{new_id}: {issue}")
        return f"Logged as #{new_id}."


class ListFeedbackCommand(Command):
    name = "list_feedback"
    description = "List open change requests and complaints. Use when asked to show the to-do list or pending feedback."

    @property
    def parameters(self) -> dict:
        return {
            "status": {
                "type": "string",
                "description": "Filter by status: 'open', 'resolved', or 'all'. Defaults to 'open'.",
            }
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, status: str = "open", **_) -> str:
        items = _load()
        if status != "all":
            items = [i for i in items if i.get("status") == status]
        if not items:
            return "No feedback items." if status == "all" else f"No {status} feedback items."

        lines = []
        for i in items:
            line = f"#{i['id']} [{i['status']}] {i['timestamp']}: {i['issue']}"
            if i.get("suggestion"):
                line += f" → {i['suggestion']}"
            lines.append(line)
        return "\n".join(lines)


class ResolveFeedbackCommand(Command):
    name = "resolve_feedback"
    description = "Mark a feedback item as resolved. Use when the user says an issue has been fixed."

    @property
    def parameters(self) -> dict:
        return {
            "id": {
                "type": "integer",
                "description": "ID of the feedback item to mark resolved",
            }
        }

    def execute(self, id: int, **_) -> str:
        items = _load()
        for item in items:
            if item["id"] == id:
                item["status"] = "resolved"
                _save(items)
                return f"#{id} resolved."
        return f"No feedback item with ID {id}."
