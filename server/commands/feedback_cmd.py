"""Feedback / change-request logging commands."""
import logging

from server.commands.base import Command
from server.commands.memory_cmd import _sanitize

logger = logging.getLogger(__name__)


class LogFeedbackCommand(Command):
    name = "log_feedback"
    description = (
        "Log a change request or complaint about something the assistant just did. "
        "Call this after gathering what went wrong and what the user wants instead. "
        "If you need more detail, ask first, then log once you have it."
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
        issue = _sanitize(issue, max_len=500)
        suggestion = _sanitize(suggestion, max_len=500)
        context = _sanitize(context, max_len=500)
        from server.brain import get_brain
        brain = get_brain()
        new_id = brain.add_feedback(issue, suggestion, context)
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
        from server.brain import get_brain
        brain = get_brain()
        items = brain.list_feedback(status)
        if not items:
            return "No feedback items." if status == "all" else f"No {status} feedback items."

        lines = []
        for entry in items:
            d = entry["data"]
            line = f"#{d.get('id', '?')} [{entry['status']}] {d.get('timestamp', '')}: {d.get('issue', '')}"
            if d.get("suggestion"):
                line += f" → {d['suggestion']}"
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
        from server.brain import get_brain
        brain = get_brain()
        entry = brain.resolve_feedback(id)
        if not entry:
            return f"No feedback item with ID {id}."

        suggestion = entry["data"].get("suggestion", "").strip()
        if suggestion:
            value = _sanitize(suggestion, max_len=500)
            brain.save_memory("behavior", f"feedback_{id}", value)
            logger.info(f"Feedback #{id} suggestion saved to behavior memory: {value}")
            return f"#{id} resolved. Behavior note saved."

        return f"#{id} resolved."
