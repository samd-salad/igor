"""HA-backed todo / list commands.

HA's `todo` domain (added 2023.11) lets us add, complete, and read items on
any todo list integration the user has configured (Shopping List, Google
Tasks, Microsoft To Do, Local To-do, etc.).

Service field names verified against /api/services on the live HA (2026-04-18):
  todo.add_item             → item, due_date, due_datetime, description
  todo.update_item          → item, rename, status, due_date, due_datetime, description
  todo.remove_item          → item
  todo.get_items            → status (needs return_response=true)
  todo.remove_completed_items → (none)
"""
import logging
from typing import Optional

from .base import Command
from server.ha_client import HAError, get_client

logger = logging.getLogger(__name__)


def _friendly_name(state: dict) -> str:
    return (state.get("attributes") or {}).get("friendly_name", "") or ""


def _resolve_todo_list(name: str = "") -> Optional[str]:
    """Resolve a todo list name to its entity_id.

    Empty name → first todo list. Name matches against friendly_name (case-
    insensitive exact, then substring) and entity_id substring.
    """
    ha = get_client()
    todos = ha.states_in_domain("todo")
    if not todos:
        return None
    if not name:
        return todos[0]["entity_id"]
    name_lower = name.strip().lower()
    for s in todos:
        if _friendly_name(s).lower() == name_lower:
            return s["entity_id"]
    for s in todos:
        if name_lower in _friendly_name(s).lower() or name_lower in s["entity_id"]:
            return s["entity_id"]
    return None


# -- commands --

class AddTodoCommand(Command):
    name = "add_todo"
    description = (
        "Add an item to a to-do or shopping list. "
        "Use for 'add eggs to my shopping list', 'remind me to call mom', etc."
    )

    @property
    def parameters(self) -> dict:
        return {
            "item": {"type": "string", "description": "The item or task to add"},
            "list": {"type": "string", "description": "List name. Omit for the default list."},
        }

    @property
    def required_parameters(self) -> list:
        return ["item"]

    def execute(self, item: str, list: str = "") -> str:
        item = item.strip()
        if not item:
            return "Item is empty"
        eid = _resolve_todo_list(list)
        if not eid:
            return "No todo list configured in Home Assistant" if not list else f"Todo list '{list}' not found"
        try:
            get_client().call_service("todo", "add_item", {
                "entity_id": eid, "item": item,
            })
        except HAError as e:
            return f"Failed to add: {e}"
        list_name = _friendly_name(get_client().get_state(eid)) or eid
        return f"Added '{item}' to {list_name}"


class CompleteTodoCommand(Command):
    name = "complete_todo"
    description = "Mark a to-do or shopping list item as complete."

    @property
    def parameters(self) -> dict:
        return {
            "item": {"type": "string", "description": "The item to complete (matched by name)"},
            "list": {"type": "string", "description": "List name. Omit for the default list."},
        }

    @property
    def required_parameters(self) -> list:
        return ["item"]

    def execute(self, item: str, list: str = "") -> str:
        eid = _resolve_todo_list(list)
        if not eid:
            return "No todo list configured in Home Assistant" if not list else f"Todo list '{list}' not found"
        try:
            get_client().call_service("todo", "update_item", {
                "entity_id": eid, "item": item, "status": "completed",
            })
        except HAError as e:
            return f"Failed to complete: {e}"
        return f"Completed '{item}'"


class RemoveTodoCommand(Command):
    name = "remove_todo"
    description = "Remove an item from a to-do or shopping list (without marking it complete)."

    @property
    def parameters(self) -> dict:
        return {
            "item": {"type": "string", "description": "The item to remove (matched by name)"},
            "list": {"type": "string", "description": "List name. Omit for the default list."},
        }

    @property
    def required_parameters(self) -> list:
        return ["item"]

    def execute(self, item: str, list: str = "") -> str:
        eid = _resolve_todo_list(list)
        if not eid:
            return "No todo list configured in Home Assistant" if not list else f"Todo list '{list}' not found"
        try:
            get_client().call_service("todo", "remove_item", {
                "entity_id": eid, "item": item,
            })
        except HAError as e:
            return f"Failed to remove: {e}"
        return f"Removed '{item}'"


class ListTodoCommand(Command):
    name = "list_todo"
    description = "Read items from a to-do or shopping list."

    @property
    def parameters(self) -> dict:
        return {
            "list": {"type": "string", "description": "List name. Omit for the default list."},
            "status": {"type": "string", "description": "'needs_action' (default), 'completed', or 'all'"},
        }

    @property
    def required_parameters(self) -> list:
        return []

    def execute(self, list: str = "", status: str = "needs_action") -> str:
        eid = _resolve_todo_list(list)
        if not eid:
            return "No todo list configured in Home Assistant" if not list else f"Todo list '{list}' not found"
        ha = get_client()
        # Build payload: HA accepts "needs_action", "completed", or omit for all
        data: dict = {"entity_id": eid}
        s = (status or "").lower().strip()
        if s in ("needs_action", "completed"):
            data["status"] = s
        try:
            resp = ha.call_service("todo", "get_items", data, return_response=True)
        except HAError as e:
            return f"Failed to read list: {e}"
        items: list[dict] = []
        if isinstance(resp, dict):
            sr = resp.get("service_response", {})
            if isinstance(sr, dict):
                # HA returns {entity_id: {"items": [...]}}
                entry = sr.get(eid) or next(iter(sr.values()), {})
                items = entry.get("items", []) if isinstance(entry, dict) else []
        list_name = _friendly_name(ha.get_state(eid)) or eid
        if not items:
            label = f" {s}" if s in ("needs_action", "completed") else ""
            return f"{list_name} has no{label} items"
        # Sort by status (needs_action first), then summary
        items.sort(key=lambda it: (it.get("status") != "needs_action", it.get("summary", "")))
        lines = []
        for it in items:
            mark = "[ ]" if it.get("status") == "needs_action" else "[x]"
            lines.append(f"  {mark} {it.get('summary', '(unnamed)')}")
        return f"{list_name}:\n" + "\n".join(lines)


class ListTodoListsCommand(Command):
    name = "list_todo_lists"
    description = "Show all to-do / shopping lists configured in Home Assistant."

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        ha = get_client()
        todos = ha.states_in_domain("todo", force_refresh=True)
        if not todos:
            return "No todo lists configured in Home Assistant"
        lines = []
        for s in todos:
            count = s.get("state", "?")
            name = _friendly_name(s) or s["entity_id"]
            lines.append(f"- {name}: {count} item(s)")
        return "Todo lists:\n" + "\n".join(lines)
