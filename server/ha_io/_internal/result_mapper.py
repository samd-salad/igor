"""Map cognition.ConversationResult to HA's response shape."""
from __future__ import annotations
import time

from server.cognition.contracts import ConversationResult
from server.ha_io.contracts import ConversationResponse


def map_result(result: ConversationResult,
               ha_conversation_id: str | None) -> ConversationResponse:
    return ConversationResponse(
        response=result.response_text,
        conversation_id=ha_conversation_id or f"igor-{int(time.time()*1000)}",
        end_conversation=result.end_conversation,
        commands_executed=result.commands_executed,
        silent=result.silent,
    )
