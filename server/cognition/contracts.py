"""Public contracts for the cognition bounded context.

External callers (ha_io, external, main) import ONLY from this module.
Aggregates and services may import from this module and from ports/.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------- Room & device context ----------

@dataclass(frozen=True)
class RoomConfig:
    """Minimal room descriptor. `ha_area` is the source of truth for device targeting."""
    room_id: str
    display_name: str
    ha_area: Optional[str] = None


# ---------- Tool call records ----------

@dataclass(frozen=True)
class ToolCallRecord:
    """A single LLM-issued tool invocation that ran during a turn."""
    name: str
    args: dict
    result: str


# ---------- The cross-cutting flow object ----------

@dataclass(frozen=True)
class VoiceTurn:
    """One conversational turn. Its `correlation_id` is the future Episode's `episode_id`.
    Stamped on every persistent write produced during the turn (facts, episodes, etc.)."""
    correlation_id: str
    started_at: datetime
    device_id: Optional[str]
    room: RoomConfig
    input_text: str
    speaker_id: Optional[str]          # nullable from day 1; resemblyzer fills later
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationResult:
    """Result of Conversation.process(turn). Returned to ha_io for mapping back to HA."""
    correlation_id: str
    response_text: str
    commands_executed: list[str]
    end_conversation: bool


# ---------- Memory aggregate entities ----------

@dataclass(frozen=True)
class Fact:
    """Semantic fact. Bi-temporal columns enable 'we don't live there anymore'
    without deletion. `embedding` is the adjacent vector slot; nullable until
    HybridRetrieval is enabled."""
    fact_id: str
    category: str
    key: str
    value: str
    tags: list[str]
    source_episode_id: Optional[str]
    embedding: Optional[bytes]
    valid_at: datetime         # world time
    invalid_at: Optional[datetime]   # null = currently true
    created_at: datetime       # transaction time


# ---------- Episode aggregate entity (also provenance anchor) ----------

@dataclass(frozen=True)
class Episode:
    """One conversational turn, persisted as a structured entity (not a summary string).
    `episode_id` == `VoiceTurn.correlation_id` always."""
    episode_id: str
    occurred_at: datetime
    speaker_id: Optional[str]
    participants: list[str]
    intent: Optional[str]
    raw_utterance: str
    tool_calls: list[ToolCallRecord]
    emotional_tone: Optional[str]
    summary: Optional[str]
    consolidated_at: Optional[datetime]


# ---------- Identity aggregate sub-collection ----------

@dataclass(frozen=True)
class Reflection:
    """Agent meta-note. Produced by Consolidator when noticing patterns
    about its own performance."""
    reflection_id: str
    occurred_at: datetime
    note: str
    source_episode_id: Optional[str]


# ---------- UserState aggregate sub-collections ----------

@dataclass(frozen=True)
class FeedbackEntry:
    feedback_id: str
    occurred_at: datetime
    issue: str
    status: str       # "open" | "resolved"
    source_episode_id: Optional[str]


@dataclass(frozen=True)
class Reminder:
    reminder_id: str
    name: str
    fire_at: datetime
    room_id: Optional[str]
    status: str       # "pending" | "fired" | "cancelled"
    source_episode_id: Optional[str]
