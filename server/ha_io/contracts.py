"""Pydantic models for HA's Custom Conversation Agent payload."""
from typing import Optional
from pydantic import BaseModel, Field


class ConversationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = Field(None, max_length=100)
    device_id: Optional[str] = Field(None, max_length=100)
    language: Optional[str] = Field(None, max_length=20)


class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    end_conversation: bool
    commands_executed: list[str]
    silent: bool = False                # HA integration skips TTS when true
