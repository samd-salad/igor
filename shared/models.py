"""Pydantic models for API contracts between client and server."""
import base64
from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, field_validator


# Enums for validated string fields
class BeepType(str, Enum):
    """Valid beep types."""
    ALERT = "alert"
    ERROR = "error"
    DONE = "done"
    START = "start"
    END = "end"


class Priority(str, Enum):
    """Valid priority levels."""
    NORMAL = "normal"
    ALERT = "alert"


class Status(str, Enum):
    """Valid status values."""
    SUCCESS = "success"
    ERROR = "error"


class HealthStatus(str, Enum):
    """Valid health status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


# Request/Response Models
class ProcessInteractionRequest(BaseModel):
    """Request from Pi to PC for processing a voice interaction."""
    audio_base64: str = Field(..., max_length=10_000_000, description="Base64-encoded WAV audio file (max ~7MB)")
    wake_word: str = Field(..., max_length=50, description="Wake word that was detected")
    timestamp: float = Field(..., ge=0, description="Unix timestamp when interaction started")
    prefer_sonos_output: bool = Field(False, description="If true, client wants TTS routed to Sonos instead of returned as audio")

    @field_validator('wake_word')
    @classmethod
    def validate_wake_word(cls, v: str) -> str:
        """Allow only word characters, spaces, and hyphens — no control chars or injection sequences."""
        import re
        if not re.match(r'^[\w\s\-]+$', v):
            raise ValueError("wake_word contains invalid characters")
        return v

    @field_validator('audio_base64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that audio is proper base64 encoding."""
        if not v:
            raise ValueError("audio_base64 cannot be empty")
        try:
            decoded = base64.b64decode(v, validate=True)
            # Sanity check decoded size (max ~7.5MB WAV)
            if len(decoded) > 8_000_000:
                raise ValueError("Decoded audio exceeds maximum size")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}")


class ProcessInteractionResponse(BaseModel):
    """Response from PC to Pi with processed results."""
    transcription: str = Field(..., max_length=10_000, description="Speech-to-text result")
    response_text: str = Field(..., max_length=50_000, description="LLM's text response")
    audio_base64: str = Field(..., max_length=20_000_000, description="Base64-encoded WAV audio of TTS response")
    commands_executed: List[str] = Field(default_factory=list, max_length=50, description="List of command names executed")
    timings: Dict[str, float] = Field(default_factory=dict, description="Performance timings for each stage")
    speaker: Optional[str] = Field(None, max_length=100, description="Identified speaker name, if recognized")
    await_followup: bool = Field(False, description="If true, bot is expecting a follow-up response (no wake word needed)")
    tts_routed: bool = Field(False, description="If true, TTS was sent to Sonos; client should not play audio locally")
    error: Optional[str] = Field(None, max_length=1000, description="Error message if processing failed")


class PlayAudioRequest(BaseModel):
    """Request from PC to Pi to play audio (for timer alerts, etc.)."""
    audio_base64: str = Field(..., max_length=20_000_000, description="Base64-encoded WAV audio to play")
    message: str = Field(..., max_length=500, description="Text message being spoken")
    priority: Priority = Field(default=Priority.NORMAL, description="Priority level")

    @field_validator('audio_base64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that audio is proper base64 encoding."""
        try:
            base64.b64decode(v, validate=True)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}")


class PlayAudioResponse(BaseModel):
    """Response from Pi after playing audio."""
    status: Status = Field(..., description="Success or error status")
    played_at: float = Field(..., description="Unix timestamp when audio was played")
    error: Optional[str] = Field(None, max_length=1000, description="Error message if playback failed")


class HardwareControlRequest(BaseModel):
    """Request from PC to Pi to execute hardware command."""
    command: str = Field(..., max_length=50, description="Command name (e.g., 'set_volume')")
    parameters: Dict = Field(default_factory=dict, description="Command parameters")

    @field_validator('command')
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command is in whitelist."""
        # Whitelist of allowed hardware commands
        ALLOWED_COMMANDS = {
            'set_volume',
            'get_volume',
        }
        if v not in ALLOWED_COMMANDS:
            raise ValueError(f"Hardware command not allowed: {v}")
        return v

    @field_validator('parameters')
    @classmethod
    def validate_parameters_size(cls, v: Dict) -> Dict:
        """Limit parameters dict size."""
        if len(str(v)) > 1000:
            raise ValueError("Parameters too large")
        return v


class HardwareControlResponse(BaseModel):
    """Response from Pi after executing hardware command."""
    status: Status = Field(..., description="Success or error status")
    result: str = Field(..., max_length=1000, description="Result message from command execution")
    error: Optional[str] = Field(None, max_length=1000, description="Error message if command failed")


class PlayBeepRequest(BaseModel):
    """Request from PC to Pi to play a beep sound."""
    beep_type: BeepType = Field(..., description="Type of beep")


class PlayBeepResponse(BaseModel):
    """Response from Pi after playing beep."""
    status: Status = Field(..., description="Success or error status")
    error: Optional[str] = Field(None, max_length=1000, description="Error message if beep failed")


class HealthCheckResponse(BaseModel):
    """Health check response from either client or server."""
    status: HealthStatus = Field(..., description="Health status")
    services: Dict[str, str] = Field(default_factory=dict, description="Status of individual services")
    uptime_seconds: float = Field(..., ge=0, description="Uptime in seconds")
    additional_info: Dict = Field(default_factory=dict, description="Additional diagnostic information")

    @field_validator('services', 'additional_info')
    @classmethod
    def validate_dict_size(cls, v: Dict) -> Dict:
        """Limit dict sizes to prevent DoS."""
        if len(str(v)) > 10_000:
            raise ValueError("Dictionary too large")
        return v
