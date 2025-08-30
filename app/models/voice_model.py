"""Data models for the application."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class VoiceType(str, Enum):
    """Voice type enumeration."""
    RECORDED = "recorded"
    UPLOADED = "uploaded"
    PRESET = "preset"


class VoiceProfile(BaseModel):
    """Voice profile model."""
    id: str
    name: str
    type: VoiceType
    file_path: str
    created_at: datetime = Field(default_factory=datetime.now)
    description: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GenerationRequest(BaseModel):
    """TTS generation request model."""
    text: str
    voice_id: str
    num_speakers: int = Field(default=1, ge=1, le=4)
    cfg_scale: float = Field(default=1.3, ge=1.0, le=2.0)
    output_format: str = Field(default="wav")


class GenerationResponse(BaseModel):
    """TTS generation response model."""
    success: bool
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    message: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)


class AudioRecording(BaseModel):
    """Audio recording model."""
    name: str
    audio_data: str  # Base64 encoded audio
    format: str = "wav"
