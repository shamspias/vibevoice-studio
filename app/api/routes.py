"""API routes for the application."""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from app.models import (
    VoiceProfile,
    GenerationRequest,
    GenerationResponse,
    AudioRecording,
    VoiceType
)
from app.services import VoiceService, AudioService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Initialize services
voice_service = VoiceService()
audio_service = AudioService()


@router.get("/voices", response_model=List[VoiceProfile])
async def get_voices():
    """Get all available voice profiles."""
    return voice_service.get_voice_profiles()


@router.post("/voices/upload")
async def upload_voice(
        file: UploadFile = File(...),
        name: str = Form(...)
):
    """Upload a voice sample."""
    try:
        # Validate file format
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported format. Use: {settings.SUPPORTED_FORMATS}")

        # Save uploaded file
        filename = f"{name}_{uuid.uuid4().hex[:8]}{file_ext}"
        filepath = settings.VOICES_DIR / filename

        content = await file.read()
        with open(filepath, 'wb') as f:
            f.write(content)

        # Add voice profile
        profile = voice_service.add_voice_profile(
            name=name,
            audio_path=str(filepath),
            voice_type=VoiceType.UPLOADED
        )

        return {"success": True, "voice": profile}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))


@router.post("/voices/record")
async def record_voice(recording: AudioRecording):
    """Save a recorded voice sample."""
    try:
        # Convert base64 to audio file
        filepath = audio_service.base64_to_audio(
            recording.audio_data,
            settings.VOICES_DIR / f"{recording.name}_{uuid.uuid4().hex[:8]}.wav"
        )

        # Add voice profile
        profile = voice_service.add_voice_profile(
            name=recording.name,
            audio_path=filepath,
            voice_type=VoiceType.RECORDED
        )

        return {"success": True, "voice": profile}

    except Exception as e:
        logger.error(f"Recording error: {e}")
        raise HTTPException(500, str(e))


@router.post("/generate", response_model=GenerationResponse)
async def generate_speech(request: GenerationRequest):
    """Generate speech from text."""
    try:
        # Generate audio
        audio_array = voice_service.generate_speech(
            text=request.text,
            voice_id=request.voice_id,
            num_speakers=request.num_speakers,
            cfg_scale=request.cfg_scale
        )

        if audio_array is None:
            return GenerationResponse(
                success=False,
                message="Failed to generate audio"
            )

        # Save generated audio
        filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = audio_service.save_audio(
            audio_array,
            filename=filename
        )

        # Calculate duration
        duration = len(audio_array) / settings.SAMPLE_RATE

        return GenerationResponse(
            success=True,
            audio_url=f"/api/audio/{filename}",
            duration=duration,
            message="Audio generated successfully"
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return GenerationResponse(
            success=False,
            message=str(e)
        )


@router.post("/generate/file")
async def generate_from_file(
        file: UploadFile = File(...),
        voice_id: str = Form(...),
        cfg_scale: float = Form(1.3)
):
    """Generate speech from text file."""
    try:
        # Read text file
        content = await file.read()
        text = content.decode('utf-8')

        # Generate audio
        request = GenerationRequest(
            text=text,
            voice_id=voice_id,
            cfg_scale=cfg_scale
        )

        return await generate_speech(request)

    except Exception as e:
        logger.error(f"File generation error: {e}")
        raise HTTPException(500, str(e))


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """Get generated audio file."""
    filepath = settings.OUTPUTS_DIR / filename

    if not filepath.exists():
        raise HTTPException(404, "Audio file not found")

    return FileResponse(
        filepath,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": voice_service.model is not None
    }
