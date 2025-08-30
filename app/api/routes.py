"""API routes for the application."""

import os
import uuid
import logging
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import numpy as np

from app.models import (
    VoiceProfile,
    GenerationRequest,
    GenerationResponse,
    AudioRecording,
    VoiceType,
)
from app.services import VoiceService, AudioService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

voice_service = VoiceService()
audio_service = AudioService()


@router.get("/voices", response_model=List[VoiceProfile])
async def get_voices():
    return voice_service.get_voice_profiles()


@router.post("/voices/upload")
async def upload_voice(
        file: UploadFile = File(...),
        name: str = Form(...)
):
    try:
        logger.info(f"Uploading voice: {name}, file: {file.filename}")

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported format. Use: {settings.SUPPORTED_FORMATS}")

        content = await file.read()
        size = len(content)
        max_size = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        if size > max_size:
            raise HTTPException(400, f"File too large. Max {settings.MAX_AUDIO_SIZE_MB}MB")

        # save raw
        raw_path = settings.VOICES_DIR / f"{name}_{uuid.uuid4().hex[:8]}{file_ext}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved voice file to: {raw_path}")

        # convert to wav if needed
        final_path = raw_path
        if file_ext != ".wav":
            wav_path = raw_path.with_suffix(".wav")
            audio_service.convert_to_wav(str(raw_path), str(wav_path))
            try:
                os.remove(raw_path)
            except Exception:
                pass
            final_path = wav_path
            logger.info(f"Converted to WAV: {final_path}")

        profile = voice_service.add_voice_profile(
            name=name,
            audio_path=str(final_path),
            voice_type=VoiceType.UPLOADED,
        )
        return {"success": True, "voice": profile, "message": "Voice uploaded successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")


@router.post("/voices/record")
async def record_voice(recording: AudioRecording):
    """Save a recorded voice sample reliably (handles webm/mp4/ogg)."""
    try:
        logger.info(f"Saving recorded voice: {recording.name}")

        # Use container extension from client format; default webm
        ext = (recording.format or "webm").lower().lstrip(".")
        raw_path = settings.VOICES_DIR / f"{recording.name}_{uuid.uuid4().hex[:8]}.{ext}"

        # Write raw and convert to wav
        wav_path = audio_service.base64_to_audio(
            base64_data=recording.audio_data,
            output_path=raw_path,
            format=ext,
        )
        logger.info(f"Saved recording to: {wav_path}")

        profile = voice_service.add_voice_profile(
            name=recording.name,
            audio_path=wav_path,
            voice_type=VoiceType.RECORDED,
        )
        return {"success": True, "voice": profile, "message": "Recording saved successfully"}

    except Exception as e:
        logger.error(f"Recording error: {e}")
        raise HTTPException(500, f"Recording failed: {str(e)}")


@router.post("/generate", response_model=GenerationResponse)
async def generate_speech(request: GenerationRequest):
    try:
        logger.info(f"Generating speech for text length: {len(request.text)}")
        audio_array = voice_service.generate_speech(
            text=request.text,
            voice_id=request.voice_id,
            num_speakers=request.num_speakers,
            cfg_scale=request.cfg_scale,
        )

        if audio_array is None:
            return GenerationResponse(
                success=False,
                message="Failed to generate audio. Please check if model is loaded."
            )

        filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = audio_service.save_audio(audio_array, filename=filename)
        logger.info(f"Saved generated audio to: {filepath}")

        duration = len(audio_array) / settings.SAMPLE_RATE
        return GenerationResponse(
            success=True,
            audio_url=f"/api/audio/{filename}",
            duration=duration,
            message="Audio generated successfully",
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return GenerationResponse(success=False, message=f"Generation failed: {str(e)}")


@router.post("/generate/file")
async def generate_from_file(
        file: UploadFile = File(...),
        voice_id: str = Form(...),
        cfg_scale: float = Form(1.3),
):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        logger.info(f"Generating from file: {file.filename}, text length: {len(text)}")
        req = GenerationRequest(text=text, voice_id=voice_id, cfg_scale=cfg_scale)
        return await generate_speech(req)
    except Exception as e:
        logger.error(f"File generation error: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    filepath = settings.OUTPUTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(
        filepath,
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": voice_service.is_model_loaded(),
    }
