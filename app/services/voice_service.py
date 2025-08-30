"""Voice synthesis service using VibeVoice."""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime
import uuid

from app.config import settings
from app.models import VoiceProfile, VoiceType

logger = logging.getLogger(__name__)


class VoiceService:
    """Service for voice synthesis operations."""

    def __init__(self):
        """Initialize the voice service."""
        self.model = None
        self.processor = None
        self.voices_cache: Dict[str, VoiceProfile] = {}
        self._initialize_model()
        self._load_voices()

    def _initialize_model(self):
        """Initialize the VibeVoice model."""
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            logger.info(f"Loading model from {settings.MODEL_PATH}")

            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)

            # Load model
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                settings.MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map=settings.DEVICE,
                attn_implementation='flash_attention_2'
            )
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to mock mode for development
            logger.warning("Running in mock mode without actual model")

    def _load_voices(self):
        """Load available voice profiles."""
        # Load preset voices
        for voice_file in settings.VOICES_DIR.glob("*.wav"):
            voice_id = str(uuid.uuid4())
            profile = VoiceProfile(
                id=voice_id,
                name=voice_file.stem,
                type=VoiceType.PRESET,
                file_path=str(voice_file)
            )
            self.voices_cache[voice_id] = profile

    def generate_speech(
            self,
            text: str,
            voice_id: str,
            num_speakers: int = 1,
            cfg_scale: float = 1.3
    ) -> Optional[np.ndarray]:
        """Generate speech from text."""
        try:
            if not self.model or not self.processor:
                # Mock generation for development
                logger.warning("Using mock audio generation")
                return self._generate_mock_audio()

            # Get voice profile
            voice_profile = self.voices_cache.get(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice profile {voice_id} not found")

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                voice_samples=[[voice_profile.file_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Generate audio
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False,
            )

            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                return outputs.speech_outputs[0].cpu().numpy()

            return None

        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            return None

    def _generate_mock_audio(self) -> np.ndarray:
        """Generate mock audio for testing."""
        duration = 3.0  # seconds
        sample_rate = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate a simple tone
        frequency = 440  # A4 note
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        return audio

    def add_voice_profile(
            self,
            name: str,
            audio_path: str,
            voice_type: VoiceType = VoiceType.UPLOADED
    ) -> VoiceProfile:
        """Add a new voice profile."""
        voice_id = str(uuid.uuid4())
        profile = VoiceProfile(
            id=voice_id,
            name=name,
            type=voice_type,
            file_path=audio_path
        )
        self.voices_cache[voice_id] = profile
        return profile

    def get_voice_profiles(self) -> List[VoiceProfile]:
        """Get all available voice profiles."""
        return list(self.voices_cache.values())

    def get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get a specific voice profile."""
        return self.voices_cache.get(voice_id)
