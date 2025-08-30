"""Voice synthesis service using VibeVoice."""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime
import uuid
import traceback

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
        self.model_loaded = False
        self._initialize_model()
        self._load_voices()

    def _initialize_model(self):
        """Initialize the VibeVoice model."""
        try:
            # Check if VibeVoice is installed
            try:
                from vibevoice.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference
                )
                from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            except ImportError:
                logger.error("VibeVoice not installed. Please install it first.")
                logger.info(
                    "Run: git clone https://github.com/microsoft/VibeVoice.git && cd VibeVoice && pip install -e .")
                return

            logger.info(f"Loading model from {settings.MODEL_PATH}")

            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)

            # Try to load model with different attention implementations
            try:
                # Try flash_attention_2 first (best)
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    settings.MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    device_map=settings.DEVICE,
                    attn_implementation='flash_attention_2'
                )
                logger.info("Model loaded with flash_attention_2")
            except Exception as e1:
                logger.warning(f"Failed with flash_attention_2: {e1}")
                try:
                    # Try SDPA (fallback)
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        settings.MODEL_PATH,
                        torch_dtype=torch.bfloat16,
                        device_map=settings.DEVICE,
                        attn_implementation='sdpa'
                    )
                    logger.info("Model loaded with SDPA")
                except Exception as e2:
                    logger.warning(f"Failed with SDPA: {e2}")
                    # Try without attention implementation specification
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        settings.MODEL_PATH,
                        torch_dtype=torch.bfloat16,
                        device_map=settings.DEVICE
                    )
                    logger.info("Model loaded with default attention")

            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)

            self.model_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Running in development mode without model")
            self.model_loaded = False

    def _load_voices(self):
        """Load available voice profiles."""
        # Create voices directory if it doesn't exist
        settings.VOICES_DIR.mkdir(exist_ok=True)

        # Add some default example voices if directory is empty
        voice_files = list(settings.VOICES_DIR.glob("*.wav")) + \
                      list(settings.VOICES_DIR.glob("*.mp3")) + \
                      list(settings.VOICES_DIR.glob("*.m4a"))

        if not voice_files:
            # Create a default voice profile for testing
            logger.info("No voice files found. Please upload or record voices.")
            # You can add code here to download sample voices if needed

        # Load existing voice files
        for voice_file in voice_files:
            voice_id = str(uuid.uuid4())
            profile = VoiceProfile(
                id=voice_id,
                name=voice_file.stem,
                type=VoiceType.PRESET,
                file_path=str(voice_file)
            )
            self.voices_cache[voice_id] = profile
            logger.info(f"Loaded voice: {voice_file.stem}")

    def generate_speech(
            self,
            text: str,
            voice_id: str,
            num_speakers: int = 1,
            cfg_scale: float = 1.3
    ) -> Optional[np.ndarray]:
        """Generate speech from text."""
        try:
            # Get voice profile
            voice_profile = self.voices_cache.get(voice_id)
            if not voice_profile:
                logger.error(f"Voice profile {voice_id} not found")
                raise ValueError(f"Voice profile {voice_id} not found")

            if not self.model_loaded or not self.model or not self.processor:
                logger.warning("Model not loaded, generating sample audio")
                return self._generate_sample_audio(text)

            logger.info(f"Generating speech with voice: {voice_profile.name}")

            # Format text for multi-speaker if needed
            formatted_text = self._format_text_for_speakers(text, num_speakers)

            # Prepare inputs
            inputs = self.processor(
                text=[formatted_text],
                voice_samples=[[voice_profile.file_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move to device
            if settings.DEVICE == "cuda" and torch.cuda.is_available():
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].cuda()

            logger.info("Starting generation...")

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
                audio_array = outputs.speech_outputs[0].cpu().numpy()
                logger.info(f"Generated audio shape: {audio_array.shape}")

                # Ensure audio is in the correct range
                audio_array = np.clip(audio_array, -1.0, 1.0)

                return audio_array
            else:
                logger.error("No speech output generated")
                return None

        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            logger.error(traceback.format_exc())
            # Return sample audio on error
            return self._generate_sample_audio(text)

    def _format_text_for_speakers(self, text: str, num_speakers: int) -> str:
        """Format text for multi-speaker generation."""
        if num_speakers == 1:
            # Single speaker - add Speaker 0 prefix if not present
            if not text.startswith("Speaker"):
                return f"Speaker 0: {text}"
            return text
        else:
            # Multi-speaker - ensure proper formatting
            lines = text.split('\n')
            formatted_lines = []
            current_speaker = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if line already has speaker format
                if line.startswith("Speaker"):
                    formatted_lines.append(line)
                else:
                    # Assign to speakers in rotation
                    formatted_lines.append(f"Speaker {current_speaker}: {line}")
                    current_speaker = (current_speaker + 1) % num_speakers

            return '\n'.join(formatted_lines)

    def _generate_sample_audio(self, text: str) -> np.ndarray:
        """Generate sample audio for testing when model is not available."""
        # Generate a more natural sounding placeholder audio
        duration = min(10.0, len(text) * 0.05)  # Approximate duration based on text length
        sample_rate = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create a more complex waveform (combination of frequencies)
        frequencies = [220, 440, 880]  # A3, A4, A5 notes
        audio = np.zeros_like(t)

        for i, freq in enumerate(frequencies):
            # Add harmonics with decreasing amplitude
            amplitude = 0.3 / (i + 1)
            audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Add envelope to make it sound more natural
        envelope = np.exp(-t * 0.5)  # Exponential decay
        audio = audio * envelope

        # Add some noise for texture
        noise = np.random.normal(0, 0.01, len(t))
        audio += noise

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8

        return audio.astype(np.float32)

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
        logger.info(f"Added voice profile: {name} (type: {voice_type})")
        return profile

    def get_voice_profiles(self) -> List[VoiceProfile]:
        """Get all available voice profiles."""
        return list(self.voices_cache.values())

    def get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get a specific voice profile."""
        return self.voices_cache.get(voice_id)

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loaded
