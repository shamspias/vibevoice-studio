"""Audio processing service."""

import os
import base64
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple
import logging
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


class AudioService:
    """Service for audio processing operations."""

    @staticmethod
    def save_audio(
            audio_data: np.ndarray,
            filename: Optional[str] = None,
            output_dir: Optional[Path] = None,
            sample_rate: int = None
    ) -> str:
        """Save audio data to file."""
        if sample_rate is None:
            sample_rate = settings.SAMPLE_RATE

        if output_dir is None:
            output_dir = settings.OUTPUTS_DIR

        if filename is None:
            filename = f"audio_{uuid.uuid4().hex[:8]}.wav"

        filepath = output_dir / filename

        try:
            sf.write(str(filepath), audio_data, sample_rate)
            logger.info(f"Audio saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    @staticmethod
    def load_audio(filepath: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            if target_sr is None:
                target_sr = settings.SAMPLE_RATE

            audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    @staticmethod
    def base64_to_audio(
            base64_data: str,
            output_path: Optional[str] = None
    ) -> str:
        """Convert base64 audio data to file."""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_data)

            # Generate output path if not provided
            if output_path is None:
                output_path = settings.UPLOADS_DIR / f"recording_{uuid.uuid4().hex[:8]}.wav"

            # Save to file
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)

            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to convert base64 to audio: {e}")
            raise

    @staticmethod
    def audio_to_base64(filepath: str) -> str:
        """Convert audio file to base64."""
        try:
            with open(filepath, 'rb') as f:
                audio_bytes = f.read()
            return base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to convert audio to base64: {e}")
            raise

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -25) -> np.ndarray:
        """Normalize audio to target dB."""
        rms = np.sqrt(np.mean(audio ** 2))
        scalar = 10 ** (target_db / 20) / (rms + 1e-6)
        normalized = audio * scalar

        # Avoid clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val

        return normalized
