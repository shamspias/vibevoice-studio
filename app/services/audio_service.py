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
import subprocess

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
            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()

            # Ensure audio is in correct range
            audio_data = np.clip(audio_data, -1.0, 1.0)

            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

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

            # Load with librosa for consistency
            audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    @staticmethod
    def base64_to_audio(
            base64_data: str,
            output_path: Optional[Path] = None,
            format: str = 'wav'
    ) -> str:
        """Convert base64 audio data to file."""
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_data)

            # Generate output path if not provided
            if output_path is None:
                output_path = settings.UPLOADS_DIR / f"recording_{uuid.uuid4().hex[:8]}.{format}"

            # Save to file
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)

            # Convert to WAV if needed (for consistency)
            if format != 'wav':
                wav_path = output_path.with_suffix('.wav')
                AudioService.convert_to_wav(str(output_path), str(wav_path))
                # Remove original and use WAV
                os.remove(output_path)
                output_path = wav_path

            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to convert base64 to audio: {e}")
            raise

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> bool:
        """Convert audio file to WAV format using ffmpeg or soundfile."""
        try:
            # Try using soundfile first
            data, sr = sf.read(input_path)
            sf.write(output_path, data, sr)
            return True
        except Exception as e1:
            logger.warning(f"Soundfile conversion failed: {e1}, trying ffmpeg")
            try:
                # Try ffmpeg as fallback
                cmd = [
                    'ffmpeg', '-i', input_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', str(settings.SAMPLE_RATE),
                    '-ac', '1',  # Mono
                    output_path,
                    '-y'  # Overwrite
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except Exception as e2:
                logger.error(f"FFmpeg conversion also failed: {e2}")
                # Last resort: use librosa
                try:
                    audio, sr = librosa.load(input_path, sr=settings.SAMPLE_RATE, mono=True)
                    sf.write(output_path, audio, sr)
                    return True
                except Exception as e3:
                    logger.error(f"All conversion methods failed: {e3}")
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
    def normalize_audio(audio: np.ndarray, target_db: float = -20) -> np.ndarray:
        """Normalize audio to target dB."""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > 0:
            # Calculate desired RMS for target dB
            target_rms = 10 ** (target_db / 20)

            # Calculate scaling factor
            scalar = target_rms / rms

            # Apply scaling
            normalized = audio * scalar

            # Prevent clipping
            max_val = np.max(np.abs(normalized))
            if max_val > 0.95:
                normalized = normalized * (0.95 / max_val)
        else:
            normalized = audio

        return normalized
