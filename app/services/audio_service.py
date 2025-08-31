"""Audio processing service (robust recording save & format conversion)."""

import os
import re
import base64
import shutil
import json
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple, List
import logging
import uuid
import subprocess
from datetime import datetime

from app.config import settings
from app.models import AudioFile

logger = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _safe_b64decode(data: str) -> bytes:
    # Make sure padding length is a multiple of 4
    pad = (-len(data)) % 4
    if pad:
        data = data + ("=" * pad)
    return base64.b64decode(data)


def sanitize_filename(name: str) -> str:
    # Keep alnum, dash, underscore; collapse spaces; trim
    name = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^A-Za-z0-9_\-]+", "", name) or f"voice_{uuid.uuid4().hex[:8]}"


class AudioService:
    """Service for audio processing operations."""

    @staticmethod
    def save_audio(
        audio_data: np.ndarray,
        filename: Optional[str] = None,
        output_dir: Optional[Path] = None,
        sample_rate: int = None,
    ) -> str:
        """Save a mono float32 NumPy array to WAV."""
        if sample_rate is None:
            sample_rate = settings.SAMPLE_RATE

        if output_dir is None:
            output_dir = settings.OUTPUTS_DIR

        if filename is None:
            filename = f"audio_{uuid.uuid4().hex[:8]}.wav"

        filepath = output_dir / filename
        _ensure_dir(filepath)

        try:
            # Ensure audio is 1D float32 in [-1,1]
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()
            audio_data = np.clip(audio_data, -1.0, 1.0)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            sf.write(str(filepath), audio_data, sample_rate)
            logger.info(f"Audio saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    @staticmethod
    def save_audio_metadata(
        filename: str, voice_name: str, duration: float, text_preview: str
    ):
        """Save metadata for generated audio file."""
        try:
            filepath = settings.OUTPUTS_DIR / filename
            metadata_file = filepath.with_suffix(".json")

            metadata = {
                "filename": filename,
                "voice_name": voice_name,
                "duration": duration,
                "text_preview": text_preview[:100],
                "created_at": datetime.now().isoformat(),
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    @staticmethod
    def get_audio_library(search: Optional[str] = None) -> List[AudioFile]:
        """Get all generated audio files with metadata."""
        audio_files = []

        try:
            # Get all wav files in outputs directory
            for wav_file in settings.OUTPUTS_DIR.glob("*.wav"):
                # Try to load metadata
                metadata_file = wav_file.with_suffix(".json")

                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    audio_file = AudioFile(
                        filename=metadata["filename"],
                        voice_name=metadata["voice_name"],
                        duration=metadata["duration"],
                        size=wav_file.stat().st_size,
                        text_preview=metadata["text_preview"],
                        created_at=datetime.fromisoformat(metadata["created_at"]),
                    )
                else:
                    # Create basic metadata from file info
                    audio_file = AudioFile(
                        filename=wav_file.name,
                        voice_name="Unknown",
                        duration=0.0,
                        size=wav_file.stat().st_size,
                        text_preview="",
                        created_at=datetime.fromtimestamp(wav_file.stat().st_mtime),
                    )

                # Apply search filter if provided
                if search:
                    search_lower = search.lower()
                    if (
                        search_lower not in audio_file.filename.lower()
                        and search_lower not in audio_file.voice_name.lower()
                        and search_lower not in audio_file.text_preview.lower()
                    ):
                        continue

                audio_files.append(audio_file)

            # Sort by creation date (newest first)
            audio_files.sort(key=lambda x: x.created_at, reverse=True)

        except Exception as e:
            logger.error(f"Failed to get audio library: {e}")

        return audio_files

    @staticmethod
    def load_audio(
        filepath: str, target_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Load audio to mono float32 with librosa."""
        try:
            if target_sr is None:
                target_sr = settings.SAMPLE_RATE
            audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
            return audio.astype(np.float32), sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    @staticmethod
    def base64_to_audio(
        base64_data: str, output_path: Optional[Path] = None, format: str = "wav"
    ) -> str:
        """
        Convert base64-encoded audio to a WAV file.

        - If `format != 'wav'`, we first write the raw
          bytes into a temp file in `uploads/`
          with the correct extension, then convert that to WAV at `output_path`.
        - If `output_path` is not provided, we write the final WAV into `uploads/`.
        """
        try:
            fmt = (format or "wav").lower().split(";")[0]
            if "/" in fmt:
                # sometimes a MIME like 'audio/webm'
                fmt = fmt.split("/")[-1]

            # Choose final WAV destination
            if output_path is None:
                final_wav = (
                    settings.UPLOADS_DIR / f"recording_{uuid.uuid4().hex[:8]}.wav"
                )
            else:
                final_wav = Path(output_path)
                if final_wav.suffix.lower() != ".wav":
                    final_wav = final_wav.with_suffix(".wav")
            _ensure_dir(final_wav)

            # If the incoming is already WAV, write directly
            if fmt == "wav":
                raw = _safe_b64decode(base64_data)
                with open(final_wav, "wb") as f:
                    f.write(raw)
                return str(final_wav)

            # Otherwise, write to temp with ORIGINAL extension, then convert to WAV
            tmp_src = settings.UPLOADS_DIR / f"tmp_{uuid.uuid4().hex[:8]}.{fmt}"
            _ensure_dir(tmp_src)
            raw = _safe_b64decode(base64_data)
            with open(tmp_src, "wb") as f:
                f.write(raw)

            # Convert temp -> final WAV
            AudioService.convert_to_wav(str(tmp_src), str(final_wav))
            try:
                tmp_src.unlink(missing_ok=True)
            except Exception:
                pass

            return str(final_wav)

        except Exception as e:
            logger.error(f"Failed to convert base64 to audio: {e}")
            raise

    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> bool:
        """
        Convert an arbitrary audio file to mono WAV @ SAMPLE_RATE.
        Uses (1) soundfile if possible, (2) ffmpeg, (3) librosa as last resort.
        Ensures input and output are NOT the same temp path.
        """
        try:
            in_path = os.path.abspath(input_path)
            out_path = os.path.abspath(output_path)
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            # If same path, use a temporary target
            need_move = False
            if in_path == out_path:
                tmp_out = out_path + ".tmp.wav"
                out_target = tmp_out
                need_move = True
            else:
                out_target = out_path

            # Try direct read/write with soundfile
            try:
                data, sr = sf.read(in_path, always_2d=False)
                if data.ndim > 1:
                    # Mixdown to mono
                    data = np.mean(data, axis=-1)
                data = AudioService._resample_if_needed(data, sr, settings.SAMPLE_RATE)
                data = data.astype(np.float32)
                sf.write(out_target, data, settings.SAMPLE_RATE)
                if need_move:
                    shutil.move(out_target, out_path)
                return True
            except Exception as e1:
                logger.warning(f"Soundfile conversion failed: {e1}, trying ffmpeg")

            # Try ffmpeg
            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    in_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(settings.SAMPLE_RATE),
                    "-ac",
                    "1",
                    out_target,
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                if need_move:
                    shutil.move(out_target, out_path)
                return True
            except Exception as e2:
                logger.error(f"FFmpeg conversion also failed: {e2}")

            # Last resort: librosa
            try:
                audio, sr = librosa.load(in_path, sr=settings.SAMPLE_RATE, mono=True)
                audio = audio.astype(np.float32)
                sf.write(out_target, audio, settings.SAMPLE_RATE)
                if need_move:
                    shutil.move(out_target, out_path)
                return True
            except Exception as e3:
                logger.error(f"All conversion methods failed: {e3}")
                raise

        except Exception as e:
            logger.error(f"convert_to_wav fatal error: {e}")
            raise

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return audio
        # Use librosa for high-quality resample
        return librosa.resample(
            audio.astype(np.float32), orig_sr=sr_in, target_sr=sr_out
        )

    @staticmethod
    def audio_to_base64(filepath: str) -> str:
        """Convert audio file to base64."""
        try:
            with open(filepath, "rb") as f:
                audio_bytes = f.read()
            return base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to convert audio to base64: {e}")
            raise

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -20) -> np.ndarray:
        """Normalize audio to target dB."""
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            scalar = target_rms / rms
            normalized = audio * scalar
            max_val = float(np.max(np.abs(normalized))) if normalized.size else 0.0
            if max_val > 0.95:
                normalized = normalized * (0.95 / max_val)
            return normalized.astype(np.float32)
        return audio.astype(np.float32)
