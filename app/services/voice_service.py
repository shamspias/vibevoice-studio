"""Voice synthesis service using VibeVoice."""

import logging
import platform
import uuid
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch

from app.config import settings
from app.models import VoiceProfile, VoiceType

logger = logging.getLogger(__name__)


def _pick_device_map() -> str:
    if settings.DEVICE and settings.DEVICE.lower() != "auto":
        return settings.DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype():
    v = (settings.TORCH_DTYPE or "auto").lower()
    if v == "float32": return torch.float32
    if v == "float16": return torch.float16
    if v == "bfloat16": return torch.bfloat16

    # auto
    if torch.cuda.is_available():
        # bfloat16 preferred on recent NVIDIA + PyTorch 2
        return torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32  # CPU


def _pick_attn_impl() -> str:
    v = (settings.ATTN_IMPL or "auto").lower()
    if v != "auto":
        return v

    # Prefer SDPA on Torch 2.x (works on CPU/CUDA/MPS)
    if torch.cuda.is_available():
        # Try to use FlashAttention2 only if actually installed
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "sdpa"
    return "eager"  # CPU fallback


class VoiceService:
    """Service for voice synthesis operations."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.voices_cache: Dict[str, VoiceProfile] = {}
        self._model_ready = False
        self._load_voices()

    def ensure_model(self):
        if self._model_ready:
            return
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            device_map = _pick_device_map()
            torch_dtype = _pick_dtype()
            attn_impl = _pick_attn_impl()

            logger.info(
                f"Loading model '{settings.MODEL_PATH}' "
                f"(device_map={device_map}, dtype={torch_dtype}, attn={attn_impl})"
            )

            self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                settings.MODEL_PATH,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl
            )
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)

            self._model_ready = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Using mock mode (no real TTS).")
            self.model = None
            self.processor = None
            self._model_ready = False

    def _load_voices(self):
        from app.config import settings
        for voice_file in settings.VOICES_DIR.glob("*.wav"):
            voice_id = str(uuid.uuid4())
            self.voices_cache[voice_id] = VoiceProfile(
                id=voice_id,
                name=voice_file.stem,
                type=VoiceType.PRESET,
                file_path=str(voice_file)
            )

    def generate_speech(self, text: str, voice_id: str, num_speakers: int = 1, cfg_scale: float = 1.3) -> Optional[
        np.ndarray]:
        try:
            self.ensure_model()
            if not self.model or not self.processor:
                logger.warning("Using mock audio generation")
                return self._generate_mock_audio()

            voice_profile = self.voices_cache.get(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice profile {voice_id} not found")

            inputs = self.processor(
                text=[text],
                voice_samples=[[voice_profile.file_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

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
        from app.config import settings
        duration = 3.0
        sample_rate = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration))
        return 0.5 * np.sin(2 * np.pi * 440 * t)

    def add_voice_profile(self, name: str, audio_path: str, voice_type: VoiceType = VoiceType.UPLOADED) -> VoiceProfile:
        voice_id = str(uuid.uuid4())
        profile = VoiceProfile(id=voice_id, name=name, type=voice_type, file_path=audio_path)
        self.voices_cache[voice_id] = profile
        return profile

    def get_voice_profiles(self) -> List[VoiceProfile]:
        return list(self.voices_cache.values())

    def get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        return self.voices_cache.get(voice_id)
