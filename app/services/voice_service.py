"""Voice synthesis service using VibeVoice (fixed for dtype/device issues)."""

import os

# Silence the fork/parallelism warning from HF tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
from pathlib import Path
from typing import Optional, List, Dict
import uuid
import numpy as np

import torch

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
        """Initialize the VibeVoice model and processor with safe dtype/device."""
        try:
            try:
                from vibevoice.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference,
                )
                from vibevoice.processor.vibevoice_processor import (
                    VibeVoiceProcessor,
                )
            except ImportError:
                logger.error(
                    "VibeVoice not installed. Install with:\n"
                    "  git clone https://github.com/microsoft/VibeVoice.git\n"
                    "  cd VibeVoice && pip install -e ."
                )
                return

            # Decide device
            use_cuda = settings.DEVICE == "cuda" and torch.cuda.is_available()
            use_mps = (
                    settings.DEVICE == "mps"
                    and getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
            )
            torch_device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")

            # IMPORTANT:
            # - CUDA can benefit from float16/bfloat16 (depending on kernels).
            # - CPU/MPS must use float32 to avoid NumPy conversion errors (bf16 unsupported).
            dtype = torch.float16 if use_cuda else torch.float32

            logger.info(
                f"Loading model from {settings.MODEL_PATH} on device={torch_device} dtype={dtype}"
            )

            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)

            # Load model (try with flash-attn on CUDA only; then fallback)
            load_kwargs = {"torch_dtype": dtype}
            if use_cuda:
                load_kwargs["attn_implementation"] = "flash_attention_2"

            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    settings.MODEL_PATH,
                    **load_kwargs,
                )
            except Exception as e1:
                logger.warning(
                    f"Primary load failed ({e1}). Retrying without attn_implementation."
                )
                load_kwargs.pop("attn_implementation", None)
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    settings.MODEL_PATH,
                    **load_kwargs,
                )

            # Move model and set eval/inference params
            self.model.to(torch_device)
            self.model.eval()
            try:
                # if available on that class
                self.model.set_ddpm_inference_steps(num_steps=10)
            except Exception:
                pass

            self.model_loaded = True
            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.model_loaded = False

    def _load_voices(self):
        """Load available voice profiles from the voices directory."""
        settings.VOICES_DIR.mkdir(exist_ok=True)

        voice_files = []
        for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"):
            voice_files.extend(settings.VOICES_DIR.glob(ext))

        if not voice_files:
            logger.info(
                "No voice files found in voices/ yet. Upload or record to add voices."
            )

        for voice_file in voice_files:
            voice_id = str(uuid.uuid4())
            profile = VoiceProfile(
                id=voice_id,
                name=voice_file.stem,
                type=VoiceType.PRESET,
                file_path=str(voice_file),
            )
            self.voices_cache[voice_id] = profile
            logger.info(f"Loaded voice: {voice_file.stem}")

    def generate_speech(
            self,
            text: str,
            voice_id: str,
            num_speakers: int = 1,
            cfg_scale: float = 1.3,
    ) -> Optional[np.ndarray]:
        """Generate speech from text. Always returns float32 NumPy audio when successful."""
        try:
            # Validate voice
            voice_profile = self.voices_cache.get(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice profile {voice_id} not found")

            # If model unavailable, return placeholder audio
            if not (self.model_loaded and self.model and self.processor):
                logger.warning("Model not loaded — returning sample placeholder audio.")
                return self._generate_sample_audio(text)

            logger.info(f"Generating speech with voice: {voice_profile.name}")

            # Format text for multi-speaker scenarios
            formatted_text = self._format_text_for_speakers(text, num_speakers)

            # Prepare inputs
            inputs = self.processor(
                text=[formatted_text],
                voice_samples=[[voice_profile.file_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move inputs to same device as model
            model_device = next(self.model.parameters()).device
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model_device)

            logger.info("Starting generation...")

            # Generate audio
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )

            # Extract audio
            if getattr(outputs, "speech_outputs", None) and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]

                # Cast to float32 so NumPy can convert (bf16/fp16 -> float32)
                if audio_tensor.dtype != torch.float32:
                    audio_tensor = audio_tensor.to(torch.float32)

                # Move to CPU and convert to NumPy
                audio_array = audio_tensor.detach().cpu().numpy()

                # Safety clamp
                audio_array = np.clip(audio_array, -1.0, 1.0)

                return audio_array

            logger.error("No speech output generated by the model.")
            return None

        except Exception as e:
            logger.error(f"Speech generation error: {e}", exc_info=True)
            # Return a nicer sample audio rather than crashing
            return self._generate_sample_audio(text)

    def _format_text_for_speakers(self, text: str, num_speakers: int) -> str:
        """Ensure text has 'Speaker i:' prefixes when multiple speakers are requested."""
        if num_speakers <= 1:
            if not text.strip().startswith("Speaker"):
                return f"Speaker 0: {text}"
            return text

        lines = [ln.strip() for ln in text.splitlines()]
        formatted = []
        current = 0
        for ln in lines:
            if not ln:
                continue
            if ln.startswith("Speaker"):
                formatted.append(ln)
            else:
                formatted.append(f"Speaker {current}: {ln}")
                current = (current + 1) % num_speakers
        return "\n".join(formatted)

    def _generate_sample_audio(self, text: str) -> np.ndarray:
        """Generate a more pleasant placeholder tone (float32)."""
        duration = float(min(10.0, max(1.0, len(text) * 0.05)))  # seconds
        sr = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        freqs = [220.0, 440.0, 660.0]
        audio = sum(0.25 / (i + 1) * np.sin(2 * np.pi * f * t) for i, f in enumerate(freqs))

        # Simple envelope
        env = np.minimum(1.0, np.linspace(0, 1.0, len(t)) * 3.0) * np.exp(-t * 0.6)
        audio = (audio * env).astype(np.float32)

        # Normalize
        peak = float(np.max(np.abs(audio))) if audio.size else 1.0
        if peak > 0:
            audio = 0.8 * (audio / peak)
        return audio

    def add_voice_profile(
            self,
            name: str,
            audio_path: str,
            voice_type: VoiceType = VoiceType.UPLOADED,
    ) -> VoiceProfile:
        """Add a new voice profile to the in-memory cache."""
        voice_id = str(uuid.uuid4())
        profile = VoiceProfile(
            id=voice_id,
            name=name,
            type=voice_type,
            file_path=audio_path,
        )
        self.voices_cache[voice_id] = profile
        logger.info(f"Added voice profile: {name} (type: {voice_type})")
        return profile

    def get_voice_profiles(self) -> List[VoiceProfile]:
        """Return all available voice profiles."""
        return list(self.voices_cache.values())

    def get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Return a specific voice profile by id."""
        return self.voices_cache.get(voice_id)

    def is_model_loaded(self) -> bool:
        """Return True if model is loaded."""
        return self.model_loaded
