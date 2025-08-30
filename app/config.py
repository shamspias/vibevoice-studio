"""Configuration module for VibeVoice application."""
from pathlib import Path
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App settings
    APP_NAME: str = "VibeVoice Studio"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model settings
    MODEL_PATH: str = "microsoft/VibeVoice-1.5B"
    DEVICE: str = "cuda"
    MAX_LENGTH: int = 1000
    CFG_SCALE: float = 1.3

    # Path settings
    BASE_DIR: Path = Path(__file__).parent.parent
    VOICES_DIR: Path = BASE_DIR / "voices"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"

    # Audio settings
    SAMPLE_RATE: int = 24000
    MAX_AUDIO_SIZE_MB: int = 50
    SUPPORTED_FORMATS: list = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.VOICES_DIR.mkdir(exist_ok=True)
        self.OUTPUTS_DIR.mkdir(exist_ok=True)
        self.UPLOADS_DIR.mkdir(exist_ok=True)


settings = Settings()
