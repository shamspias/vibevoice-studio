# 🎙️ VibeVoice Studio

A beautiful, modern web application for AI-powered voice synthesis using Microsoft's VibeVoice model. Generate natural-sounding speech from text with custom voice profiles.

![VibeVoice Studio](https://img.shields.io/badge/VibeVoice-Studio-purple?style=for-the-badge&logo=microphone)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green?style=for-the-badge&logo=fastapi)

## ✨ Features

- 🎤 **Voice Training**: Upload audio files or record your voice directly  
- 📝 **Text-to-Speech**: Convert text or text files to natural speech  
- 🎭 **Multiple Speakers**: Support for up to 4 distinct speakers  
- 💾 **Voice Library**: Save and manage custom voice profiles  
- 🎨 **Beautiful UI**: Modern, responsive design with dark/light themes  
- ⚡ **Real-time Processing**: Fast speech generation with streaming support  
- 📊 **Audio Visualization**: Live waveform display during recording  
- 💾 **Download & Save**: Export generated audio files  

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher  
- CUDA-capable GPU (recommended)  
- 8GB+ RAM  

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shamspias/vibevoice-studio.git
cd vibevoice-studio
````

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install VibeVoice**

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
cd ..
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your settings
```

6. **Run the application**

```bash
uvicorn app.main:app --reload
```

7. **Open in browser**

```
http://localhost:8000
```

## 🎨 Features Overview

### Voice Management

* Upload or record voices
* Support for WAV, MP3, M4A, FLAC
* Organized voice library

### Text Processing

* Manual input or upload `.txt` files
* Multi-speaker support for conversations

### Generation Settings

* Voice strength (CFG scale 1.0–2.0)
* Up to 4 speakers
* Adjustable inference steps

### Output Options

* Play in browser
* Download WAV file
* Save to library

## 🔧 Configuration

Edit `.env`:

```env
HOST=0.0.0.0
PORT=8000
DEBUG=False
MODEL_PATH=microsoft/VibeVoice-1.5B
DEVICE=cuda
CFG_SCALE=1.3
SAMPLE_RATE=24000
```

## 🎯 Usage Examples

### Basic TTS

1. Select/upload a voice
2. Enter text
3. Click "Generate Speech"

### Multi-Speaker

```text
Speaker 1: Hello, welcome!
Speaker 2: Thanks, glad to be here.
```

### Voice Cloning

1. Record 10–30s of clear speech
2. Save with name
3. Use for TTS generation

## 🛠️ API Documentation

### Endpoints

* `GET /api/voices` — list voices
* `POST /api/voices/upload` — upload voice
* `POST /api/voices/record` — record voice
* `POST /api/generate` — generate speech
* `GET /api/audio/{filename}` — download audio

## 🚦 System Requirements

**Minimum**: Python 3.9+, 8GB RAM, CPU with AVX
**Recommended**: Python 3.10+, 16GB RAM, NVIDIA GPU (8GB+ VRAM)

## 🐛 Troubleshooting

* **OOM**: Use smaller model, reduce batch size
* **Low quality**: Use better voice samples, adjust CFG scale
* **Slow generation**: Enable GPU, shorten text

## 📈 Performance Tips

* Use GPU for 10–20× speed
* Batch texts
* Cache voices
* Try quantized models

## 🤝 Contributing

1. Fork repo
2. Create feature branch
3. Commit & push
4. Open PR

## 📄 License

MIT License

## 🙏 Acknowledgments

* Microsoft VibeVoice team
* FastAPI community
* Contributors & users

## 📞 Support

* Issues: [GitHub Issues](https://github.com/shamspias/vibevoice-app/issues)

## 🔗 Links

* [VibeVoice Model](https://github.com/microsoft/VibeVoice)
* [FastAPI Docs](https://fastapi.tiangolo.com)
