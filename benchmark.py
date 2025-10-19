"""
ENHANCED VIBEVOICE BENCHMARK WITH QUALITY METRICS
Measures performance AND quality (WER, speaker similarity) for both:
- VibeVoice core model (https://github.com/shamspias/VibeVoice)
- VibeVoice Studio application (https://github.com/shamspias/vibevoice-studio)

Author: Shamsuddin Ahmed
Project: VibeVoice Studio - Consent-First AI Voice Synthesis
"""

import time
import torch
import psutil
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore')


class EnhancedBenchmark:
    """Enhanced benchmark: performance + quality metrics + deployment analysis"""

    def __init__(self, model_path="microsoft/VibeVoice-1.5B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = None
        self.results = []

        # Quality measurement tools (optional)
        self.whisper_model = None
        self.speaker_encoder = None
        self.quality_available = False

        # Detect implementation type
        self.is_studio = self._detect_studio_environment()

    def _detect_studio_environment(self):
        """Detect if running in VibeVoice Studio environment"""
        # Check if studio app structure exists
        if Path("app").exists() and Path("app/services").exists():
            return True
        return False

    def setup_model(self):
        """Load VibeVoice model (works for both standalone and studio)"""
        print("=" * 70)
        print("SETUP: Loading VibeVoice Model")
        print(f"Model path: {self.model_path}")
        print(f"Environment: {'VibeVoice Studio' if self.is_studio else 'Standalone VibeVoice'}")
        print(f"Core TTS: https://github.com/shamspias/VibeVoice")
        if self.is_studio:
            print(f"Studio App: https://github.com/shamspias/vibevoice-studio")
        print("=" * 70)

        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import (
                VibeVoiceProcessor
            )
        except ImportError as e:
            print(f"ERROR: VibeVoice not installed: {e}")
            print("\nInstall from your fork:")
            print("  git clone https://github.com/shamspias/VibeVoice")
            print("  cd VibeVoice && pip install -e .")
            sys.exit(1)

        # Detect hardware
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU Detected: {gpu_name}")
            print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
            dtype = torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            print(f"✓ Apple Silicon (MPS) Detected")
            dtype = torch.float32
        else:
            self.device = "cpu"
            print(f"✓ CPU Mode")
            dtype = torch.float32

        # Load processor
        print(f"\nLoading processor from {self.model_path}...")
        load_start = time.time()
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        print(f"✓ Processor loaded in {time.time() - load_start:.2f}s")

        # Load model
        print(f"Loading model...")
        load_start = time.time()

        try:
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=self.device if self.device == "cuda" else "cpu"
            )
        except Exception as e:
            print(f"Loading with device_map failed, trying without: {e}")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=dtype
            )
            self.model.to(self.device)

        self.model.eval()
        print(f"✓ Model loaded in {time.time() - load_start:.2f}s")
        print("=" * 70)

    def setup_quality_metrics(self):
        """Setup optional quality measurement tools"""
        print("\n" + "=" * 70)
        print("OPTIONAL: Setting up quality metrics")
        print("=" * 70)

        # Try to load Whisper for WER
        try:
            import whisper
            print("Loading Whisper for WER measurement...")
            self.whisper_model = whisper.load_model("base")
            print("✓ Whisper loaded (will measure WER)")
        except ImportError:
            print("⚠ Whisper not installed (WER will be skipped)")
            print("  Install: pip install openai-whisper")
        except Exception as e:
            print(f"⚠ Whisper load failed: {e}")

        # Try to load speaker encoder for similarity
        try:
            from resemblyzer import VoiceEncoder
            print("Loading Resemblyzer for speaker similarity...")
            self.speaker_encoder = VoiceEncoder()
            print("✓ Resemblyzer loaded (will measure similarity)")
        except ImportError:
            print("⚠ Resemblyzer not installed (similarity will be skipped)")
            print("  Install: pip install resemblyzer")
        except Exception as e:
            print(f"⚠ Resemblyzer load failed: {e}")

        self.quality_available = (
                self.whisper_model is not None or
                self.speaker_encoder is not None
        )

        if self.quality_available:
            print("✓ Quality metrics enabled")
        else:
            print("ℹ Quality metrics disabled (install libraries to enable)")

        print("=" * 70)

    def measure_wer(self, audio_path, reference_text):
        """Measure Word Error Rate using Whisper"""
        if self.whisper_model is None:
            return None

        try:
            result = self.whisper_model.transcribe(audio_path)
            transcribed_text = result["text"].strip().lower()

            # Remove "speaker X:" from reference text for fair comparison
            ref_clean = reference_text.lower()
            if "speaker" in ref_clean:
                # Extract just the text without speaker labels
                import re
                ref_clean = re.sub(r'speaker\s+\d+:\s*', '', ref_clean)

            # Calculate WER
            ref_words = ref_clean.strip().split()
            hyp_words = transcribed_text.split()

            wer = self._calculate_wer(ref_words, hyp_words)
            return wer
        except Exception as e:
            print(f"  ⚠ WER measurement failed: {e}")
            return None

    def _calculate_wer(self, ref, hyp):
        """Calculate Word Error Rate (Levenshtein distance)"""
        d = np.zeros((len(ref) + 1, len(hyp) + 1))

        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j

        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        wer = (d[len(ref)][len(hyp)] / len(ref)) * 100 if len(ref) > 0 else 0
        return round(wer, 2)

    def measure_speaker_similarity(self, original_audio, generated_audio):
        """Measure speaker similarity using voice encoder"""
        if self.speaker_encoder is None:
            return None

        try:
            import soundfile as sf

            # Load audio files
            orig_wav, sr1 = sf.read(original_audio)
            gen_wav, sr2 = sf.read(generated_audio)

            # Ensure mono
            if len(orig_wav.shape) > 1:
                orig_wav = np.mean(orig_wav, axis=1)
            if len(gen_wav.shape) > 1:
                gen_wav = np.mean(gen_wav, axis=1)

            # Get embeddings
            orig_embed = self.speaker_encoder.embed_utterance(orig_wav)
            gen_embed = self.speaker_encoder.embed_utterance(gen_wav)

            # Cosine similarity
            similarity = np.dot(orig_embed, gen_embed)

            return round(float(similarity), 3)
        except Exception as e:
            print(f"  ⚠ Similarity measurement failed: {e}")
            return None

    def _format_text_for_vibevoice(self, text, num_speakers=1):
        """Format plain text into VibeVoice's expected format with Speaker labels"""
        # If text already has Speaker labels, return as-is
        if "Speaker" in text and ":" in text:
            return text

        # Otherwise, format it properly
        formatted_lines = []

        if num_speakers == 1:
            # Single speaker - just add Speaker 0 label
            formatted_lines.append(f"Speaker 0: {text}")
        else:
            # Multi-speaker - split by sentences and alternate
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    speaker_id = i % num_speakers
                    formatted_lines.append(f"Speaker {speaker_id}: {sentence}")

        return "\n".join(formatted_lines)

    def measure_generation(self, text, voice_file, cfg_scale=1.3, num_speakers=1, num_runs=3):
        """Measure ACTUAL generation performance + quality"""
        print(f"\nBenchmarking: {len(text)} chars, {num_runs} runs, {num_speakers} speaker(s)")

        # Format text properly for VibeVoice
        formatted_text = self._format_text_for_vibevoice(text, num_speakers)

        results = []
        temp_audio_files = []

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)

            # Measure memory before
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_start = torch.cuda.memory_allocated() / 1e6
            else:
                process = psutil.Process()
                mem_start = process.memory_info().rss / 1e6

            try:
                # Prepare inputs
                inputs = self.processor(
                    text=[formatted_text],
                    voice_samples=[[voice_file]],
                    padding=True,
                    return_tensors="pt",
                )

                # Move to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)

                # TIME THE GENERATION
                start_time = time.time()

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                    )

                latency = time.time() - start_time

                # Measure memory after
                if self.device == "cuda":
                    mem_peak = torch.cuda.max_memory_allocated() / 1e6
                else:
                    mem_peak = process.memory_info().rss / 1e6

                memory_used = mem_peak - mem_start

                # Get audio output
                if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                    audio = outputs.speech_outputs[0]

                    # Handle MPS tensors
                    if self.device == "mps":
                        audio = audio.cpu()

                    audio_duration = len(audio) / 24000  # 24kHz sample rate

                    # Save audio for quality measurement
                    temp_audio_path = f"temp_audio_run{run}.wav"
                    self._save_audio(audio, temp_audio_path)
                    temp_audio_files.append(temp_audio_path)

                    run_result = {
                        "latency_sec": latency,
                        "audio_duration_sec": audio_duration,
                        "memory_used_mb": memory_used,
                        "memory_peak_mb": mem_peak,
                        "temp_audio": temp_audio_path,
                    }

                    results.append(run_result)

                    rtf = latency / audio_duration
                    print(f"✓ {latency:.2f}s (RTF: {rtf:.2f}x)")
                else:
                    print("✗ No audio generated")

            except Exception as e:
                print(f"✗ Failed: {e}")
                continue

        # Calculate statistics
        if results:
            avg_result = {
                "latency_sec": np.mean([r["latency_sec"] for r in results]),
                "audio_duration_sec": np.mean([r["audio_duration_sec"] for r in results]),
                "memory_peak_mb": np.mean([r["memory_peak_mb"] for r in results]),
                "std_latency": np.std([r["latency_sec"] for r in results]),
            }
            avg_result["real_time_factor"] = (
                    avg_result["latency_sec"] / avg_result["audio_duration_sec"]
            )
            avg_result["latency_per_min_audio"] = (
                    60 * avg_result["latency_sec"] / avg_result["audio_duration_sec"]
            )

            # Measure quality metrics (if available)
            if self.quality_available and temp_audio_files:
                print("  Measuring quality metrics...", end=" ", flush=True)

                # WER
                if self.whisper_model:
                    wers = []
                    for audio_file in temp_audio_files:
                        wer = self.measure_wer(audio_file, text)
                        if wer is not None:
                            wers.append(wer)
                    if wers:
                        avg_result["wer_percent"] = round(np.mean(wers), 2)
                        print(f"WER: {avg_result['wer_percent']}%", end=" ")

                # Speaker similarity
                if self.speaker_encoder:
                    sims = []
                    for audio_file in temp_audio_files:
                        sim = self.measure_speaker_similarity(voice_file, audio_file)
                        if sim is not None:
                            sims.append(sim)
                    if sims:
                        avg_result["speaker_similarity"] = round(np.mean(sims), 3)
                        print(f"SIM: {avg_result['speaker_similarity']}", end=" ")

                print("✓")

            # Cleanup temp files
            for temp_file in temp_audio_files:
                try:
                    Path(temp_file).unlink()
                except:
                    pass

            return avg_result
        else:
            return None

    def _save_audio(self, audio_array, filepath, sample_rate=24000):
        """Save audio array to WAV file"""
        try:
            import soundfile as sf

            # Convert to numpy if tensor
            if torch.is_tensor(audio_array):
                audio_array = audio_array.cpu().numpy()

            # Ensure audio is float32 and 1D
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()
            audio_array = np.clip(audio_array, -1.0, 1.0).astype(np.float32)

            sf.write(filepath, audio_array, sample_rate)
        except Exception as e:
            print(f"Warning: Could not save audio: {e}")

    def run_full_benchmark(self, voice_file):
        """Run complete benchmark suite"""
        if not Path(voice_file).exists():
            print(f"ERROR: Voice file not found: {voice_file}")
            print("Please provide a valid voice sample WAV file")
            sys.exit(1)

        self.setup_model()
        self.setup_quality_metrics()

        # Test cases: varying complexity - FORMATTED FOR VIBEVOICE
        test_cases = [
            ("Short", "Hello, this is a test of AI voice synthesis technology.", 1),
            ("Medium",
             "Artificial intelligence is rapidly transforming the creative economy through advanced voice synthesis capabilities. This technology enables content creators to produce high-quality audio at unprecedented scale and efficiency. However, it also raises important questions about creator rights, compensation, and ethical deployment.",
             1),
            ("Long",
             "This paper proposes a consent-first architecture for AI voice cloning tailored to the creative economy. We combine explicit rights management with provenance watermarking and attribution-based revenue models. The system ensures that voice talent receive fair compensation while enabling legitimate commercial applications. Our deployment analysis examines performance, cost, latency, and energy consumption trade-offs relevant to small creative studios. Results demonstrate that robust ethical safeguards can coexist with efficient generation and reduced computational budgets.",
             1),
        ]

        print("\n" + "=" * 70)
        print("BENCHMARK SUITE")
        print("=" * 70)

        all_results = []

        for name, text, num_speakers in test_cases:
            print(f"\n[{name}] Text length: {len(text)} chars")
            result = self.measure_generation(text, voice_file, cfg_scale=1.3, num_speakers=num_speakers, num_runs=3)

            if result:
                result["test_name"] = name
                result["text_length"] = len(text)
                all_results.append(result)

                print(f"  → Avg latency: {result['latency_sec']:.2f}s")
                print(f"  → Audio duration: {result['audio_duration_sec']:.2f}s")
                print(f"  → Real-time factor: {result['real_time_factor']:.2f}x")
                print(f"  → Latency per min: {result['latency_per_min_audio']:.2f}s")
                print(f"  → Memory peak: {result['memory_peak_mb']:.1f} MB")
                if "wer_percent" in result:
                    print(f"  → Word Error Rate: {result['wer_percent']}%")
                if "speaker_similarity" in result:
                    print(f"  → Speaker Similarity: {result['speaker_similarity']}")

        self.results = all_results
        return all_results

    def estimate_costs_and_energy(self):
        """Add cost and energy estimates based on hardware"""
        if not self.results:
            return

        gpu_name = ""
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
        elif self.device == "mps":
            gpu_name = "Apple Silicon (M1/M2/M3)"

        # Estimate based on hardware type
        if "H100" in gpu_name:
            power_watts = 700
            cost_per_hour = 3.50
            hardware_type = "Cloud H100 GPU"
        elif "A100" in gpu_name:
            power_watts = 400
            cost_per_hour = 2.00
            hardware_type = "Cloud A100 GPU"
        elif "4090" in gpu_name or "RTX 4090" in gpu_name:
            power_watts = 450
            cost_per_hour = 0.10
            hardware_type = "Edge RTX 4090"
        elif "4080" in gpu_name or "3090" in gpu_name or "RTX" in gpu_name:
            power_watts = 350
            cost_per_hour = 0.08
            hardware_type = f"Edge GPU ({gpu_name})"
        elif "Apple Silicon" in gpu_name or self.device == "mps":
            power_watts = 30
            cost_per_hour = 0.00
            hardware_type = "Apple Silicon (MPS)"
        else:
            power_watts = 65
            cost_per_hour = 0.05
            hardware_type = "CPU"

        for result in self.results:
            # Energy estimate
            latency_hours = result["latency_sec"] / 3600
            energy_kwh = (power_watts * latency_hours) / 1000

            # Cost estimate
            cost = latency_hours * cost_per_hour

            # Per minute of audio
            audio_mins = result["audio_duration_sec"] / 60
            result["energy_per_audio_min_kwh"] = energy_kwh / audio_mins if audio_mins > 0 else 0
            result["cost_per_audio_min_usd"] = cost / audio_mins if audio_mins > 0 else 0
            result["hardware_type"] = hardware_type
            result["estimated_power_watts"] = power_watts

        print("\n" + "=" * 70)
        print("COST & ENERGY ESTIMATES")
        print("=" * 70)
        print(f"Hardware: {hardware_type}")
        print(f"Estimated Power: {power_watts}W")
        print(f"Cost per hour: ${cost_per_hour:.2f}")
        print(f"\nNOTE: These are estimates. Use a power meter for exact measurements!")
        print("=" * 70)

    def save_results(self, output_file="benchmark_results.json"):
        """Save results to JSON"""

        self.estimate_costs_and_energy()

        output = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "model_path": self.model_path,
            "environment": "VibeVoice Studio" if self.is_studio else "Standalone VibeVoice",
            "repositories": {
                "core_tts": "https://github.com/shamspias/VibeVoice",
                "studio_app": "https://github.com/shamspias/vibevoice-studio" if self.is_studio else None
            },
            "quality_metrics_enabled": self.quality_available,
            "hardware_info": self._get_hardware_info(),
            "results": self.results,
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        # Generate summary for paper
        self._print_summary()

    def _get_hardware_info(self):
        """Collect hardware details"""
        info = {
            "device": self.device,
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "cpu_count": psutil.cpu_count(),
        }

        if self.device == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["cuda_version"] = torch.version.cuda
            info["pytorch_version"] = torch.__version__
        elif self.device == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"
            info["pytorch_version"] = torch.__version__

        return info

    def _print_summary(self):
        """Print summary table for paper"""
        if not self.results:
            return

        print("\n" + "=" * 70)
        print("SUMMARY FOR YOUR PAPER")
        print("=" * 70)

        # Average across all tests
        avg_latency_per_min = np.mean([r["latency_per_min_audio"] for r in self.results])
        avg_rtf = np.mean([r["real_time_factor"] for r in self.results])
        avg_energy = np.mean([r.get("energy_per_audio_min_kwh", 0) * 60 for r in self.results])
        avg_cost = np.mean([r.get("cost_per_audio_min_usd", 0) for r in self.results])

        # Quality metrics if available
        wers = [r.get("wer_percent") for r in self.results if "wer_percent" in r]
        sims = [r.get("speaker_similarity") for r in self.results if "speaker_similarity" in r]

        avg_wer = np.mean(wers) if wers else None
        avg_sim = np.mean(sims) if sims else None

        hardware = self.results[0].get("hardware_type", "Unknown")

        print(f"\n📊 Configuration: {hardware}")
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Average latency per minute of audio: {avg_latency_per_min:.2f}s")
        print(f"  Average real-time factor: {avg_rtf:.2f}x")
        print(f"  Estimated energy per hour: {avg_energy:.2f} kWh")
        print(f"  Estimated cost per minute: ${avg_cost:.3f}")

        if avg_wer is not None or avg_sim is not None:
            print(f"\nQUALITY METRICS:")
            if avg_wer is not None:
                print(f"  Average Word Error Rate: {avg_wer:.2f}%")
            if avg_sim is not None:
                print(f"  Average Speaker Similarity: {avg_sim:.3f}")

        print("\n" + "=" * 70)
        print("FOR YOUR PAPER TABLE 1:")
        print("=" * 70)
        print(f"Hardware: {hardware}")
        print(f"Latency: {avg_latency_per_min:.1f} seconds")
        print(f"Cost/min: ${avg_cost:.2f}")
        print(f"Energy: {avg_energy:.2f} kWh/hr")
        if avg_wer is not None:
            print(f"WER: {avg_wer:.1f}%")
        if avg_sim is not None:
            print(f"SIM: {avg_sim:.2f}")
        print("=" * 70)


def main():
    """
    Main benchmark script

    Usage:
        python enhanced_benchmark.py <voice_sample.wav> [--model-path MODEL_PATH]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced VibeVoice Benchmark")
    parser.add_argument("voice_file", type=str, help="Path to voice sample WAV file")
    parser.add_argument("--model-path", type=str, default="microsoft/VibeVoice-1.5B",
                        help="Model path (default: microsoft/VibeVoice-1.5B)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for results (default: benchmark_results.json)")

    args = parser.parse_args()

    print("=" * 70)
    print("ENHANCED VIBEVOICE PERFORMANCE + QUALITY BENCHMARK")
    print("Core TTS: https://github.com/shamspias/VibeVoice")
    print("Studio App: https://github.com/shamspias/vibevoice-studio")
    print("=" * 70)

    benchmark = EnhancedBenchmark(model_path=args.model_path)
    benchmark.run_full_benchmark(args.voice_file)
    benchmark.save_results(args.output)

    print("\n✅ Benchmark complete!")
    print("📄 Use the summary values above in your paper's Table 1")
    print(f"📁 Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
