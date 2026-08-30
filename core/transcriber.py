import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from .utils import get_ffmpeg_exe, ensure_dir

# Global cache for loaded Whisper models to prevent re-instantiation latency
_WHISPER_MODELS: Dict[str, Any] = {}


def detect_whisper_device() -> tuple[str, str]:
    """
    Safely probes for functional CUDA support.
    Returns (device, compute_type) -> e.g. ('cuda', 'float16') or ('cpu', 'int8').
    """
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            compute = "float16" if "float16" in cuda_types else "int8"
            return "cuda", compute
    except Exception:
        pass
    return "cpu", "int8"


def extract_audio(video_path: str, output_audio_path: str) -> str:
    """
    Extracts a 16kHz mono WAV audio file from a video using FFmpeg with fast single-channel stream.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(str(Path(output_audio_path).parent))

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-threads", "0",
        str(output_audio_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    return output_audio_path


def extract_compressed_audio(video_path: str, output_audio_path: str) -> str:
    """
    Extracts a lightweight, compressed 48kbps mono MP3 audio file for fast cloud LLM upload (<5MB).
    """
    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(str(Path(output_audio_path).parent))

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "libmp3lame",
        "-b:a", "48k",
        "-ar", "16000",
        "-ac", "1",
        "-threads", "0",
        str(output_audio_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        # Fallback to standard wav if libmp3lame is not compiled into ffmpeg
        return extract_audio(video_path, output_audio_path.replace(".mp3", ".wav"))

    return output_audio_path



class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        cpu_threads: Optional[int] = None
    ):
        """
        Initializes the Whisper transcriber with model caching and device auto-detection.
        model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
        """
        self.model_size = model_size
        if device is None or compute_type is None:
            auto_dev, auto_compute = detect_whisper_device()
            self.device = device or auto_dev
            self.compute_type = compute_type or auto_compute
        else:
            self.device = device
            self.compute_type = compute_type
            
        self.cpu_threads = cpu_threads or min(4, os.cpu_count() or 4)

    def _get_model(self):
        cache_key = f"{self.model_size}_{self.device}_{self.compute_type}_{self.cpu_threads}"
        global _WHISPER_MODELS
        if cache_key not in _WHISPER_MODELS:
            from faster_whisper import WhisperModel
            try:
                _WHISPER_MODELS[cache_key] = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                )
            except Exception as e:
                # Fallback to CPU int8 if CUDA or specific compute type fails
                if self.device != "cpu":
                    self.device = "cpu"
                    self.compute_type = "int8"
                    cpu_key = f"{self.model_size}_cpu_int8_{self.cpu_threads}"
                    if cpu_key not in _WHISPER_MODELS:
                        _WHISPER_MODELS[cpu_key] = WhisperModel(
                            self.model_size,
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=self.cpu_threads,
                        )
                    _WHISPER_MODELS[cache_key] = _WHISPER_MODELS[cpu_key]
                else:
                    raise e
        return _WHISPER_MODELS[cache_key]

    def transcribe(
        self,
        video_path: str,
        temp_dir: str,
        beam_size: int = 1,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribes the video file and returns segment-level timestamped transcripts.
        Uses beam_size=1 (greedy) by default for 3x-5x faster CPU transcription.
        """
        if progress_callback:
            progress_callback("Extracting audio from video...")

        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        extract_audio(video_path, audio_path)

        if progress_callback:
            progress_callback(f"Transcribing speech with Whisper ({self.model_size}, {self.device})...")

        model = self._get_model()
        segments_generator, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            best_of=1 if beam_size == 1 else 5,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        segments = []
        for segment in segments_generator:
            text = segment.text.strip()
            if text:
                segments.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                })

        return segments


def transcribe_with_groq_or_openai(
    audio_path: str,
    api_key: str,
    provider: str = "groq"
) -> List[Dict[str, Any]]:
    """
    Cloud fallback using Groq Whisper (ultra-fast, ~1-2s) or OpenAI Whisper API.
    """
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        segments = []
        for seg in getattr(transcription, "segments", []):
            segments.append({
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "").strip()
            })
        return segments
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-1",
                response_format="verbose_json"
            )
        segments = []
        for seg in getattr(transcription, "segments", []):
            segments.append({
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "").strip()
            })
        return segments

    raise ValueError(f"Unsupported cloud transcriber provider: {provider}")

