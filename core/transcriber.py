import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from .utils import get_ffmpeg_exe, ensure_dir


def extract_audio(video_path: str, output_audio_path: str) -> str:
    """
    Extracts a 16kHz mono WAV audio file from a video using FFmpeg.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(str(Path(output_audio_path).parent))
    
    cmd = [
        ffmpeg_exe,
        "-y",  # Overwrite
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_audio_path)
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
        
    return output_audio_path


class AudioTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the Whisper transcriber.
        model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            # Load model (auto-downloads to huggingface cache)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self,
        video_path: str,
        temp_dir: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribes the video file and returns segment-level timestamped transcripts.
        """
        if progress_callback:
            progress_callback("Extracting audio from video...")
            
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        extract_audio(video_path, audio_path)

        if progress_callback:
            progress_callback(f"Transcribing speech with Whisper ({self.model_size})...")

        model = self._get_model()
        segments_generator, info = model.transcribe(
            audio_path,
            beam_size=5,
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
    Cloud fallback using Groq Whisper (ultra-fast) or OpenAI Whisper API.
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
