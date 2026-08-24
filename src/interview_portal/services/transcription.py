"""Whisper.cpp speech-to-text transcription service."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import time
from typing import Any

import imageio_ffmpeg
from pywhispercpp.model import Model

_MODELS_CACHE: dict[str, Model] = {}


def get_whisper_model(model_name: str = "base.en", n_threads: int = 4) -> Model:
    """Retrieve or load and cache a whisper.cpp Model instance."""
    if model_name not in _MODELS_CACHE:
        _MODELS_CACHE[model_name] = Model(model_name, n_threads=n_threads)
    return _MODELS_CACHE[model_name]


def convert_audio_to_wav(audio_input: bytes | io.BytesIO | str) -> str:
    """
    Convert any input audio (WAV, MP3, M4A, OGG, AAC, WebM, FLAC) to a
    16kHz 16-bit mono WAV file required by whisper.cpp.
    Returns the path to the converted temporary WAV file.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # If audio_input is bytes or BytesIO, write to a temporary source file first
    temp_src = None
    if isinstance(audio_input, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw_audio") as src_file:
            src_file.write(audio_input)
            temp_src = src_file.name
        input_path = temp_src
    elif hasattr(audio_input, "read") and hasattr(audio_input, "seek"):
        # e.g. BytesIO or Streamlit UploadedFile
        audio_input.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw_audio") as src_file:
            src_file.write(audio_input.read())
            temp_src = src_file.name
        input_path = temp_src
    elif isinstance(audio_input, str) and os.path.exists(audio_input):
        input_path = audio_input
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    # Destination 16kHz mono WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst_file:
        output_wav_path = dst_file.name

    try:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return output_wav_path
    finally:
        if temp_src and os.path.exists(temp_src):
            try:
                os.remove(temp_src)
            except OSError:
                pass


def transcribe_audio(
    audio_input: bytes | io.BytesIO | str,
    model_name: str = "base.en",
    n_threads: int = 4,
) -> dict[str, Any]:
    """
    Transcribe audio data using whisper.cpp.

    Returns a dictionary:
    {
        "text": str,
        "segments": list[dict],
        "duration_sec": float,
        "model_name": str,
    }
    """
    start_time = time.perf_counter()
    wav_path = convert_audio_to_wav(audio_input)

    try:
        model = get_whisper_model(model_name=model_name, n_threads=n_threads)
        segments = model.transcribe(wav_path)

        segment_list = []
        full_text_parts = []

        for seg in segments:
            seg_text = seg.text.strip()
            if seg_text:
                full_text_parts.append(seg_text)
                segment_list.append(
                    {
                        "t0": getattr(seg, "t0", 0),
                        "t1": getattr(seg, "t1", 0),
                        "text": seg_text,
                    }
                )

        full_text = " ".join(full_text_parts)
        elapsed = time.perf_counter() - start_time

        return {
            "text": full_text,
            "segments": segment_list,
            "duration_sec": round(elapsed, 2),
            "model_name": model_name,
        }
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
