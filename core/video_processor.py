import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from .utils import get_ffmpeg_exe, ensure_dir, seconds_to_timestamp


def extract_single_clip(
    source_video: str,
    start_time: float,
    end_time: float,
    output_clip_path: str
) -> str:
    """
    Extracts an individual video clip from start_time to end_time using FFmpeg.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(str(Path(output_clip_path).parent))

    duration = max(0.5, end_time - start_time)

    # Accurate cutting with re-encoding to avoid black frames or desynced audio
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", str(start_time),
        "-i", str(source_video),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_clip_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to extract clip ({start_time}s - {end_time}s): {result.stderr}")

    return output_clip_path


def extract_clips(
    source_video: str,
    highlights: List[Any],
    output_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[str]:
    """
    Extracts all highlighted segments from the source video into separate mp4 files.
    """
    ensure_dir(output_dir)
    clip_paths = []
    total = len(highlights)

    for idx, hl in enumerate(highlights, 1):
        # hl can be HighlightClip object or dict
        start_time = getattr(hl, "start_time", None) if hasattr(hl, "start_time") else hl.get("start_time")
        end_time = getattr(hl, "end_time", None) if hasattr(hl, "end_time") else hl.get("end_time")

        clip_filename = f"clip_{idx:03d}_{start_time:.1f}_{end_time:.1f}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)

        extract_single_clip(source_video, float(start_time), float(end_time), clip_path)
        clip_paths.append(os.path.abspath(clip_path))

        if progress_callback:
            progress_callback(idx, total)

    return clip_paths


def generate_snapshots(
    source_video: str,
    highlights: List[Any],
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Generates representative snapshot images (JPEG) for each highlight segment.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(output_dir)
    snapshots = []

    for idx, hl in enumerate(highlights, 1):
        start_time = getattr(hl, "start_time", None) if hasattr(hl, "start_time") else hl.get("start_time")
        end_time = getattr(hl, "end_time", None) if hasattr(hl, "end_time") else hl.get("end_time")
        title = getattr(hl, "title", f"Highlight {idx}") if hasattr(hl, "title") else hl.get("title", f"Highlight {idx}")
        reason = getattr(hl, "reason", "") if hasattr(hl, "reason") else hl.get("reason", "")

        # Pick timestamp midway through the clip
        mid_time = float(start_time) + (float(end_time) - float(start_time)) / 2.0
        snapshot_filename = f"snapshot_{idx:03d}.jpg"
        snapshot_path = os.path.join(output_dir, snapshot_filename)

        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", str(mid_time),
            "-i", str(source_video),
            "-vframes", "1",
            "-q:v", "2",
            str(snapshot_path)
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and os.path.exists(snapshot_path):
            snapshots.append({
                "index": idx,
                "timestamp": mid_time,
                "timestamp_str": seconds_to_timestamp(mid_time),
                "title": title,
                "reason": reason,
                "image_path": os.path.abspath(snapshot_path)
            })

    return snapshots


def concatenate_clips(
    clip_paths: List[str],
    output_path: str
) -> str:
    """
    Concatenates a list of video clips into a single summary.mp4 file.
    """
    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(str(Path(output_path).parent))

    # If only 1 clip, just copy/rename
    if len(clip_paths) == 1:
        import shutil
        shutil.copyfile(clip_paths[0], output_path)
        return os.path.abspath(output_path)

    # Create concat list file
    concat_list_path = os.path.join(str(Path(output_path).parent), "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            # Escape single quotes and use forward slashes for FFmpeg
            norm_p = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{norm_p}'\n")

    # Concat demuxer with re-encoding to ensure perfect stream synchronization
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concatenation failed: {result.stderr}")

    return os.path.abspath(output_path)
