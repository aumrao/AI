import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple


def get_ffmpeg_exe() -> str:
    """
    Locates or prepares the FFmpeg executable.
    1. Checks system PATH first (e.g., Linux /usr/bin/ffmpeg from packages.txt or Windows PATH).
    2. Falls back to imageio_ffmpeg bundled binary, creates a standard bin/ffmpeg (or bin/ffmpeg.exe),
       sets executable permissions on POSIX (Linux/macOS), and prepends to os.environ["PATH"].
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        raw_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if raw_exe and os.path.exists(raw_exe):
            is_windows = os.name == 'nt'
            project_root = Path(__file__).resolve().parent.parent
            bin_dir = project_root / "bin"
            bin_dir.mkdir(exist_ok=True)
            
            target_name = "ffmpeg.exe" if is_windows else "ffmpeg"
            std_ffmpeg = bin_dir / target_name
            
            if not std_ffmpeg.exists():
                try:
                    shutil.copyfile(raw_exe, std_ffmpeg)
                except Exception:
                    pass

            # Ensure executable permissions on Linux/macOS
            if not is_windows:
                try:
                    if std_ffmpeg.exists():
                        std_ffmpeg.chmod(0o755)
                    Path(raw_exe).chmod(0o755)
                except Exception:
                    pass

            # Prepend bin_dir and raw binary directory to os.environ["PATH"]
            bin_str = str(bin_dir.resolve())
            raw_dir = str(Path(raw_exe).parent.resolve())
            current_path = os.environ.get("PATH", "")
            
            for d in [bin_str, raw_dir]:
                if d not in current_path:
                    current_path = f"{d}{os.pathsep}{current_path}"
            os.environ["PATH"] = current_path

            # Re-check shutil.which after updating PATH
            which_now = shutil.which("ffmpeg")
            if which_now:
                return which_now

            if std_ffmpeg.exists():
                return str(std_ffmpeg)
            return raw_exe
    except Exception as e:
        print(f"Warning resolving FFmpeg: {e}")

    return "ffmpeg"


# Run once on module import to ensure PATH is populated
_INIT_FFMPEG = get_ffmpeg_exe()



def get_video_duration(video_path: str) -> float:
    """
    Extracts total duration in seconds from a video file using ffmpeg/ffprobe.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    
    # Try using ffmpeg to get duration info
    cmd = [
        ffmpeg_exe,
        "-i", str(video_path),
        "-hide_banner"
    ]
    try:
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, errors="replace")
        for line in result.stderr.splitlines():
            line = line.strip()
            if line.startswith("Duration:"):
                # e.g., Duration: 00:04:12.34, start: ...
                parts = line.split(",")
                duration_str = parts[0].replace("Duration:", "").strip()
                return timestamp_to_seconds(duration_str)
    except Exception as e:
        print(f"Warning: Failed to get duration via ffmpeg info: {e}")

    return 0.0


def seconds_to_timestamp(seconds: float, include_ms: bool = False) -> str:
    """
    Converts seconds float into a human-readable timestamp string (MM:SS or HH:MM:SS).
    """
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)

    if hrs > 0:
        base = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    else:
        base = f"{mins:02d}:{secs:02d}"

    if include_ms:
        return f"{base}.{ms:03d}"
    return base


def timestamp_to_seconds(ts_str: str) -> float:
    """
    Converts timestamp string like '01:23:45.67' or '04:12' to seconds float.
    """
    ts_str = ts_str.strip()
    parts = ts_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return float(m) * 60 + float(s)
        elif len(parts) == 1:
            return float(parts[0])
    except ValueError:
        pass
    return 0.0


def ensure_dir(path: str) -> Path:
    """Ensures a directory exists and returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_temp_dir(dir_path: str):
    """Safely cleans up temporary processing directory."""
    try:
        p = Path(dir_path)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        print(f"Warning cleaning directory {dir_path}: {e}")
