import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import yt_dlp
from .utils import ensure_dir, get_video_duration, get_ffmpeg_exe


def is_valid_youtube_url(url: str) -> bool:
    """Validates if a URL is a supported YouTube URL."""
    if not url:
        return False
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return bool(re.match(youtube_regex, url.strip()))


def get_youtube_info(url: str) -> Dict[str, Any]:
    """Extracts metadata from a YouTube video without downloading."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'ffmpeg_location': get_ffmpeg_exe(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "YouTube Video"),
            "duration": info.get("duration", 0),
            "thumbnail": info.get("thumbnail", ""),
            "author": info.get("uploader", "Unknown"),
            "description": info.get("description", "")[:300] + "..." if info.get("description") else "",
            "id": info.get("id", ""),
            "url": url,
        }


def download_youtube_video(
    url: str,
    output_dir: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Downloads a YouTube video to the output directory as MP4.
    Returns metadata including local video_path.
    """
    ensure_dir(output_dir)
    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")

    def _progress_hook(d):
        if progress_callback and d.get('status') == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
            downloaded = d.get('downloaded_bytes', 0)
            percent = (downloaded / total) * 100
            progress_callback({
                "status": "downloading",
                "percent": percent,
                "speed": d.get('speed', 0),
                "eta": d.get('eta', 0),
            })

    ffmpeg_exe = get_ffmpeg_exe()
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[ext=mp4]/best',
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'nopart': True,                        # Prevents Windows .part file renaming lock collisions
        'windowsfilenames': True,              # Windows path safety
        'retries': 10,                         # Auto retry on socket drop
        'fragment_retries': 10,                # Auto retry on DASH fragments
        'concurrent_fragment_downloads': 1,    # Avoid thread file-locking on Windows
        'overwrites': True,
        'ffmpeg_location': ffmpeg_exe,
        'progress_hooks': [_progress_hook] if progress_callback else [],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")
        # Locate the downloaded file
        expected_path = os.path.join(output_dir, f"{video_id}.mp4")
        if not os.path.exists(expected_path):
            # Check if another extension was produced
            for file in os.listdir(output_dir):
                if file.startswith(video_id):
                    expected_path = os.path.join(output_dir, file)
                    break

        duration = info.get("duration") or get_video_duration(expected_path)

        return {
            "title": info.get("title", "YouTube Video"),
            "duration": float(duration),
            "thumbnail": info.get("thumbnail", ""),
            "author": info.get("uploader", "Unknown"),
            "video_path": os.path.abspath(expected_path),
            "video_id": video_id,
            "source_type": "youtube",
        }


def extract_youtube_subtitles(url: str) -> Optional[List[Dict[str, Any]]]:
    """
    Tries to retrieve subtitles directly from YouTube (manual or automatic)
    with start/end timestamps. Returns list of segments or None.
    """
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'en-US', 'en-orig'],
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get('subtitles') or info.get('automatic_captions') or {}
            # Check for English subtitles in json3 or vtt format
            for lang in ['en', 'en-US', 'en-orig']:
                if lang in subs:
                    for fmt in subs[lang]:
                        if fmt.get('ext') == 'json3':
                            # yt-dlp can fetch this format with detailed word/event timestamps
                            import requests
                            resp = requests.get(fmt['url'], timeout=10)
                            if resp.status_code == 200:
                                data = resp.json()
                                segments = []
                                for event in data.get('events', []):
                                    if 'segs' in event:
                                        text = "".join(s.get('utf8', '') for s in event['segs']).strip()
                                        if text and text != '\n':
                                            start = event.get('tStartMs', 0) / 1000.0
                                            duration = event.get('dDurationMs', 0) / 1000.0
                                            segments.append({
                                                "start": round(start, 2),
                                                "end": round(start + duration, 2),
                                                "text": text
                                            })
                                if segments:
                                    return segments
    except Exception as e:
        print(f"Warning: Failed to fetch online subtitles: {e}")
    return None


def save_uploaded_video(uploaded_file, output_dir: str) -> Dict[str, Any]:
    """
    Saves an uploaded video file object from Streamlit to disk.
    """
    ensure_dir(output_dir)
    file_name = uploaded_file.name
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', file_name)
    save_path = os.path.join(output_dir, clean_name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    duration = get_video_duration(save_path)
    title = Path(file_name).stem.replace("_", " ").title()

    return {
        "title": title,
        "duration": float(duration),
        "thumbnail": None,
        "author": "Uploaded File",
        "video_path": os.path.abspath(save_path),
        "video_id": Path(clean_name).stem,
        "source_type": "upload",
    }
