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
    resolution: str = "720p",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Downloads a YouTube video to the output directory as MP4.
    Defaults to 720p for fast download throughput while maintaining crisp summary visual quality.
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
    
    # Format selector based on resolution setting
    if resolution == "1080p":
        fmt = 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[ext=mp4]/best'
    elif resolution == "480p":
        fmt = 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best'
    else:  # Default 720p - optimum speed & quality balance
        fmt = 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best'

    ydl_opts = {
        'format': fmt,
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'nopart': True,                        # Prevents Windows .part file renaming lock collisions
        'windowsfilenames': True,              # Windows path safety
        'retries': 5,                          # Auto retry on socket drop
        'fragment_retries': 5,                 # Auto retry on DASH fragments
        'concurrent_fragment_downloads': 4,    # Multi-threaded fragment downloading
        'overwrites': False,                   # Re-use existing download if available
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
        subtitles = parse_subtitles_from_info_dict(info)

        return {
            "title": info.get("title", "YouTube Video"),
            "duration": float(duration),
            "thumbnail": info.get("thumbnail", ""),
            "author": info.get("uploader", "Unknown"),
            "video_path": os.path.abspath(expected_path),
            "video_id": video_id,
            "source_type": "youtube",
            "subtitles": subtitles,
        }




def parse_vtt_subtitles(vtt_text: str) -> List[Dict[str, Any]]:
    """Parses WebVTT subtitle text into timestamped segment dicts."""
    segments = []
    pattern = re.compile(r'(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})')
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = pattern.search(line)
        if match:
            g = match.groups()
            h1 = int(g[0]) if g[0] else 0
            m1 = int(g[1])
            s1 = int(g[2])
            ms1 = int(g[3])
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0

            h2 = int(g[4]) if g[4] else 0
            m2 = int(g[5])
            s2 = int(g[6])
            ms2 = int(g[7])
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0

            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not pattern.search(lines[i]):
                clean = re.sub(r'<[^>]+>', '', lines[i].strip())
                if clean and clean not in text_lines:
                    text_lines.append(clean)
                i += 1
            text = " ".join(text_lines).strip()
            if text:
                segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
        else:
            i += 1
    return segments


def parse_xml_subtitles(xml_text: str) -> List[Dict[str, Any]]:
    """Parses XML/TTML/SRV YouTube subtitles into timestamped segment dicts."""
    import html
    segments = []
    pattern = re.compile(r'<text[^>]*start="([\d.]+)"[^>]*(?:dur="([\d.]+)")?[^>]*>(.*?)</text>', re.DOTALL)
    for match in pattern.finditer(xml_text):
        start = float(match.group(1))
        dur = float(match.group(2)) if match.group(2) else 2.5
        raw = match.group(3)
        clean = html.unescape(re.sub(r'<[^>]+>', '', raw)).strip()
        if clean:
            segments.append({"start": round(start, 2), "end": round(start + dur, 2), "text": clean})
    return segments


def parse_subtitles_from_info_dict(info: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Extracts and parses subtitles directly from yt-dlp info dictionary."""
    import requests
    subs = {}
    if info.get('subtitles'):
        subs.update(info.get('subtitles'))
    if info.get('automatic_captions'):
        for k, v in info.get('automatic_captions').items():
            if k not in subs:
                subs[k] = v

    if not subs:
        return None

    ordered_keys = [k for k in subs if any(x in k.lower() for x in ['en', 'orig', 'auto'])] + list(subs.keys())

    for lang in ordered_keys:
        formats = subs.get(lang, [])
        # 1. Try json3 format
        for fmt in formats:
            if fmt.get('ext') == 'json3':
                try:
                    resp = requests.get(fmt['url'], timeout=6)
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
                                        "end": round(start + max(0.5, duration), 2),
                                        "text": text
                                    })
                        if segments:
                            return segments
                except Exception:
                    pass

        # 2. Try vtt format
        for fmt in formats:
            if fmt.get('ext') == 'vtt':
                try:
                    resp = requests.get(fmt['url'], timeout=6)
                    if resp.status_code == 200:
                        segments = parse_vtt_subtitles(resp.text)
                        if segments:
                            return segments
                except Exception:
                    pass

        # 3. Try srv1 / srv2 / srv3 / ttml format
        for fmt in formats:
            if fmt.get('ext') in ['srv1', 'srv2', 'srv3', 'ttml', 'xml']:
                try:
                    resp = requests.get(fmt['url'], timeout=6)
                    if resp.status_code == 200:
                        segments = parse_xml_subtitles(resp.text)
                        if segments:
                            return segments
                except Exception:
                    pass
    return None


def extract_youtube_subtitles(url: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves subtitles (manual or auto-captions) from YouTube in JSON3, VTT, or XML/SRV format.
    Supports all English variants and fallbacks. Returns list of segments or None.
    """
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en.*', 'en', 'a.en', 'en-orig', 'en-US', 'en-GB', 'en-IN', 'en-CA', 'en-AU'],
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return parse_subtitles_from_info_dict(info)
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
