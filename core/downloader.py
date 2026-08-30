import os
import re
import requests
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
        r'(watch\?v=|embed/|v/|shorts/|.+\?v=)?([^&=%\?]{11})'
    )
    return bool(re.match(youtube_regex, url.strip()))


def extract_youtube_video_id(url: str) -> str:
    """Extracts standard 11-character video ID from any YouTube URL."""
    if not url:
        return ""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'youtube\.com\/embed\/([0-9A-Za-z_-]{11})',
        r'youtube\.com\/shorts\/([0-9A-Za-z_-]{11})',
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    # Fallback to last 11 chars
    clean = url.strip()
    return clean[-11:] if len(clean) >= 11 else clean


def get_base_ytdlp_opts() -> Dict[str, Any]:
    """
    Returns base yt-dlp configuration with multi-client extraction arguments,
    browser user-agents, and cookie support to prevent HTTP 403 Forbidden on cloud datacenters.
    """
    opts: Dict[str, Any] = {
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': get_ffmpeg_exe(),
        'geo_bypass': True,
        'nocheckcertificate': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android_creator', 'android', 'ios', 'mweb', 'web'],
                'player_skip': ['webpage', 'configs'],
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Sec-Fetch-Mode': 'navigate',
        },
    }

    # If cookies.txt exists or YTDLP_COOKIES is defined, attach it
    if os.path.exists("cookies.txt"):
        opts['cookiefile'] = "cookies.txt"
    elif os.getenv("YTDLP_COOKIES_PATH") and os.path.exists(os.getenv("YTDLP_COOKIES_PATH", "")):
        opts['cookiefile'] = os.getenv("YTDLP_COOKIES_PATH")

    return opts


def get_youtube_info(url: str) -> Dict[str, Any]:
    """
    Extracts metadata from a YouTube video without triggering bot detection.
    Uses official YouTube OEmbed API first (100% cloud-safe), with yt-dlp fallback.
    """
    video_id = extract_youtube_video_id(url)
    
    # 1. Primary: YouTube Official OEmbed API (Never blocked on Cloud/Datacenters)
    try:
        clean_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else url
        oembed_url = f"https://www.youtube.com/oembed?url={clean_url}&format=json"
        resp = requests.get(oembed_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title", "YouTube Video"),
                "duration": 0.0,
                "thumbnail": data.get("thumbnail_url", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"),
                "author": data.get("author_name", "YouTube Creator"),
                "description": f"Video by {data.get('author_name', 'YouTube Creator')}",
                "id": video_id,
                "url": url,
            }
    except Exception:
        pass

    # 2. Secondary: yt-dlp extractor
    ydl_opts = {
        **get_base_ytdlp_opts(),
        'skip_download': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "YouTube Video"),
                "duration": float(info.get("duration", 0) or 0.0),
                "thumbnail": info.get("thumbnail", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"),
                "author": info.get("uploader", "YouTube Creator"),
                "description": info.get("description", "")[:300] + "..." if info.get("description") else "",
                "id": info.get("id", video_id),
                "url": url,
            }
    except Exception:
        pass

    return {
        "title": f"YouTube Video ({video_id})",
        "duration": 0.0,
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg" if video_id else "",
        "author": "YouTube Creator",
        "description": "",
        "id": video_id,
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
    If video stream download is blocked by cloud bot detection (Sign in to confirm you're not a bot / 403),
    gracefully recovers by extracting metadata + subtitles directly so the AI pipeline completes 100%.
    """
    ensure_dir(output_dir)
    video_id = extract_youtube_video_id(url)
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

    target_height = 720
    if resolution == "1080p":
        target_height = 1080
    elif resolution == "480p":
        target_height = 480

    fmt = f'bestvideo[height<={target_height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={target_height}]+bestaudio/best[height<={target_height}]/best[ext=mp4]/best'

    base_opts = get_base_ytdlp_opts()
    ydl_opts = {
        **base_opts,
        'format': fmt,
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4',
        'nopart': True,
        'windowsfilenames': True,
        'retries': 5,
        'fragment_retries': 5,
        'concurrent_fragment_downloads': 4,
        'overwrites': False,
        'progress_hooks': [_progress_hook] if progress_callback else [],
    }

    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as primary_err:
        print(f"Notice: Direct video stream download bypassed ({primary_err}). Activating Cloud Resilient Metadata & Subtitles Engine...")
        # Graceful Cloud Recovery: Extract metadata + subtitles without downloading binary video
        meta = get_youtube_info(url)
        subs = extract_youtube_subtitles(url)
        calc_dur = float(meta.get("duration", 0.0))
        if calc_dur <= 0 and subs:
            calc_dur = float(subs[-1].get("end", 0.0))

        return {
            "title": meta.get("title", "YouTube Video"),
            "duration": calc_dur,
            "thumbnail": meta.get("thumbnail", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"),
            "author": meta.get("author", "YouTube Creator"),
            "video_path": "",
            "video_id": video_id,
            "source_type": "youtube",
            "subtitles": subs,
        }

    vid_id = info.get("id", video_id)
    expected_path = os.path.join(output_dir, f"{vid_id}.mp4")
    if not os.path.exists(expected_path):
        for file in os.listdir(output_dir):
            if file.startswith(vid_id):
                expected_path = os.path.join(output_dir, file)
                break

    duration = info.get("duration") or get_video_duration(expected_path)
    subtitles = parse_subtitles_from_info_dict(info)
    if not subtitles:
        subtitles = extract_youtube_subtitles(url)

    return {
        "title": info.get("title", "YouTube Video"),
        "duration": float(duration),
        "thumbnail": info.get("thumbnail", f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg"),
        "author": info.get("uploader", "YouTube Creator"),
        "video_path": os.path.abspath(expected_path) if os.path.exists(expected_path) else "",
        "video_id": vid_id,
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
    Retrieves subtitles (manual or auto-captions) from YouTube across any language.
    1. Primary: Uses youtube-transcript-api (Cloud-safe Innertube timedtext client).
    2. Fallback: Uses multi-format yt-dlp subtitle extractor.
    """
    video_id = extract_youtube_video_id(url)

    # 1. Primary Engine: youtube-transcript-api
    if video_id:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            yta = YouTubeTranscriptApi()
            t_list = list(yta.list(video_id))
            if t_list:
                # Prefer English transcripts first if present, otherwise take first available transcript (Hindi, Spanish, etc.)
                target_t = None
                for t in t_list:
                    code = getattr(t, 'language_code', '').lower()
                    if 'en' in code:
                        target_t = t
                        break
                if not target_t:
                    target_t = t_list[0]

                raw_data = target_t.fetch()
                segments = []
                for item in raw_data:
                    text = item.text if hasattr(item, 'text') else (item.get('text', '') if isinstance(item, dict) else str(item))
                    start = float(item.start if hasattr(item, 'start') else (item.get('start', 0.0) if isinstance(item, dict) else 0.0))
                    duration = float(item.duration if hasattr(item, 'duration') else (item.get('duration', 0.0) if isinstance(item, dict) else 0.0))
                    if text and text.strip() and text.strip() != '\n':
                        segments.append({
                            "start": round(start, 2),
                            "end": round(start + max(0.5, duration), 2),
                            "text": text.strip()
                        })
                if segments:
                    return segments
        except Exception as api_err:
            print(f"Notice: youtube-transcript-api ({api_err}), attempting yt-dlp fallback...")

    # 2. Secondary Engine: yt-dlp multi-format parser
    ydl_opts = {
        **get_base_ytdlp_opts(),
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['all', 'en.*', 'hi.*', 'en', 'hi', 'a.en', 'en-orig'],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return parse_subtitles_from_info_dict(info)
    except Exception as e:
        print(f"Notice: yt-dlp subtitle extraction skipped: {e}")
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
