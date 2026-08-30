import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from .utils import get_ffmpeg_exe, ensure_dir, seconds_to_timestamp


def extract_single_clip(
    source_video: str,
    start_time: float,
    end_time: float,
    output_clip_path: str,
    preset: str = "ultrafast"
) -> str:
    """
    Extracts an individual video clip with frame accuracy using fast seeking & ultrafast encoding.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(str(Path(output_clip_path).parent))

    duration = max(0.5, end_time - start_time)

    cmd_encode = [
        ffmpeg_exe,
        "-y",
        "-ss", str(start_time),
        "-i", str(source_video),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "0",
        str(output_clip_path)
    ]

    result = subprocess.run(cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to extract clip ({start_time}s - {end_time}s): {result.stderr}")

    return output_clip_path




def extract_clips(
    source_video: str,
    highlights: List[Any],
    output_dir: str,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[str]:
    """
    Extracts all highlighted segments in parallel using ThreadPoolExecutor for high throughput.
    """
    ensure_dir(output_dir)
    total = len(highlights)
    if total == 0:
        return []

    workers = max_workers or min(4, os.cpu_count() or 4)
    tasks = []

    for idx, hl in enumerate(highlights, 1):
        start_time = getattr(hl, "start_time", None) if hasattr(hl, "start_time") else hl.get("start_time")
        end_time = getattr(hl, "end_time", None) if hasattr(hl, "end_time") else hl.get("end_time")

        clip_filename = f"clip_{idx:03d}_{float(start_time):.1f}_{float(end_time):.1f}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)
        tasks.append((idx, float(start_time), float(end_time), clip_path))

    clip_results: Dict[int, str] = {}

    def _worker(task_info):
        t_idx, s_time, e_time, c_path = task_info
        extract_single_clip(source_video, s_time, e_time, c_path)
        return t_idx, os.path.abspath(c_path)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            t_idx, c_path = fut.result()
            clip_results[t_idx] = c_path
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total)

    # Return clips sorted chronologically by original index
    return [clip_results[i] for i in range(1, total + 1) if i in clip_results]


def generate_snapshots(
    source_video: str,
    highlights: List[Any],
    output_dir: str,
    max_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generates representative snapshot images (JPEG) concurrently across worker threads.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(output_dir)
    total = len(highlights)
    if total == 0:
        return []

    tasks = []
    for idx, hl in enumerate(highlights, 1):
        start_time = getattr(hl, "start_time", None) if hasattr(hl, "start_time") else hl.get("start_time")
        end_time = getattr(hl, "end_time", None) if hasattr(hl, "end_time") else hl.get("end_time")
        title = getattr(hl, "title", f"Highlight {idx}") if hasattr(hl, "title") else hl.get("title", f"Highlight {idx}")
        reason = getattr(hl, "reason", "") if hasattr(hl, "reason") else hl.get("reason", "")

        mid_time = float(start_time) + (float(end_time) - float(start_time)) / 2.0
        snapshot_filename = f"snapshot_{idx:03d}.jpg"
        snapshot_path = os.path.join(output_dir, snapshot_filename)

        tasks.append((idx, mid_time, title, reason, snapshot_path))

    def _snap_worker(item):
        i, m_time, t, r, s_path = item
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", str(m_time),
            "-i", str(source_video),
            "-vframes", "1",
            "-q:v", "2",
            str(s_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and os.path.exists(s_path):
            return {
                "index": i,
                "timestamp": m_time,
                "timestamp_str": seconds_to_timestamp(m_time),
                "title": t,
                "reason": r,
                "image_path": os.path.abspath(s_path)
            }
        return None

    workers = max_workers or min(6, os.cpu_count() or 4)
    snapshots_dict: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_snap_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                snapshots_dict[res["index"]] = res

    return [snapshots_dict[i] for i in range(1, total + 1) if i in snapshots_dict]


def concatenate_clips(
    clip_paths: List[str],
    output_path: str,
    preset: str = "ultrafast"
) -> str:
    """
    Concatenates a list of video clips into a single summary.mp4 file.
    Tries fast stream-copy first, falling back to ultrafast re-encoding.
    """
    if not clip_paths:
        raise ValueError("No clips to concatenate.")

    ffmpeg_exe = get_ffmpeg_exe()
    ensure_dir(str(Path(output_path).parent))

    # If only 1 clip, just copy
    if len(clip_paths) == 1:
        import shutil
        shutil.copyfile(clip_paths[0], output_path)
        return os.path.abspath(output_path)

    # Create concat list file
    concat_list_path = os.path.join(str(Path(output_path).parent), "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            norm_p = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{norm_p}'\n")

    # Fast attempt: Stream Copy concat
    copy_cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path)
    ]
    copy_res = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if copy_res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return os.path.abspath(output_path)

    # Fallback: ultrafast re-encoding concat
    reencode_cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", "0",
        str(output_path)
    ]

    result = subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concatenation failed: {result.stderr}")

    return os.path.abspath(output_path)

