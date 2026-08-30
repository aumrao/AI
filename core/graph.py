import os
import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from .utils import ensure_dir, get_video_duration
from .downloader import download_youtube_video, extract_youtube_subtitles
from .transcriber import AudioTranscriber, extract_audio, extract_compressed_audio, transcribe_with_groq_or_openai
from .llm_summarizer import generate_video_summary, generate_video_summary_from_audio
from .video_processor import extract_clips, generate_snapshots, concatenate_clips


class VideoSummarizerState(TypedDict, total=False):
    # User Inputs & Config
    source_type: str  # "youtube" or "upload"
    source_url_or_path: str
    output_base_dir: str
    target_summary_ratio: float
    custom_focus_prompt: Optional[str]
    video_resolution: str  # "720p", "1080p", "480p"

    # LLM & Whisper credentials & parameters
    llm_provider: str  # "google", "openai", "groq", "ollama", "openrouter"
    llm_api_key: str
    llm_model_name: Optional[str]
    whisper_model_size: str  # "tiny", "base", "small", "medium"
    transcription_provider: str  # "auto", "groq", "openai", "local", "captions"
    cloud_whisper_api_key: Optional[str]

    # Media metadata
    video_title: str
    video_path: str
    video_duration: float
    video_thumbnail: Optional[str]

    # Transcription
    transcript_segments: List[Dict[str, Any]]
    transcription_source: str  # "youtube_captions", "gemini_audio_fast", "groq_whisper", "local_whisper"
    direct_summary: Optional[Dict[str, Any]]

    # LLM Output
    summary_title: str
    overview: str
    key_takeaways: List[str]
    highlights: List[Dict[str, Any]]

    # Video Processing
    extracted_clips: List[str]
    snapshots: List[Dict[str, Any]]
    final_video_path: str

    # State tracking & Performance Telemetry
    current_step: str
    progress_pct: int
    timing_metrics: Dict[str, float]
    error: Optional[str]


def download_or_prepare_media_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Downloads YouTube video (optimized 720p/DASH) or prepares uploaded file."""
    t0 = time.perf_counter()
    output_dir = state.get("output_base_dir", "./temp_workspace")
    ensure_dir(output_dir)
    source_type = state.get("source_type", "youtube")
    source = state.get("source_url_or_path", "")
    resolution = state.get("video_resolution", "720p")
    timings = dict(state.get("timing_metrics", {}))

    if source_type == "youtube":
        dl_info = download_youtube_video(
            url=source,
            output_dir=os.path.join(output_dir, "downloads"),
            resolution=resolution
        )
        timings["download_time"] = round(time.perf_counter() - t0, 2)
        return {
            "video_title": dl_info.get("title", "YouTube Video"),
            "video_path": dl_info.get("video_path", ""),
            "video_duration": dl_info.get("duration", 0.0),
            "video_thumbnail": dl_info.get("thumbnail", ""),
            "transcript_segments": dl_info.get("subtitles"),
            "current_step": "Video ready",
            "progress_pct": 20,
            "timing_metrics": timings,
        }
    else:
        video_path = source
        duration = get_video_duration(video_path)
        timings["download_time"] = round(time.perf_counter() - t0, 2)
        return {
            "video_path": video_path,
            "video_duration": duration,
            "current_step": "Video ready",
            "progress_pct": 20,
            "timing_metrics": timings,
        }


def transcribe_media_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Transcribes audio with multi-tier acceleration: Pre-parsed Captions (0s) -> Online Captions (0.5s) -> Gemini Audio (4s) -> Groq Cloud (2s) -> Local Whisper."""
    t0 = time.perf_counter()
    source_type = state.get("source_type", "youtube")
    source = state.get("source_url_or_path", "")
    video_path = state.get("video_path", "")
    duration = state.get("video_duration", 0.0)
    output_dir = state.get("output_base_dir", "./temp_workspace")
    provider = state.get("llm_provider", "google").lower()
    api_key = state.get("llm_api_key", "")
    model_name = state.get("llm_model_name")
    trans_provider = state.get("transcription_provider", "auto").lower()
    cloud_key = state.get("cloud_whisper_api_key") or (api_key if provider in ["groq", "openai"] else "")
    target_ratio = state.get("target_summary_ratio", 0.30)
    custom_focus = state.get("custom_focus_prompt")
    timings = dict(state.get("timing_metrics", {}))

    # 1. Tier 1A: Pre-parsed YouTube Captions (Instant 0.00s!)
    segments = state.get("transcript_segments")
    trans_source = "youtube_captions" if segments else "local_whisper"
    direct_sum_dict = None

    # 1. Tier 1B: Online Captions Fetch if not pre-parsed
    if not segments and trans_provider in ["auto", "captions"] and source_type == "youtube":
        try:
            segments = extract_youtube_subtitles(source)
            if segments:
                trans_source = "youtube_captions"
        except Exception:
            segments = None

    # 2. Tier 2: Google Gemini Flash Direct Audio Understanding (~3-5s lightweight MP3)
    if not segments and provider in ["google", "gemini"] and api_key and trans_provider in ["auto", "gemini"]:
        try:
            temp_audio_dir = os.path.join(output_dir, "audio")
            audio_path = os.path.join(temp_audio_dir, "extracted_audio.mp3")
            extract_compressed_audio(video_path, audio_path)

            gemini_res, gemini_segs = generate_video_summary_from_audio(
                audio_path=audio_path,
                total_duration=duration,
                api_key=api_key,
                model_name=model_name,
                target_summary_ratio=target_ratio,
                custom_focus_prompt=custom_focus
            )
            segments = gemini_segs
            trans_source = "gemini_audio_fast"
            direct_sum_dict = {
                "title": gemini_res.title,
                "overview": gemini_res.overview,
                "key_takeaways": gemini_res.key_takeaways,
                "highlights": [h.model_dump() for h in gemini_res.highlights]
            }
        except Exception as e:
            print(f"Notice: Gemini direct audio fallback to Whisper ({e})")
            segments = None

    # 3. Tier 3: Groq Cloud Whisper (~1-2s ultra-fast)
    if not segments and (trans_provider == "groq" or (trans_provider == "auto" and cloud_key and (provider == "groq" or "gsk_" in cloud_key))):
        try:
            temp_audio_dir = os.path.join(output_dir, "audio")
            audio_path = os.path.join(temp_audio_dir, "extracted_audio.mp3")
            extract_compressed_audio(video_path, audio_path)
            segments = transcribe_with_groq_or_openai(audio_path=audio_path, api_key=cloud_key, provider="groq")
            if segments:
                trans_source = "groq_whisper"
        except Exception as e:
            print(f"Notice: Groq cloud whisper skipped ({e}), falling back to local Whisper.")
            segments = None

    # 4. Tier 4: Local Accelerated Whisper (beam_size=1 greedy)
    if not segments:
        model_size = state.get("whisper_model_size", "base")
        transcriber = AudioTranscriber(model_size=model_size)
        temp_audio_dir = os.path.join(output_dir, "audio")
        segments = transcriber.transcribe(video_path=video_path, temp_dir=temp_audio_dir, beam_size=1)
        trans_source = f"local_whisper_{model_size}"


    if not segments:
        raise ValueError("Could not extract speech or transcript from the video.")

    timings["transcribe_time"] = round(time.perf_counter() - t0, 2)

    return {
        "transcript_segments": segments,
        "transcription_source": trans_source,
        "direct_summary": direct_sum_dict,
        "current_step": "Transcription complete",
        "progress_pct": 45,
        "timing_metrics": timings,
    }


def llm_select_highlights_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Uses LLM to analyze transcript and select key highlights & executive summary (or reuse direct Gemini summary)."""
    t0 = time.perf_counter()
    timings = dict(state.get("timing_metrics", {}))

    # Check if direct summary was already synthesized in fast Gemini audio pass
    direct_sum = state.get("direct_summary")
    if direct_sum:
        timings["llm_time"] = round(time.perf_counter() - t0, 2)
        return {
            "summary_title": direct_sum.get("title", "Video Summary"),
            "overview": direct_sum.get("overview", ""),
            "key_takeaways": direct_sum.get("key_takeaways", []),
            "highlights": direct_sum.get("highlights", []),
            "current_step": "Highlights & summary synthesized by AI",
            "progress_pct": 65,
            "timing_metrics": timings,
        }

    transcript_segments = state.get("transcript_segments", [])
    duration = state.get("video_duration", 0.0)
    provider = state.get("llm_provider", "google")
    api_key = state.get("llm_api_key", "")
    model_name = state.get("llm_model_name")
    target_ratio = state.get("target_summary_ratio", 0.30)
    custom_focus = state.get("custom_focus_prompt")

    summary_result = generate_video_summary(
        transcript_segments=transcript_segments,
        total_duration=duration,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        target_summary_ratio=target_ratio,
        custom_focus_prompt=custom_focus
    )

    highlights_dict = [h.model_dump() for h in summary_result.highlights]
    timings["llm_time"] = round(time.perf_counter() - t0, 2)

    return {
        "summary_title": summary_result.title,
        "overview": summary_result.overview,
        "key_takeaways": summary_result.key_takeaways,
        "highlights": highlights_dict,
        "current_step": "Highlights & summary synthesized by AI",
        "progress_pct": 65,
        "timing_metrics": timings,
    }



def extract_video_clips_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Cuts highlighted segments from source video using parallel multi-threaded FFmpeg."""
    t0 = time.perf_counter()
    source_video = state.get("video_path", "")
    highlights = state.get("highlights", [])
    output_dir = os.path.join(state.get("output_base_dir", "./temp_workspace"), "clips")
    timings = dict(state.get("timing_metrics", {}))

    clips = extract_clips(source_video=source_video, highlights=highlights, output_dir=output_dir)
    timings["clip_extract_time"] = round(time.perf_counter() - t0, 2)

    return {
        "extracted_clips": clips,
        "current_step": "Key clips extracted",
        "progress_pct": 80,
        "timing_metrics": timings,
    }


def generate_snapshots_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Generates snapshot images for key highlight chapters in parallel."""
    t0 = time.perf_counter()
    source_video = state.get("video_path", "")
    highlights = state.get("highlights", [])
    output_dir = os.path.join(state.get("output_base_dir", "./temp_workspace"), "snapshots")
    timings = dict(state.get("timing_metrics", {}))

    snapshots = generate_snapshots(source_video=source_video, highlights=highlights, output_dir=output_dir)
    timings["snapshot_time"] = round(time.perf_counter() - t0, 2)

    return {
        "snapshots": snapshots,
        "current_step": "Snapshots captured",
        "progress_pct": 90,
        "timing_metrics": timings,
    }


def concatenate_summary_video_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Concatenates all extracted clips into summary.mp4 with stream-copy / fast stitching."""
    t0 = time.perf_counter()
    clips = state.get("extracted_clips", [])
    output_dir = os.path.join(state.get("output_base_dir", "./temp_workspace"), "final")
    output_path = os.path.join(output_dir, "summary.mp4")
    timings = dict(state.get("timing_metrics", {}))

    final_video = concatenate_clips(clip_paths=clips, output_path=output_path)
    timings["concat_time"] = round(time.perf_counter() - t0, 2)
    timings["total_pipeline_time"] = round(
        timings.get("download_time", 0) +
        timings.get("transcribe_time", 0) +
        timings.get("llm_time", 0) +
        timings.get("clip_extract_time", 0) +
        timings.get("snapshot_time", 0) +
        timings.get("concat_time", 0),
        2
    )

    return {
        "final_video_path": final_video,
        "current_step": "Summarized video ready!",
        "progress_pct": 100,
        "timing_metrics": timings,
    }


def create_video_summarizer_graph():
    """Builds and compiles the LangGraph state graph."""
    workflow = StateGraph(VideoSummarizerState)

    workflow.add_node("prepare_media", download_or_prepare_media_node)
    workflow.add_node("transcribe", transcribe_media_node)
    workflow.add_node("select_highlights", llm_select_highlights_node)
    workflow.add_node("extract_clips", extract_video_clips_node)
    workflow.add_node("generate_snapshots", generate_snapshots_node)
    workflow.add_node("concatenate_summary", concatenate_summary_video_node)

    workflow.set_entry_point("prepare_media")
    workflow.add_edge("prepare_media", "transcribe")
    workflow.add_edge("transcribe", "select_highlights")
    workflow.add_edge("select_highlights", "extract_clips")
    workflow.add_edge("extract_clips", "generate_snapshots")
    workflow.add_edge("generate_snapshots", "concatenate_summary")
    workflow.add_edge("concatenate_summary", END)

    return workflow.compile()

