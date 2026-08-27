import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from .utils import ensure_dir, get_video_duration
from .downloader import download_youtube_video, extract_youtube_subtitles
from .transcriber import AudioTranscriber
from .llm_summarizer import generate_video_summary
from .video_processor import extract_clips, generate_snapshots, concatenate_clips


class VideoSummarizerState(TypedDict, total=False):
    # User Inputs & Config
    source_type: str  # "youtube" or "upload"
    source_url_or_path: str
    output_base_dir: str
    target_summary_ratio: float
    custom_focus_prompt: Optional[str]

    # LLM & Whisper credentials & parameters
    llm_provider: str  # "google", "openai", "groq"
    llm_api_key: str
    llm_model_name: Optional[str]
    whisper_model_size: str  # "tiny", "base", "small", "medium"

    # Media metadata
    video_title: str
    video_path: str
    video_duration: float
    video_thumbnail: Optional[str]

    # Transcription
    transcript_segments: List[Dict[str, Any]]

    # LLM Output
    summary_title: str
    overview: str
    key_takeaways: List[str]
    highlights: List[Dict[str, Any]]

    # Video Processing
    extracted_clips: List[str]
    snapshots: List[Dict[str, Any]]
    final_video_path: str

    # State tracking
    current_step: str
    progress_pct: int
    error: Optional[str]


def download_or_prepare_media_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Downloads YouTube video or verifies local uploaded file."""
    output_dir = state.get("output_base_dir", "./temp_workspace")
    ensure_dir(output_dir)
    source_type = state.get("source_type", "youtube")
    source = state.get("source_url_or_path", "")

    if source_type == "youtube":
        dl_info = download_youtube_video(url=source, output_dir=os.path.join(output_dir, "downloads"))
        return {
            "video_title": dl_info.get("title", "YouTube Video"),
            "video_path": dl_info.get("video_path", ""),
            "video_duration": dl_info.get("duration", 0.0),
            "video_thumbnail": dl_info.get("thumbnail", ""),
            "current_step": "Video ready",
            "progress_pct": 20,
        }
    else:
        # Uploaded file path already prepared
        video_path = source
        duration = get_video_duration(video_path)
        return {
            "video_path": video_path,
            "video_duration": duration,
            "current_step": "Video ready",
            "progress_pct": 20,
        }


def transcribe_media_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Transcribes audio using YouTube captions or Whisper."""
    source_type = state.get("source_type", "youtube")
    source = state.get("source_url_or_path", "")
    video_path = state.get("video_path", "")
    output_dir = state.get("output_base_dir", "./temp_workspace")

    segments = None
    # Fast path: check online captions for YouTube
    if source_type == "youtube":
        try:
            segments = extract_youtube_subtitles(source)
        except Exception:
            segments = None

    if not segments:
        model_size = state.get("whisper_model_size", "base")
        transcriber = AudioTranscriber(model_size=model_size)
        temp_audio_dir = os.path.join(output_dir, "audio")
        segments = transcriber.transcribe(video_path=video_path, temp_dir=temp_audio_dir)

    if not segments:
        raise ValueError("Could not extract speech or transcript from the video.")

    return {
        "transcript_segments": segments,
        "current_step": "Transcription complete",
        "progress_pct": 45,
    }


def llm_select_highlights_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Uses LLM to analyze timestamped transcript and select key highlights & executive summary."""
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

    return {
        "summary_title": summary_result.title,
        "overview": summary_result.overview,
        "key_takeaways": summary_result.key_takeaways,
        "highlights": highlights_dict,
        "current_step": "Highlights & summary synthesized by AI",
        "progress_pct": 65,
    }


def extract_video_clips_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Cuts highlighted segments from source video using FFmpeg."""
    source_video = state.get("video_path", "")
    highlights = state.get("highlights", [])
    output_dir = os.path.join(state.get("output_base_dir", "./temp_workspace"), "clips")

    clips = extract_clips(source_video=source_video, highlights=highlights, output_dir=output_dir)

    return {
        "extracted_clips": clips,
        "current_step": "Key clips extracted",
        "progress_pct": 80,
    }


def generate_snapshots_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Generates snapshot images for key highlight chapters."""
    source_video = state.get("video_path", "")
    highlights = state.get("highlights", [])
    output_dir = os.path.join(state.get("output_base_dir", "./temp_workspace"), "snapshots")

    snapshots = generate_snapshots(source_video=source_video, highlights=highlights, output_dir=output_dir)

    return {
        "snapshots": snapshots,
        "current_step": "Snapshots captured",
        "progress_pct": 90,
    }


def concatenate_summary_video_node(state: VideoSummarizerState) -> Dict[str, Any]:
    """Concatenates all extracted clips into summary.mp4."""
    clips = state.get("extracted_clips", [])
    output_dir = os.path.join(state.get("output_base_dir", "./temp_workspace"), "final")
    output_path = os.path.join(output_dir, "summary.mp4")

    final_video = concatenate_clips(clip_paths=clips, output_path=output_path)

    return {
        "final_video_path": final_video,
        "current_step": "Summarized video ready!",
        "progress_pct": 100,
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
