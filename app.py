import os
import sys
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core modules
from core.utils import (
    get_ffmpeg_exe,
    seconds_to_timestamp,
    get_video_duration,
    ensure_dir
)
from core.downloader import (
    is_valid_youtube_url,
    get_youtube_info,
    save_uploaded_video
)
from core.graph import create_video_summarizer_graph

# Page configuration
st.set_page_config(
    page_title="AI Video Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_file = Path(__file__).parent / "static" / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize Session State
if "processing" not in st.session_state:
    st.session_state.processing = False
if "result_state" not in st.session_state:
    st.session_state.result_state = None
if "active_video_info" not in st.session_state:
    st.session_state.active_video_info = None


def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🎬 AI Video Summarizer</div>
        <div class="main-subtitle">
            Condense lengthy videos into concise, highlight-packed summary videos with synchronized audio, chapter snapshots, and executive insights.
        </div>
    </div>
    """, unsafe_allow_html=True)


from core.llm_summarizer import get_installed_ollama_models


def render_sidebar():
    st.sidebar.markdown("### ⚙️ AI & Engine Configuration")

    # LLM Provider Selection
    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=[
            "Ollama (100% Free Local - No Key)",
            "Groq (Free Fast Cloud Tier)",
            "OpenRouter (Free Cloud Models)",
            "Google Gemini (Free AI Studio Key)",
            "OpenAI",
        ],
        index=0,
        help="Choose from 100% free local models (Ollama), free cloud APIs (Groq, OpenRouter), or Gemini/OpenAI."
    )

    provider_key_map = {
        "Ollama (100% Free Local - No Key)": ("ollama", "", "http://localhost:11434"),
        "Groq (Free Fast Cloud Tier)": ("groq", "GROQ_API_KEY", "https://console.groq.com/keys"),
        "OpenRouter (Free Cloud Models)": ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
        "Google Gemini (Free AI Studio Key)": ("google", "GEMINI_API_KEY", "https://aistudio.google.com/app/apikey"),
        "OpenAI": ("openai", "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    }

    prov_id, env_var, key_url = provider_key_map[provider]
    env_key = os.getenv(env_var, "") if env_var else ""

    if prov_id == "ollama":
        st.sidebar.success("🎉 **100% Free Local AI** (No API Key Required!)")
        ollama_models = get_installed_ollama_models()
        model_name = st.sidebar.selectbox(
            "Local Ollama Model",
            options=ollama_models,
            index=0,
            help="Models detected directly from your local Ollama installation."
        )
        user_api_key = "ollama"
        st.sidebar.caption("⚡ Powered by local Ollama on your PC")

    elif prov_id == "groq":
        st.sidebar.markdown("**🔑 Groq Free API Key**")
        user_api_key = st.sidebar.text_input(
            "Enter Groq API Key",
            value=env_key,
            type="password",
            placeholder="gsk_...",
            help=f"Get a free API key at: {key_url}"
        )
        if user_api_key:
            st.sidebar.caption("✅ Groq API Key configured")
        else:
            st.sidebar.info("💡 Groq provides free high-speed API access.")
            st.sidebar.markdown(f"[👉 Get a free Groq API Key]({key_url})")

        model_name = st.sidebar.selectbox(
            "Groq Free Model",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
            index=0
        )

    elif prov_id == "openrouter":
        st.sidebar.markdown("**🔑 OpenRouter API Key**")
        user_api_key = st.sidebar.text_input(
            "Enter OpenRouter Key",
            value=env_key,
            type="password",
            placeholder="sk-or-...",
            help=f"Get a free API key at: {key_url}"
        )
        if user_api_key:
            st.sidebar.caption("✅ OpenRouter Key configured")
        else:
            st.sidebar.info("💡 Use OpenRouter's free `:free` model tier.")
            st.sidebar.markdown(f"[👉 Get a free OpenRouter Key]({key_url})")

        model_name = st.sidebar.selectbox(
            "Free OpenRouter Model",
            options=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-2-9b-it:free",
                "mistralai/mistral-7b-instruct:free",
                "qwen/qwen-2.5-72b-instruct:free",
            ],
            index=0
        )

    elif prov_id == "google":
        st.sidebar.markdown("**🔑 Gemini API Key**")
        user_api_key = st.sidebar.text_input(
            "Enter Gemini API Key",
            value=env_key,
            type="password",
            placeholder="AIzaSy...",
            help=f"Get a free key at: {key_url}"
        )
        if user_api_key:
            st.sidebar.caption("✅ Gemini Key configured")
        else:
            st.sidebar.info("💡 Google AI Studio provides free API keys.")
            st.sidebar.markdown(f"[👉 Get a free Gemini Key]({key_url})")

        model_name = st.sidebar.selectbox(
            "Gemini Model",
            options=["gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview"],
            index=0
        )

    else:  # OpenAI
        st.sidebar.markdown("**🔑 OpenAI API Key**")
        user_api_key = st.sidebar.text_input(
            "Enter OpenAI API Key",
            value=env_key,
            type="password",
            placeholder="sk-...",
            help=f"Get an API key at: {key_url}"
        )
        if user_api_key:
            st.sidebar.caption("✅ OpenAI Key configured")
        else:
            st.sidebar.markdown(f"[👉 Get an OpenAI API Key]({key_url})")

        model_name = st.sidebar.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎙️ Speech & Video Settings")

    whisper_model = st.sidebar.selectbox(
        "Whisper Speech Model",
        options=["base", "tiny", "small", "medium"],
        index=0,
        help="'base' or 'tiny' are fastest for CPU, 'small' gives higher accuracy."
    )

    summary_ratio = st.sidebar.slider(
        "Target Summary Duration (%)",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
        help="Percentage of original video duration to retain in the condensed summary video."
    )

    focus_mode = st.sidebar.selectbox(
        "Summary Focus Mode",
        options=[
            "Balanced Highlights (Default)",
            "Key Insights & Explanations",
            "Actionable Steps & Conclusions",
            "Technical Deep-Dive",
            "Custom Prompt"
        ],
        index=0
    )

    custom_focus = None
    if focus_mode == "Custom Prompt":
        custom_focus = st.sidebar.text_area(
            "Custom Focus Instructions",
            placeholder="e.g., Focus primarily on the demo section and benchmark comparisons..."
        )
    elif focus_mode != "Balanced Highlights (Default)":
        custom_focus = f"Prioritize clips focusing on: {focus_mode}"

    st.sidebar.markdown("---")
    ffmpeg_status = "✅ Ready" if get_ffmpeg_exe() else "❌ Not found"
    st.sidebar.caption(f"FFmpeg Engine: {ffmpeg_status}")

    return {
        "provider": prov_id,
        "api_key": user_api_key.strip(),
        "model_name": model_name,
        "whisper_model": whisper_model,
        "summary_ratio": summary_ratio / 100.0,
        "custom_focus": custom_focus,
    }


def main():
    render_header()
    config = render_sidebar()

    # Workspace directory for processing
    workspace_dir = os.path.abspath("./workspace_runs")
    ensure_dir(workspace_dir)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Select Video Input")

    tab_yt, tab_upload = st.tabs(["🌐 YouTube Video URL", "📁 Upload Video File"])

    source_type = None
    source_target = None
    video_title_preview = None
    video_duration_preview = None
    video_thumb_preview = None

    with tab_yt:
        yt_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video link here."
        )

        if yt_url:
            if is_valid_youtube_url(yt_url):
                source_type = "youtube"
                source_target = yt_url.strip()

                # Fetch and display info preview
                with st.spinner("Fetching YouTube video details..."):
                    try:
                        info = get_youtube_info(source_target)
                        video_title_preview = info.get("title")
                        video_duration_preview = info.get("duration")
                        video_thumb_preview = info.get("thumbnail")

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if video_thumb_preview:
                                st.image(video_thumb_preview, use_container_width=True)
                        with col2:
                            st.markdown(f"**{video_title_preview}**")
                            st.markdown(f"👤 Channel: `{info.get('author')}`")
                            st.markdown(f"⏱️ Duration: `{seconds_to_timestamp(video_duration_preview)}`")
                            est_summary = (video_duration_preview or 0) * config["summary_ratio"]
                            st.markdown(f"✂️ Estimated Summary: ~`{seconds_to_timestamp(est_summary)}` ({int(config['summary_ratio']*100)}%)")
                    except Exception as e:
                        st.info(f"Ready to process YouTube URL: {source_target}")
            else:
                st.error("Please enter a valid YouTube URL.")

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a Video File",
            type=["mp4", "mov", "mkv", "avi", "webm"],
            help="Supported formats: MP4, MOV, MKV, AVI, WEBM"
        )
        if uploaded_file is not None:
            source_type = "upload"
            # Save uploaded file to temp directory
            uploads_dir = os.path.join(workspace_dir, "uploads")
            saved_info = save_uploaded_video(uploaded_file, uploads_dir)
            source_target = saved_info["video_path"]
            video_title_preview = saved_info["title"]
            video_duration_preview = saved_info["duration"]

            st.success(f"Uploaded: **{video_title_preview}** ({seconds_to_timestamp(video_duration_preview)})")
            st.video(source_target)

    st.markdown('</div>', unsafe_allow_html=True)

    # API Key Notice if missing (not required for local Ollama)
    if not config["api_key"] and config["provider"] != "ollama":
        st.markdown("""
        <div class="api-banner">
            <span style="font-size: 1.2rem;">🔑</span> 
            <strong>API Key Required:</strong> Please enter your API key in the left sidebar under <em>"LLM & Engine Configuration"</em> to enable AI video summarization, or select <strong>Ollama (100% Free Local)</strong>.
        </div>
        """, unsafe_allow_html=True)

    # Action Button
    can_process = bool(source_type and source_target and (config["api_key"] or config["provider"] == "ollama"))

    col_btn, _ = st.columns([2, 3])
    with col_btn:
        start_button = st.button(
            "✨ Generate Video Summary",
            disabled=not can_process,
            use_container_width=True,
        )

    if start_button:
        st.session_state.processing = True
        st.session_state.result_state = None

        # Clean old run files
        run_timestamp = int(time.time())
        current_run_dir = os.path.join(workspace_dir, f"run_{run_timestamp}")
        ensure_dir(current_run_dir)

        # Progress containers
        progress_bar = st.progress(5)
        status_text = st.empty()

        status_text.markdown("⏳ **Initializing LangGraph Video Summarizer Pipeline...**")

        try:
            # Build initial state
            initial_state = {
                "source_type": source_type,
                "source_url_or_path": source_target,
                "output_base_dir": current_run_dir,
                "target_summary_ratio": config["summary_ratio"],
                "custom_focus_prompt": config["custom_focus"],
                "llm_provider": config["provider"],
                "llm_api_key": config["api_key"],
                "llm_model_name": config["model_name"],
                "whisper_model_size": config["whisper_model"],
            }

            # Compile and stream graph
            graph = create_video_summarizer_graph()

            step_messages = {
                "prepare_media": ("📥 Ingesting & downloading media...", 20),
                "transcribe": ("🎙️ Transcribing audio with timestamps (Whisper)...", 45),
                "select_highlights": ("🧠 AI analyzing transcript & selecting key highlights...", 70),
                "extract_clips": ("✂️ FFmpeg cutting highlight clips...", 85),
                "generate_snapshots": ("📸 Capturing chapter snapshots...", 92),
                "concatenate_summary": ("🎬 Stitching final summary.mp4...", 98),
            }

            final_state = initial_state
            for output in graph.stream(initial_state):
                for node_name, state_update in output.items():
                    final_state.update(state_update)
                    if node_name in step_messages:
                        msg, pct = step_messages[node_name]
                        status_text.markdown(f"**{msg}**")
                        progress_bar.progress(pct)

            progress_bar.progress(100)
            status_text.markdown("🎉 **Video summarization complete!**")
            st.session_state.result_state = final_state
            st.session_state.processing = False
            st.rerun()

        except Exception as e:
            st.session_state.processing = False
            status_text.empty()
            progress_bar.empty()
            st.error(f"❌ Error during processing: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Display Results Dashboard if available
    if st.session_state.result_state:
        render_results_dashboard(st.session_state.result_state)


def render_results_dashboard(res: dict):
    st.markdown("---")
    st.markdown("## 📊 Summarization Results")

    orig_dur = res.get("video_duration", 0.0)
    final_video = res.get("final_video_path")
    final_dur = get_video_duration(final_video) if final_video else 0.0
    time_saved = max(0.0, orig_dur - final_dur)
    pct_saved = ((orig_dur - final_dur) / orig_dur * 100) if orig_dur > 0 else 0
    highlights = res.get("highlights", [])
    snapshots = res.get("snapshots", [])

    # Stats Banner
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{seconds_to_timestamp(orig_dur)}</div>
            <div class="stat-label">Original Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{seconds_to_timestamp(final_dur)}</div>
            <div class="stat-label">Summary Video Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value" style="color: #10B981;">{pct_saved:.0f}%</div>
            <div class="stat-label">Time Saved</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{len(highlights)}</div>
            <div class="stat-label">Key Highlights Extracted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Video Player Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("#### 📺 Original Video")
        orig_path = res.get("video_path")
        if orig_path and os.path.exists(orig_path):
            st.video(orig_path)
        elif res.get("source_type") == "youtube":
            st.video(res.get("source_url_or_path"))

    with v_col2:
        st.markdown("#### ⭐ Summarized Highlights Video (`summary.mp4`)")
        if final_video and os.path.exists(final_video):
            st.video(final_video)
            
            # Download Button
            with open(final_video, "rb") as vf:
                video_bytes = vf.read()
                st.download_button(
                    label="⬇️ Download Summarized Video (MP4)",
                    data=video_bytes,
                    file_name="summary.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        else:
            st.warning("Summarized video file not found.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Executive Overview & Takeaways
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### 📝 {res.get('summary_title', 'Video Summary')}")
    st.markdown(res.get("overview", ""))

    st.markdown("#### 💡 Key Takeaways")
    takeaways = res.get("key_takeaways", [])
    for t in takeaways:
        st.markdown(f"- {t}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Interactive Highlights Timeline & Chapter Snapshots
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⏱️ Extracted Highlights & Timeline Chapters")

    # Map snapshots by index
    snapshot_map = {s.get("index"): s.get("image_path") for s in snapshots}

    for idx, hl in enumerate(highlights, 1):
        s_time = hl.get("start_time", 0.0)
        e_time = hl.get("end_time", 0.0)
        title = hl.get("title", f"Highlight {idx}")
        reason = hl.get("reason", "")
        score = hl.get("importance_score", 8)
        img_path = snapshot_map.get(idx)

        h_col1, h_col2 = st.columns([1, 4])
        with h_col1:
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_container_width=True, caption=f"Snapshot @ {seconds_to_timestamp((s_time+e_time)/2)}")
            else:
                st.markdown(f"📸 `Clip #{idx}`")

        with h_col2:
            st.markdown(f"""
            <div class="timeline-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="font-size: 1.1rem; color: #FFFFFF;">#{idx}. {title}</strong>
                    <div>
                        <span class="badge badge-time">⏱️ {seconds_to_timestamp(s_time)} - {seconds_to_timestamp(e_time)}</span>
                        <span class="badge badge-score">⭐ {score}/10</span>
                    </div>
                </div>
                <div style="color: #94A3B8; font-size: 0.95rem;">{reason}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Transcript Accordion
    with st.expander("📜 View Full Timestamped Transcript"):
        segments = res.get("transcript_segments", [])
        for seg in segments:
            st.markdown(
                f"`[{seconds_to_timestamp(seg.get('start', 0))} - {seconds_to_timestamp(seg.get('end', 0))}]` "
                f"{seg.get('text', '')}"
            )


if __name__ == "__main__":
    main()
