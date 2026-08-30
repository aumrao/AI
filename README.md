# 🎬 AI Video Summarizer Portal

An AI-powered video summarization web application that ingests **YouTube URLs** or **uploaded video files**, transcribes speech with segment timestamps via **Whisper**, orchestrates highlight selection and structured summarization with **LangGraph** + **LLMs**, and produces a condensed summary video (`summary.mp4`) with chapter snapshots using **FFmpeg**.

---

- **High-Performance Video Pipeline (70%-90% Faster)**:
  - ⚡ **Turbo Accelerated Mode**: 720p stream optimization, greedy Whisper beam decoding, and fast-start muxing.
  - 🔑 **Multi-Account Gemini Key Pooling & Parallel Map-Reduce**: Provide 2-5 Gemini API keys from different accounts to process long video sections concurrently in parallel with zero rate-limit blocks!
  - 🚀 **Multi-Tier Speech Engine**: Instant YouTube captions (sub-second) -> Gemini Direct Audio (~4s) -> Groq Cloud Whisper (~1-2s) -> Local Whisper.
  - 🧵 **Multi-Threaded FFmpeg Parallel Cutting**: Concurrent clip extraction and snapshot generation across CPU cores.
  - 💾 **Smart Model Caching & Deduplication**: Model weights cached in-memory with automatic CUDA GPU & CPU multi-thread detection.
  - 📊 **Real-Time Telemetry Dashboard**: Complete execution time profiling breakdown for each processing phase.

- **100% Free LLM Support**:
  - 🆓 **Local Ollama (0 API Key / 0 Cost)**: Runs completely on your PC (auto-detects `mistral`, `deepseek-r1`, `qwen`, `gemma`, etc.).
  - 🆓 **Groq Free Fast Cloud Tier**: Free high-speed models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `gemma2-9b-it`).
  - 🆓 **OpenRouter Free Tier**: Free open models (`llama-3.3-70b-instruct:free`, `mistral-7b-instruct:free`).
  - 🆓 **Google AI Studio Free Tier**: Free Gemini keys for `gemini-3.7-flash`, `gemini-3.5-flash-lite`.
- **Flexible Ingestion**: Paste any YouTube URL (with thumbnail & metadata preview) or drag-and-drop local video files (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`).
- **Direct GUI API Key Input**: Enter optional API keys directly in the UI sidebar without configuring files.
- **Whisper Speech Transcription**: Fast local CPU/GPU transcription with `faster-whisper` and instant online caption extraction for YouTube videos.
- **LangGraph AI Orchestration**: Multi-node state graph executing intelligent highlight selection, speech-boundary optimization, and executive summary generation.
- **FFmpeg Video Studio**: Zero-config FFmpeg cutting, chapter snapshot extraction, and seamless clip concatenation into `summary.mp4`.
- **Comparison Player & Download**: Side-by-side view (Original vs Condensed Highlight Video) with 1-click `summary.mp4` download.
- **Interactive Timeline**: Clickable timeline chapters with duration badges, importance scores, and snapshots.


---

## 🏗️ Architecture Pipeline

```text
Streamlit UI (URL / Upload + API Key in GUI)
                 │
                 ▼
        Download / Ingestion (yt-dlp)
                 │
                 ▼
        Transcription (Whisper / Captions)
                 │
                 ▼
        Timestamped Transcript Segments
                 │
                 ▼
        LangGraph Orchestration
                 │
                 ▼
         LLM Highlight Selection & Summary
                 │
                 ▼
       Clip Boundary & Duration Optimizer
                 │
                 ▼
        FFmpeg Clip Extractor
                 │
                 ▼
        Chapter Snapshot Generator
                 │
                 ▼
        FFmpeg Concatenation Engine
                 │
                 ▼
          summary.mp4 (Download & Preview)
```

---

## 🚀 Getting Started

### 1. Installation
Install the dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
You can provide your API key directly on the web interface, or copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```

### 3. Launch the Web Application
```powershell
streamlit run app.py
```

---

## 🧪 Running Automated Tests
```powershell
python tests/test_pipeline.py
python tests/test_graph.py
```
