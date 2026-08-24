# AI Technical Interview Portal (LangGraph & Whisper)

An enterprise-grade, multi-agent AI technical interview platform built with **LangGraph**, **Streamlit**, **OpenAI GPT-4o**, **Google Gemini**, and local **Whisper.cpp** speech-to-text transcription.

---

## 🌟 Key Capabilities

### 1. LangGraph Multi-Agent Architecture
- **Single Agent Mode**: Direct execution with **OpenAI GPT-4o** or **Google Gemini**.
- **Dual Agent + Evaluator Synthesis Mode (`Both`)**:
  - Concurrently dispatches generation to both Gemini and OpenAI LLMs.
  - An independent **Evaluator Bar Raiser Agent** assesses both outputs for depth, relevance, and coverage.
  - Merges and synthesizes the optimal **360-Degree 1-Hour Technical Interview Guide**.

### 2. 360-Degree Technical Interview Assessment (20 Q&A Pairs)
Covers all dimensions for senior/staff engineering screening:
1. **System Design & Distributed Architecture** (Saga transactions, event-driven microservices, Kafka, Redis rate limiting).
2. **Core Language & Concurrency Internals** (Virtual Threads, JVM memory model, G1GC/ZGC tuning, lock contention).
3. **Framework Ecosystem & Reactive Streams** (Spring Boot auto-configuration, Spring Security OAuth2/JWT, WebFlux backpressure).
4. **Database Design & Query Optimization** (JPA N+1 prevention, locking strategies, PostgreSQL execution plans, Outbox pattern).
5. **Testing Strategy & Observability** (Testing pyramid, Testcontainers, Contract Testing, OpenTelemetry tracing).
6. **Technical Leadership & AI Adoption** (Architecture reviews, ADRs, team mentorship, AI-assisted engineering with SonarQube quality gates).

### 3. Whisper.cpp Speech-to-Text & Live Candidate Evaluation
- **Microphone Recording & Audio Upload**: Live audio capture with local C++ accelerated `pywhispercpp`.
- **Automatic Performance Evaluation**:
  - Spoken interview transcript is immediately analyzed by the AI Evaluation Engine.
  - **Eligibility Badge**: `ELIGIBLE FOR SELECTION` / `STRONGLY RECOMMENDED` / `BORDERLINE` / `NOT ELIGIBLE`.
  - **Selection Probability**: Calculated percentage score (e.g. `90%`).
  - **Question-by-Question Assessment**: Evaluates technical accuracy, depth, and clarity.
  - **Strengths & Gaps**: Identifies candidate competencies and areas of concern.
- **Audio Session Reset**: `🔄 Reset` button clears recordings, transcripts, and evaluation results.

---

## 🏗️ Project Structure

```text
interview_portal/
├── app.py                      # Main application entry point with load_dotenv()
├── requirements.txt            # Python dependencies (Streamlit, LangGraph, LangChain, pywhispercpp)
├── Makefile                    # Make targets for linting, formatting, and running
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules (protects API keys and virtual environments)
├── README.md                   # Project documentation
├── .streamlit/
│   └── config.toml             # Streamlit server and theme configuration
└── src/
    └── interview_portal/
        ├── __init__.py         # Package initialization
        ├── state.py            # Streamlit session state management & getters/setters
        ├── styles.py           # Custom CSS design system (badges, cards, glassmorphism)
        ├── ui.py               # Main UI rendering and event handlers
        └── services/
            ├── __init__.py     # Service exports
            ├── graph.py        # LangGraph StateGraph workflow & evaluation engine
            └── transcription.py # Whisper.cpp local audio speech-to-text service
```

---

## 🚀 Quickstart Guide

### 1. Clone & Checkout Branch
```bash
git clone -b interviewportal https://github.com/aumrao/AI.git
cd AI
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Add your API credentials:
```ini
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Setup Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
