import hashlib
import streamlit as st

from .services.graph import evaluate_candidate_interview, run_interview_graph
from .services.transcription import transcribe_audio
from .state import (
    clear_current_mode,
    get_audio_mode,
    get_candidate_evaluation,
    get_current_mode,
    get_langgraph_last_run_info,
    get_live_interview_questions,
    get_summary_qa_text,
    reset_audio_state,
    set_audio_mode,
    set_candidate_evaluation,
    set_current_mode,
    set_generated_interview_data,
)
from .styles import load_css









APP_TITLE = "Interview Portal — AI Technical Screening"

DUMMY_SUMMARY_QA = """### Section 1: System Design, Architecture & Distributed Systems
Q1: How would you architect a distributed, multi-tenant enterprise platform handling 50,000+ concurrent requests using Spring Boot, Spring Cloud, and event-driven microservices?
A1: A resilient architecture uses Spring Cloud Gateway with OAuth2/JWT token validation, distributed rate limiting via Redis, and service discovery via Eureka/Consul. Asynchronous inter-service workflows (e.g. notifications, event logging) leverage Apache Kafka with idempotent consumer processing, while synchronous low-latency RPCs use gRPC. Multi-tenancy is enforced at the data layer using tenant discriminator columns or separate database schemas managed via Spring Data JPA and Hibernate multitenancy filters.

Q1 (Follow-up): How do you implement distributed transactions and guarantee data consistency across microservices when a business workflow fails midway?
A1 (Follow-up): Distributed transactions are coordinated using the Saga Pattern (orchestration with Camunda or choreography with Kafka event topics). Each step executes a local ACID transaction and emits a completion event. If a downstream service fails (e.g., payment failure), compensating transactions are triggered in reverse order to roll back previously executed states idempotently.

Q2: How do you design and implement resilient API Gateways, circuit breakers, and distributed rate limiting to handle traffic surges?
A2: We configure Spring Cloud Gateway with Resilience4j circuit breakers (Closed, Open, Half-Open) and fallback routes. Distributed rate limiting is implemented using the Token Bucket algorithm via Redis Reactive templates (`RedisRateLimiter`), preventing resource exhaustion from bursty traffic.

Q2 (Follow-up): How do you avoid cache stampedes and stale cache reads when caching high-traffic entity data in Redis?
A2 (Follow-up): Cache stampedes (thundering herd) are mitigated using probabilistic early expiration (XFetch algorithm), distributed mutex locks (Redisson), and cache-aside with asynchronous background warming. Stale reads are minimized by publishing cache eviction events across instances via Redis Pub/Sub whenever entity mutations occur.

### Section 2: Core Java 17+, JVM Internals & High-Throughput Concurrency
Q3: How do Virtual Threads (Project Loom in Java 21) differ from platform OS threads, and when should you use WebFlux vs Virtual Threads?
A3: Platform threads map 1:1 to OS kernel threads, incurring heavy memory overhead (~1MB stack) and expensive OS context switching. Virtual threads are lightweight user-mode threads managed by the JVM (~few KB stack), mounted onto Carrier Threads during CPU execution and unmounted during blocking I/O (sockets, files). Virtual Threads allow simple imperative synchronous code (`Thread.ofVirtual()`) to scale to millions of concurrent I/O operations without reactive boilerplate. WebFlux remains preferable when building continuous reactive streaming pipelines (backpressure, Server-Sent Events, WebSockets).

Q3 (Follow-up): Can you explain JVM garbage collection tuning (G1GC vs ZGC) and how you diagnose production memory leaks and high GC pause times?
A3 (Follow-up): G1GC divides the heap into regions for mixed generational collections, while ZGC achieves sub-millisecond max pause times using colored pointers and load barriers. Diagnosing memory leaks involves generating heap dumps via `jcmd <pid> GC.heap_dump` and analyzing retaining paths in Eclipse MAT, alongside CPU profiling with async-profiler to spot hot allocation sites.

### Section 3: Spring Ecosystem & Database Performance (JPA / PostgreSQL)
Q4: In Spring Data JPA, how do you resolve the N+1 select query problem and optimize complex batch data processing?
A4: The N+1 query issue occurs when lazy-loaded child relationships trigger individual queries per parent entity. We resolve it using `JOIN FETCH` in JPQL, `@EntityGraph` (defining attribute paths), or DTO projection queries. For high-volume batch processing, we use Spring Batch with chunk-based processing, disabling Hibernate dirty checking, and configuring `spring.jpa.properties.hibernate.jdbc.batch_size=50` with `order_inserts=true`.

Q4 (Follow-up): How do you implement optimistic vs pessimistic locking in Spring Data JPA to prevent lost updates during concurrent edits?
A4 (Follow-up): Optimistic locking uses `@Version` fields on entities; Hibernate checks the version during UPDATE statements and throws `OptimisticLockException` if updated concurrently. Pessimistic locking (`@Lock(LockModeType.PESSIMISTIC_WRITE)`) issues `SELECT ... FOR UPDATE` directly in the database, holding row locks during critical transactional updates.

### Section 4: Engineering Leadership, AI Productivity & Quality
Q5: As a Staff Engineer, how do you incorporate AI-assisted development tools (Cursor, GitHub Copilot, Claude) into the team's workflow while maintaining strict code quality and security?
A5: AI tools accelerate boilerplate generation, unit test creation, and architectural prototyping. However, we mandate strict human verification: AI-generated code must pass automated SonarQube static analysis, CI integration test suites, and peer code review. We use AI prompt libraries for standardizing architectural decision records (ADRs) and generating OpenAPI contract mocks.

Q5 (Follow-up): How do you mentor junior and mid-level engineers, conduct architectural reviews, and drive technical alignment across cross-functional teams?
A5 (Follow-up): Mentorship combines pair programming, structured code reviews with pedagogical feedback, and technical RFC (Request For Comments) review sessions. We conduct quarterly architecture roadmaps, establish shared design principles, and foster a blameless post-mortem culture for production incidents."""

DUMMY_INTERVIEW_QUESTIONS = """--- [Section 1: System Design, Architecture & Distributed Systems] ---
1. How would you architect a distributed, multi-tenant enterprise platform handling 50,000+ concurrent requests using Spring Boot, Spring Cloud, and event-driven microservices?
  └─ Follow-up: How do you implement distributed transactions and guarantee data consistency across microservices when a business workflow fails midway?
2. How do you design and implement resilient API Gateways, circuit breakers, and distributed rate limiting to handle traffic surges?
  └─ Follow-up: How do you avoid cache stampedes and stale cache reads when caching high-traffic entity data in Redis?

--- [Section 2: Core Java 17+, JVM Internals & Concurrency] ---
3. How do Virtual Threads (Project Loom in Java 21) differ from platform OS threads, and when should you use WebFlux vs Virtual Threads?
  └─ Follow-up: Can you explain JVM garbage collection tuning (G1GC vs ZGC) and how you diagnose production memory leaks and high GC pause times?

--- [Section 3: Spring Ecosystem & Database Performance] ---
4. In Spring Data JPA, how do you resolve the N+1 select query problem and optimize complex batch data processing?
  └─ Follow-up: How do you implement optimistic vs pessimistic locking in Spring Data JPA to prevent lost updates during concurrent edits?

--- [Section 4: Engineering Leadership & AI Productivity] ---
5. As a Staff Engineer, how do you incorporate AI-assisted development tools (Cursor, GitHub Copilot, Claude) into the team's workflow while maintaining strict code quality and security?
  └─ Follow-up: How do you mentor junior and mid-level engineers, conduct architectural reviews, and drive technical alignment across cross-functional teams?"""




def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero-header">
            <div class="hero-left">
                <div class="hero-logo-box">⚡</div>
                <div class="hero-title-group">
                    <h1>AI Technical Interview Portal</h1>
                    <p>Intelligent technical screening, live audio simulation & Q&A analysis</p>
                </div>
            </div>
            <div class="hero-status-pill">
                <span class="status-dot"></span>
                <span>System Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_section() -> tuple[str, str, str]:
    st.markdown('<div class="config-card">', unsafe_allow_html=True)
    st.markdown('<div class="config-section-title">⚙️ Configuration & Agent Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-grid">', unsafe_allow_html=True)

    st.markdown('<div class="section-label">📝 User Instructions</div>', unsafe_allow_html=True)
    instructions = st.text_area(
        "User Instructions",
        value=(
            "default instruction:prepare one hour technical interview questions "
            "and answers along with follow up interview questions and answers "
            "for below requirement:"
        ),
        height=85,
        label_visibility="collapsed",
        key="user_instructions",
    )

    st.markdown('<div class="section-label">💼 Job Requirement</div>', unsafe_allow_html=True)
    job_requirement = st.text_area(
        "Job Requirement",
        placeholder="Paste technical job requirement, skills, or tech stack here...",
        height=85,
        label_visibility="collapsed",
        key="job_requirement",
    )

    st.markdown('<div class="section-label">🤖 Select LLM Agent</div>', unsafe_allow_html=True)
    llm = st.radio(
        "Select LLM Agent",
        options=["Gemini (Google)", "OpenAI (GPT-4o)", "Both (Gemini & OpenAI)"],
        horizontal=True,
        label_visibility="collapsed",
        key="llm_agent",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # LangGraph Execution Control
    st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)
    gen_col1, _ = st.columns([2.5, 3.5])
    with gen_col1:
        if st.button(
            "⚡ Generate Interview Q&A with LangGraph",
            key="btn_generate_langgraph",
            type="primary",
            use_container_width=True,
        ):
            spinner_msg = (
                "🤖 Running LangGraph Multi-Agent Workflow: [Gemini Agent] + [OpenAI Agent] in parallel -> [AI Evaluator Agent] synthesizing best questions..."
                if "Both" in llm
                else f"🤖 Running LangGraph Workflow with {llm}..."
            )
            with st.spinner(spinner_msg):
                graph_result = run_interview_graph(instructions, job_requirement, llm)
                qa_text = graph_result.get("extracted_qa_text", "")
                questions_text = graph_result.get("extracted_questions_text", "")
                eval_notes = graph_result.get("evaluator_notes", None)

                set_generated_interview_data(
                    qa_text=qa_text,
                    questions_text=questions_text,
                    agent_name=llm,
                    evaluator_notes=eval_notes,
                )
                st.toast(f"✓ LangGraph ({llm}) completed! Q&A loaded.", icon="⚡")


    # Display LangGraph Execution Status / Evaluator Report
    last_run = get_langgraph_last_run_info()
    if last_run:
        agent_used = last_run.get("agent", "")
        eval_notes = last_run.get("evaluator_notes")

        if eval_notes and "Both" in agent_used:
            with st.expander("🔍 View LangGraph Evaluator Agent Synthesis Report", expanded=True):
                st.markdown(
                    f"""
                    <div class="eval-report-card">
                        <div class="eval-badge">🤖 LangGraph Evaluator Agent & Synthesis</div>
                        <div style="font-size: 14px; line-height: 1.6; color: #1e293b;">{eval_notes}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption(f"✓ Active Q&A generated by LangGraph ({agent_used})")

    st.markdown("</div>", unsafe_allow_html=True)
    return instructions, job_requirement, llm



def render_action_buttons() -> None:
    st.markdown('<div class="action-row">', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    current_mode = get_current_mode()

    with left:
        sum_btn_type = "primary" if current_mode == "summary" else "secondary"
        if st.button(
            "📋 Summarize Interview Questions & Answers",
            key="summarize_button",
            use_container_width=True,
            type=sum_btn_type,
        ):
            set_current_mode("summary")
            st.rerun()

    with right:
        int_btn_type = "primary" if current_mode == "interview" else "secondary"
        if st.button(
            "🎙️ Give Live Interview",
            key="interview_button",
            use_container_width=True,
            type=int_btn_type,
        ):
            set_current_mode("interview")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_dynamic_inputs(
    instructions: str = "",
    job_requirement: str = "",
    llm: str = "OpenAI (GPT-4o)",
) -> None:

    current_mode = get_current_mode()

    if not current_mode:
        return

    st.markdown("---")

    if current_mode == "summary":
        with st.container(border=True):
            col_content, col_remove = st.columns([13, 1])

            with col_remove:
                if st.button("✕", key="remove_summary", help="Close summary workspace"):
                    clear_current_mode()
                    st.rerun()

            with col_content:
                st.markdown(
                    '<div class="card-header-badge badge-summary">📋 Technical Q&A Summary Workspace</div>',
                    unsafe_allow_html=True,
                )
                st.caption("✨ Interview questions & answers generated and formatted for summarization:")
                qa_ver = st.session_state.get("summary_qa_version", 0)
                current_qa = get_summary_qa_text(default_value=DUMMY_SUMMARY_QA)
                st.text_area(
                    "Interview Questions & Answers",
                    value=current_qa,
                    height=450,
                    key=f"summary_text_input_{qa_ver}",
                )

    elif current_mode == "interview":
        audio_mode = get_audio_mode()

        with st.container(border=True):
            col_content, col_remove = st.columns([13, 1])

            with col_remove:
                if st.button("✕", key="remove_interview", help="Close interview session"):
                    clear_current_mode()
                    st.rerun()

            with col_content:
                st.markdown(
                    '<div class="card-header-badge badge-interview">🎙️ Live Audio & Text Interview Session</div>',
                    unsafe_allow_html=True,
                )
                st.caption("✨ Technical interview questions list for this live session:")
                int_ver = st.session_state.get("interview_questions_version", 0)
                current_questions = get_live_interview_questions(default_value=DUMMY_INTERVIEW_QUESTIONS)
                st.text_area(
                    "Interview Session Text & Notes",
                    value=current_questions,
                    height=320,
                    key=f"interview_text_input_{int_ver}",
                )


                st.markdown('<div class="audio-controls-label">🎧 Audio Response Options:</div>', unsafe_allow_html=True)
                btn_col1, btn_col2, btn_col3, _ = st.columns([1.4, 1.4, 1.0, 1.4])

                with btn_col1:
                    rec_btn_type = "primary" if audio_mode == "record" else "secondary"
                    if st.button(
                        "🎙️ Record Microphone",
                        key="btn_rec_interview",
                        type=rec_btn_type,
                        use_container_width=True,
                    ):
                        set_audio_mode("record")
                        st.rerun()

                with btn_col2:
                    up_btn_type = "primary" if audio_mode == "upload" else "secondary"
                    if st.button(
                        "📁 Upload Audio File",
                        key="btn_up_interview",
                        type=up_btn_type,
                        use_container_width=True,
                    ):
                        set_audio_mode("upload")
                        st.rerun()

                with btn_col3:
                    if st.button(
                        "🔄 Reset",
                        key="btn_reset_audio",
                        type="secondary",
                        use_container_width=True,
                        help="Reset audio recording, upload, and transcribed text",
                    ):
                        reset_audio_state()
                        st.toast("✓ Audio session reset!", icon="🔄")
                        st.rerun()

                audio_v = st.session_state.get("audio_session_version", 0)

                if audio_mode == "record":
                    st.markdown('<div class="audio-section-box">', unsafe_allow_html=True)
                    st.caption("🎙️ Click the microphone button to record your response. When you stop recording, speech is automatically converted to text and evaluated:")
                    recorded_audio = st.audio_input(
                        "Record Audio",
                        key=f"audio_rec_input_{audio_v}",
                        label_visibility="collapsed",
                    )
                    if recorded_audio is not None:
                        st.audio(recorded_audio)
                        audio_bytes = recorded_audio.getvalue()
                        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

                        if audio_hash != st.session_state.get("last_processed_audio_hash"):
                            with st.spinner("🎙️ Audio recording stopped. Converting speech to text with whisper.cpp..."):
                                res = transcribe_audio(audio_bytes, model_name="base.en")
                                trans_text = res.get("text", "").strip()
                                st.session_state["latest_transcribed_text"] = trans_text
                                st.session_state["last_processed_audio_hash"] = audio_hash

                            if trans_text:
                                with st.spinner("🤖 Evaluating candidate answers & calculating selection probability..."):
                                    eval_res = evaluate_candidate_interview(
                                        transcript=trans_text,
                                        job_requirement=job_requirement,
                                        instructions=instructions,
                                        selected_agent=llm,
                                    )
                                    set_candidate_evaluation(eval_res)
                                    st.toast(
                                        f"✓ Candidate evaluated: {eval_res.get('eligibility_status')} ({eval_res.get('selection_probability')}%)",
                                        icon="📊",
                                    )
                            st.rerun()

                    latest_text = st.session_state.get("latest_transcribed_text")
                    if latest_text:
                        st.markdown(
                            f"""
                            <div class="transcript-card">
                                <div class="transcript-header">
                                    <span class="transcript-badge">🎙️ Transcribed Audio Text</span>
                                </div>
                                <div class="transcript-text-display">{latest_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Candidate Performance & Selection Evaluation Card
                        eval_data = get_candidate_evaluation()
                        if eval_data:
                            eligibility = eval_data.get("eligibility_status", "ELIGIBLE FOR SELECTION")
                            probability = eval_data.get("selection_probability", 85)
                            report_text = eval_data.get("report_text", "")

                            badge_class = "eligibility-badge-eligible"
                            if "NOT" in eligibility.upper() or "REJECT" in eligibility.upper():
                                badge_class = "eligibility-badge-rejected"
                            elif "BORDERLINE" in eligibility.upper() or "CONDITIONAL" in eligibility.upper():
                                badge_class = "eligibility-badge-borderline"

                            st.markdown(
                                f"""
                                <div class="candidate-eval-card">
                                    <div class="candidate-eval-header">
                                        <div class="{badge_class}">
                                            <span>⚖️ {eligibility}</span>
                                        </div>
                                        <div class="prob-pill">
                                            <span>🎯 Selection Probability: {probability}%</span>
                                        </div>
                                    </div>
                                    <div class="eval-report-content">
                                        {report_text}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

                elif audio_mode == "upload":
                    st.markdown('<div class="audio-section-box">', unsafe_allow_html=True)
                    st.caption("📁 Choose or drop an audio file (WAV, MP3, M4A, OGG, AAC, FLAC, WEBM):")
                    uploaded_audio = st.file_uploader(
                        "Upload Audio File",
                        type=["wav", "mp3", "m4a", "ogg", "flac", "aac", "webm"],
                        key=f"audio_up_input_{audio_v}",
                        label_visibility="collapsed",
                    )
                    if uploaded_audio is not None:
                        st.audio(uploaded_audio)
                        upload_bytes = uploaded_audio.getvalue()
                        upload_hash = hashlib.sha256(upload_bytes).hexdigest()

                        if upload_hash != st.session_state.get("last_processed_upload_hash"):
                            with st.spinner("📁 Converting uploaded audio to text with whisper.cpp..."):
                                res = transcribe_audio(upload_bytes, model_name="base.en")
                                trans_text = res.get("text", "").strip()
                                st.session_state["latest_transcribed_text"] = trans_text
                                st.session_state["last_processed_upload_hash"] = upload_hash

                            if trans_text:
                                with st.spinner("🤖 Evaluating candidate answers & calculating selection probability..."):
                                    eval_res = evaluate_candidate_interview(
                                        transcript=trans_text,
                                        job_requirement=job_requirement,
                                        instructions=instructions,
                                        selected_agent=llm,
                                    )
                                    set_candidate_evaluation(eval_res)
                                    st.toast(
                                        f"✓ Candidate evaluated: {eval_res.get('eligibility_status')} ({eval_res.get('selection_probability')}%)",
                                        icon="📊",
                                    )
                            st.rerun()

                    latest_text = st.session_state.get("latest_transcribed_text")
                    if latest_text:
                        st.markdown(
                            f"""
                            <div class="transcript-card">
                                <div class="transcript-header">
                                    <span class="transcript-badge">📁 Transcribed Audio Text</span>
                                </div>
                                <div class="transcript-text-display">{latest_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Candidate Performance & Selection Evaluation Card
                        eval_data = get_candidate_evaluation()
                        if eval_data:
                            eligibility = eval_data.get("eligibility_status", "ELIGIBLE FOR SELECTION")
                            probability = eval_data.get("selection_probability", 85)
                            report_text = eval_data.get("report_text", "")

                            badge_class = "eligibility-badge-eligible"
                            if "NOT" in eligibility.upper() or "REJECT" in eligibility.upper():
                                badge_class = "eligibility-badge-rejected"
                            elif "BORDERLINE" in eligibility.upper() or "CONDITIONAL" in eligibility.upper():
                                badge_class = "eligibility-badge-borderline"

                            st.markdown(
                                f"""
                                <div class="candidate-eval-card">
                                    <div class="candidate-eval-header">
                                        <div class="{badge_class}">
                                            <span>⚖️ {eligibility}</span>
                                        </div>
                                        <div class="prob-pill">
                                            <span>🎯 Selection Probability: {probability}%</span>
                                        </div>
                                    </div>
                                    <div class="eval-report-content">
                                        {report_text}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    st.markdown("</div>", unsafe_allow_html=True)


def render_app() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    load_css()
    render_header()

    instructions, job_requirement, llm = render_input_section()

    st.session_state["request_context"] = {
        "instructions": instructions,
        "job_requirement": job_requirement,
        "llm_agent": llm,
    }

    render_action_buttons()
    render_dynamic_inputs(instructions=instructions, job_requirement=job_requirement, llm=llm)



