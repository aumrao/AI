import streamlit as st


def _ensure_state() -> None:
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = None

    if "audio_mode" not in st.session_state:
        st.session_state.audio_mode = "record"

    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = None

    if "whisper_model_name" not in st.session_state:
        st.session_state.whisper_model_name = "base.en"

    if "interview_notes_version" not in st.session_state:
        st.session_state.interview_notes_version = 0

    if "interview_notes_content" not in st.session_state:
        st.session_state.interview_notes_content = None

    if "latest_transcribed_text" not in st.session_state:
        st.session_state.latest_transcribed_text = None

    if "audio_session_version" not in st.session_state:
        st.session_state.audio_session_version = 0

    if "summary_qa_content" not in st.session_state:
        st.session_state.summary_qa_content = None

    if "summary_qa_version" not in st.session_state:
        st.session_state.summary_qa_version = 0

    if "interview_questions_content" not in st.session_state:
        st.session_state.interview_questions_content = None

    if "interview_questions_version" not in st.session_state:
        st.session_state.interview_questions_version = 0

    if "langgraph_last_run_info" not in st.session_state:
        st.session_state.langgraph_last_run_info = None

    if "candidate_evaluation_result" not in st.session_state:
        st.session_state.candidate_evaluation_result = None

    if "candidate_evaluation_version" not in st.session_state:
        st.session_state.candidate_evaluation_version = 0



def set_current_mode(mode: str | None) -> None:
    _ensure_state()
    st.session_state.current_mode = mode
    if mode == "interview":
        st.session_state.audio_mode = "record"


def get_current_mode() -> str | None:
    _ensure_state()
    return st.session_state.current_mode


def set_audio_mode(audio_mode: str | None) -> None:
    _ensure_state()
    st.session_state.audio_mode = audio_mode


def get_audio_mode() -> str:
    _ensure_state()
    return st.session_state.audio_mode or "record"


def set_transcription_result(result: dict | None) -> None:
    _ensure_state()
    st.session_state.transcription_result = result


def get_transcription_result() -> dict | None:
    _ensure_state()
    return st.session_state.transcription_result


def clear_transcription_result() -> None:
    _ensure_state()
    st.session_state.transcription_result = None


def set_whisper_model_name(model_name: str) -> None:
    _ensure_state()
    st.session_state.whisper_model_name = model_name


def get_whisper_model_name() -> str:
    _ensure_state()
    return st.session_state.whisper_model_name or "base.en"


def get_interview_notes(default_value: str = "") -> str:
    _ensure_state()
    if st.session_state.interview_notes_content is None:
        st.session_state.interview_notes_content = default_value
    return st.session_state.interview_notes_content


def set_interview_notes(content: str) -> None:
    _ensure_state()
    st.session_state.interview_notes_content = content


def get_summary_qa_text(default_value: str = "") -> str:
    _ensure_state()
    if st.session_state.summary_qa_content is None:
        st.session_state.summary_qa_content = default_value
    return st.session_state.summary_qa_content


def set_summary_qa_text(content: str) -> None:
    _ensure_state()
    st.session_state.summary_qa_content = content


def get_live_interview_questions(default_value: str = "") -> str:
    _ensure_state()
    if st.session_state.interview_questions_content is None:
        st.session_state.interview_questions_content = default_value
    return st.session_state.interview_questions_content


def set_live_interview_questions(content: str) -> None:
    _ensure_state()
    st.session_state.interview_questions_content = content


def set_generated_interview_data(
    qa_text: str,
    questions_text: str,
    agent_name: str = "",
    evaluator_notes: str | None = None,
) -> None:
    _ensure_state()
    st.session_state.summary_qa_content = qa_text
    st.session_state.summary_qa_version += 1
    st.session_state.interview_questions_content = questions_text
    st.session_state.interview_questions_version += 1
    st.session_state.langgraph_last_run_info = {
        "agent": agent_name,
        "evaluator_notes": evaluator_notes,
    }


def get_langgraph_last_run_info() -> dict | None:
    _ensure_state()
    return st.session_state.langgraph_last_run_info


def set_candidate_evaluation(result: dict | None) -> None:
    _ensure_state()
    st.session_state.candidate_evaluation_result = result
    st.session_state.candidate_evaluation_version += 1


def get_candidate_evaluation() -> dict | None:
    _ensure_state()
    return st.session_state.candidate_evaluation_result


def reset_audio_state() -> None:
    _ensure_state()
    st.session_state.latest_transcribed_text = None
    st.session_state.last_processed_audio_hash = None
    st.session_state.last_processed_upload_hash = None
    st.session_state.transcription_result = None
    st.session_state.candidate_evaluation_result = None
    st.session_state.audio_session_version += 1


def clear_current_mode() -> None:
    _ensure_state()
    st.session_state.current_mode = None
    st.session_state.audio_mode = "record"
    st.session_state.transcription_result = None
    st.session_state.last_processed_audio_hash = None
    st.session_state.last_processed_upload_hash = None
    st.session_state.interview_notes_content = None
    st.session_state.interview_notes_version = 0
    st.session_state.latest_transcribed_text = None
    st.session_state.candidate_evaluation_result = None
    st.session_state.audio_session_version = 0










