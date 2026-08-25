import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

# Suppress internal Google GenAI SDK AFC warnings
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

from langgraph.graph import END, START, StateGraph

# Free, ultra-fast Gemini models ordered by speed and availability
DEFAULT_GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash-latest",
]


class InterviewGraphState(TypedDict):
    instructions: str
    job_requirement: str
    selected_agent: str
    api_key: Optional[str]
    gemini_output: Optional[str]
    final_output: Optional[str]
    extracted_qa_text: str
    extracted_questions_text: str
    error: Optional[str]


def _extract_questions_only(text: str) -> str:
    """Extract only the primary and follow-up questions into a structured interview list."""
    lines = text.strip().split("\n")
    questions = []
    q_counter = 1
    in_answer_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for section header
        if stripped.startswith(("#", "---")) or (stripped.startswith("**") and stripped.endswith("**") and any(k in stripped.lower() for k in ["section", "part", "topic", "domain"])):
            header = stripped.lstrip("#- *").rstrip("* -").strip()
            if header and len(header) < 80 and not any(k in header.lower() for k in ["answer", "model"]):
                questions.append(f"\n--- [{header}] ---")
                in_answer_block = False
                continue

        # Detect Answer start: A1:, Answer 1:, **A1:**, **Answer:**, etc.
        if re.match(r"^(?:(?:\*\*|\#\#|\*|\-)?\s*(?:A\d+|Answer(?:\s*\d+)?)\s*[:.)-]?\s*(?:\*\*)?)", stripped, re.IGNORECASE):
            in_answer_block = True
            continue

        # Detect Question start: Q1:, Q1 (Follow-up):, Question 1:, **Q1:**, 1., etc.
        q_match = re.match(
            r"^(?:(?:\*\*|\#\#|\*|\-)?\s*(?:Q\d+(?:\s*\([^)]+\)|\.\d+)?|Question\s*\d+(?:\s*\([^)]+\)|\.\d+)?|\d+[\.\)])\s*[:.)-]?\s*(?:\*\*)?)\s*(.+)$",
            stripped,
            re.IGNORECASE,
        )
        if q_match:
            in_answer_block = False
            q_text = q_match.group(1).strip().strip("*").strip()
            # Verify not an answer line
            if not re.match(r"^(?:A\d+|Answer\s*\d+)", q_text, re.IGNORECASE) and len(q_text) > 8:
                is_followup = bool(re.search(r"follow-?up", stripped, re.IGNORECASE))
                clean_q = re.sub(r"^follow-?up\s*[:.)-]?\s*", "", q_text, flags=re.IGNORECASE).strip()
                prefix = "  --> Follow-up:" if is_followup else f"{q_counter}."
                questions.append(f"{prefix} {clean_q}")
                if not is_followup:
                    q_counter += 1
                continue

        # Detect bulleted follow-up question
        if stripped.startswith(("- ", "* ", "> ")) and ("?" in stripped or "follow" in stripped.lower()):
            in_answer_block = False
            clean_q = re.sub(r"^[-*>]\s*", "", stripped).strip().strip("*")
            is_followup = "follow" in stripped.lower()
            clean_q = re.sub(r"^follow-?up\s*[:.)-]?\s*", "", clean_q, flags=re.IGNORECASE).strip()
            if is_followup:
                questions.append(f"  --> Follow-up: {clean_q}")
            else:
                questions.append(f"{q_counter}. {clean_q}")
                q_counter += 1
            continue

        # If line ends with '?' and we are not in an answer block
        if stripped.endswith("?") and not in_answer_block and len(stripped) > 15:
            clean_q = re.sub(r"^(?:(?:\*\*|\#\#|\*|\-)?\s*(?:Q\d+|Question\s*\d+|\d+[\.\)])\s*[:.)-]?\s*(?:\*\*)?)\s*", "", stripped).strip().strip("*")
            questions.append(f"{q_counter}. {clean_q}")
            q_counter += 1
            continue

    if len([q for q in questions if not q.startswith("\n---")]) >= 2:
        return "\n".join(questions).strip()

    return text


def _get_active_models(client=None) -> List[str]:
    """Dynamically discover free/flash models for this API key, or use fast defaults."""
    discovered = []
    if client:
        try:
            for m in client.models.list():
                name = m.name.replace("models/", "") if hasattr(m, "name") else str(m)
                if "gemini" in name.lower() and not any(k in name.lower() for k in ["embed", "imagen", "aqa", "bison"]):
                    if "flash" in name.lower():
                        discovered.append(name)
        except Exception:
            pass

    for fb in DEFAULT_GEMINI_MODELS:
        if fb not in discovered:
            discovered.append(fb)

    return discovered


def _call_gemini_llm(
    instructions: str,
    job_requirement: str,
    api_key: Optional[str] = None,
) -> str:
    if not api_key or not api_key.strip():
        raise ValueError("Google Gemini API Key is missing. Please provide your Gemini API Key in the UI above.")

    # Dynamically build prompt using user instructions and job requirement without hardcoding
    prompt_sections = [
        "You are an expert Technical Interview Creator and Lead Technical Bar Raiser.",
        "Your task is to generate technical interview questions and comprehensive model answers tailored to the specified job requirement and user instructions.",
    ]

    if instructions and instructions.strip():
        prompt_sections.append(f"\n### User Instructions:\n{instructions.strip()}")

    if job_requirement and job_requirement.strip():
        prompt_sections.append(f"\n### Job Requirement / Technical Profile:\n{job_requirement.strip()}")

    prompt_sections.append(
        "\n### CRITICAL INTERVIEW VOLUME & 1-HOUR DURATION REQUIREMENTS:\n"
        "- This is a comprehensive ONE-HOUR technical interview. Generating only 3-4 questions is UNACCEPTABLE.\n"
        "- You MUST generate at least 10 to 12 distinct Primary Technical Questions, each paired with its own deep-dive Follow-up Probe Question (totaling 20+ questions across the session).\n"
        "- Ensure full 360-degree coverage of all core technical skills, architectural patterns, concurrency/internals, database design, and troubleshooting specified in the job requirement.\n\n"
        "### OUTPUT FORMAT (YOU MUST GENERATE BOTH SECTIONS EXACTLY AS SPECIFIED):\n\n"
        "=== SECTION 1: INTERVIEW QUESTIONS ONLY ===\n"
        "List all 10 to 12 primary technical interview questions and their deep-dive follow-up probe questions.\n"
        "DO NOT include any answers, explanations, or solutions in Section 1.\n"
        "Format Section 1 strictly as:\n"
        "1. <Primary Technical Question 1>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 1>\n"
        "2. <Primary Technical Question 2>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 2>\n"
        "3. <Primary Technical Question 3>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 3>\n"
        "4. <Primary Technical Question 4>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 4>\n"
        "5. <Primary Technical Question 5>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 5>\n"
        "6. <Primary Technical Question 6>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 6>\n"
        "7. <Primary Technical Question 7>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 7>\n"
        "8. <Primary Technical Question 8>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 8>\n"
        "9. <Primary Technical Question 9>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 9>\n"
        "10. <Primary Technical Question 10>\n"
        "   --> Follow-up: <Deep-Dive Technical Follow-up Question 10>\n\n"
        "=== SECTION 2: FULL QUESTIONS & MODEL ANSWERS ===\n"
        "Provide the complete technical questions, follow-up probe questions, and concise technical model answers (2-3 sentences each for fast generation) for all questions for the interviewer's reference.\n"
        "Format Section 2 strictly as:\n"
        "Q1: <Primary Question 1>\n"
        "A1: <Concise Model Answer 1>\n"
        "Q1 (Follow-up): <Deep-Dive Follow-up Question 1>\n"
        "A1 (Follow-up): <Concise Model Answer 1>\n\n"
        "Q2: <Primary Question 2>\n"
        "A2: <Concise Model Answer 2>\n"
        "Q2 (Follow-up): <Deep-Dive Follow-up Question 2>\n"
        "A2 (Follow-up): <Concise Model Answer 2>\n"
        "...\n"
        "Q10: <Primary Question 10>\n"
        "A10: <Concise Model Answer 10>\n"
        "Q10 (Follow-up): <Deep-Dive Follow-up Question 10>\n"
        "A10 (Follow-up): <Concise Model Answer 10>\n"
    )

    prompt = "\n".join(prompt_sections)

    last_err = None

    # 1. Primary: Use Fast Flash models via google.genai SDK
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(timeout=120000),
        )

        candidate_models = _get_active_models(client)

        for model_name in candidate_models:
            try:
                # Fast direct generation with zero thinking delay and 8192 token window
                config = types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    return str(response.text)
            except Exception as e:
                last_err = e
                try:
                    config_fallback = types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=8192,
                    )
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config_fallback,
                    )
                    if response and response.text:
                        return str(response.text)
                except Exception as e2:
                    last_err = e2
                    continue
    except Exception as e:
        last_err = e

    # 2. Fallback: Try langchain_google_genai
    candidate_models = _get_active_models(None)
    for model_name in candidate_models:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key.strip(),
                temperature=0.3,
                timeout=120,
                max_retries=1,
            )
            response = llm.invoke(prompt)
            if response and response.content:
                return str(response.content)
        except Exception as e:
            last_err = e
            continue


    raise RuntimeError(f"Failed to generate interview questions via Google Gemini API: {last_err}")


# --- LangGraph Nodes ---


def gemini_node(state: InterviewGraphState) -> Dict[str, Any]:
    instructions = state.get("instructions", "")
    job_req = state.get("job_requirement", "")
    api_key = state.get("api_key", None)
    output = _call_gemini_llm(instructions, job_req, api_key=api_key)
    return {
        "gemini_output": output,
        "final_output": output,
    }


def format_output_node(state: InterviewGraphState) -> Dict[str, Any]:
    final_text = state.get("final_output", "") or ""
    extracted_questions = ""
    extracted_qa = ""

    if "=== SECTION 1: INTERVIEW QUESTIONS ONLY ===" in final_text and "=== SECTION 2: FULL QUESTIONS & MODEL ANSWERS ===" in final_text:
        parts = final_text.split("=== SECTION 2: FULL QUESTIONS & MODEL ANSWERS ===")
        sec1_raw = parts[0].split("=== SECTION 1: INTERVIEW QUESTIONS ONLY ===")[1].strip()
        sec2_raw = parts[1].strip()
        extracted_questions = sec1_raw
        extracted_qa = sec2_raw
    elif "=== SECTION 1:" in final_text and "=== SECTION 2:" in final_text:
        parts = re.split(r"===\s*SECTION\s*2.*===", final_text, flags=re.IGNORECASE)
        sec1_raw = re.split(r"===\s*SECTION\s*1.*===", parts[0], flags=re.IGNORECASE)[-1].strip()
        sec2_raw = parts[1].strip() if len(parts) > 1 else ""
        extracted_questions = sec1_raw
        extracted_qa = sec2_raw
    else:
        extracted_qa = final_text
        extracted_questions = _extract_questions_only(final_text)

    if not extracted_questions or len(extracted_questions) < 20:
        extracted_questions = _extract_questions_only(final_text)

    return {
        "extracted_qa_text": extracted_qa.strip(),
        "extracted_questions_text": extracted_questions.strip(),
    }


def create_interview_graph():
    workflow = StateGraph(InterviewGraphState)

    workflow.add_node("gemini_node", gemini_node)
    workflow.add_node("format_output_node", format_output_node)

    workflow.add_edge(START, "gemini_node")
    workflow.add_edge("gemini_node", "format_output_node")
    workflow.add_edge("format_output_node", END)

    return workflow.compile()


# Module-level cached graph
_COMPILED_GRAPH = None


def run_interview_graph(
    instructions: str,
    job_requirement: str,
    selected_agent: str = "Gemini (Google)",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = create_interview_graph()

    initial_state: InterviewGraphState = {
        "instructions": instructions,
        "job_requirement": job_requirement,
        "selected_agent": selected_agent,
        "api_key": api_key,
        "gemini_output": None,
        "final_output": None,
        "extracted_qa_text": "",
        "extracted_questions_text": "",
        "error": None,
    }

    result = _COMPILED_GRAPH.invoke(initial_state)
    return result


def evaluate_candidate_interview(
    transcript: str,
    job_requirement: str = "",
    instructions: str = "",
    selected_agent: str = "Gemini (Google)",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    if not transcript or not transcript.strip():
        return {
            "eligibility_status": "PENDING EVALUATION",
            "selection_probability": 0,
            "report_text": "No speech audio detected in transcript to evaluate.",
        }

    if not api_key or not api_key.strip():
        return {
            "eligibility_status": "API KEY REQUIRED",
            "selection_probability": 0,
            "report_text": "Please provide your Google Gemini API Key in the UI above to evaluate the candidate's spoken response.",
        }

    eval_prompt_parts = [
        "You are an expert Technical Interview Evaluator.",
        "The candidate has completed an interview where they spoke interview questions followed by their answers.",
        f"\nCANDIDATE SPOKEN TRANSCRIPT:\n{transcript.strip()}",
    ]

    if job_requirement and job_requirement.strip():
        eval_prompt_parts.append(f"\nJOB REQUIREMENT:\n{job_requirement.strip()}")

    if instructions and instructions.strip():
        eval_prompt_parts.append(f"\nUSER INSTRUCTIONS / EVALUATION CRITERIA:\n{instructions.strip()}")

    eval_prompt_parts.append(
        "\nYOUR OBJECTIVE:\n"
        "1. Determine if the candidate is eligible for selection ('ELIGIBLE FOR SELECTION', 'STRONGLY RECOMMENDED', 'BORDERLINE / CONDITIONAL', or 'NOT ELIGIBLE').\n"
        "2. Calculate the selection probability percentage (0% to 100%).\n"
        "3. Provide a Question-by-Question technical evaluation of the candidate's answers based on the job requirements.\n"
        "4. Summarize candidate strengths, technical gaps, and final hiring recommendation.\n\n"
        "FORMAT YOUR EXACT OUTPUT AS FOLLOWS:\n"
        "ELIGIBILITY: <status>\n"
        "PROBABILITY: <XX>%\n\n"
        "### Question-by-Question Technical Assessment\n<assessment>\n\n"
        "### Key Strengths & Technical Gaps\n<strengths and gaps>\n\n"
        "### Final Summary & Hiring Recommendation\n<recommendation>"
    )

    eval_prompt = "\n".join(eval_prompt_parts)

    raw_response = None
    last_err = None

    # 1. Fast Flash model via google.genai Client
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(timeout=60000),
        )
        candidate_models = _get_active_models(client)
        for model_name in candidate_models:
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=2048,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                if res and res.text:
                    raw_response = str(res.text)
                    break
            except Exception as e:
                last_err = e
                try:
                    res_fallback = client.models.generate_content(
                        model=model_name,
                        contents=eval_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=2048,
                        ),
                    )
                    if res_fallback and res_fallback.text:
                        raw_response = str(res_fallback.text)
                        break
                except Exception as e2:
                    last_err = e2
                    continue
    except Exception as e:
        last_err = e

    # 2. Fallback to langchain_google_genai
    if not raw_response:
        candidate_models = _get_active_models(None)
        for model_name in candidate_models:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key.strip(),
                    temperature=0.3,
                    timeout=60,
                    max_retries=1,
                )
                res = llm.invoke(eval_prompt)
                if res and res.content:
                    raw_response = str(res.content)
                    break
            except Exception as e:
                last_err = e
                continue

    if not raw_response:
        return {
            "eligibility_status": "EVALUATION FAILED",
            "selection_probability": 0,
            "report_text": f"Error calling Google Gemini API for candidate evaluation: {last_err}",
        }

    # Parse Eligibility, Probability, and Report Text from actual Gemini output
    eligibility = "ELIGIBLE FOR SELECTION"
    probability = 50

    elig_match = re.search(r"ELIGIBILITY:\s*([^\n\r]+)", raw_response, re.IGNORECASE)
    if elig_match:
        eligibility = elig_match.group(1).strip().strip("*")

    prob_match = re.search(r"PROBABILITY:\s*(\d{1,3})\s*%", raw_response, re.IGNORECASE)
    if prob_match:
        try:
            probability = max(0, min(100, int(prob_match.group(1))))
        except Exception:
            probability = 50

    # Extract report body
    report_body = raw_response
    if "### Question-by-Question" in raw_response:
        report_body = raw_response[raw_response.find("### Question-by-Question") :].strip()

    return {
        "eligibility_status": eligibility,
        "selection_probability": probability,
        "report_text": report_body,
        "raw_response": raw_response,
    }
