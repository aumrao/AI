import os
import re
import json
import urllib.request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage


class HighlightClip(BaseModel):
    start_time: float = Field(description="Start timestamp in seconds where the key point begins")
    end_time: float = Field(description="End timestamp in seconds where the key point ends")
    title: str = Field(description="Short chapter title (3-6 words) describing this highlight")
    reason: str = Field(description="Why this segment is important to the overall understanding")
    importance_score: int = Field(default=8, description="Importance rating from 1 to 10")


class VideoSummaryResult(BaseModel):
    title: str = Field(description="Concise, informative title summarizing the video's content")
    overview: str = Field(description="2-3 paragraph comprehensive executive summary")
    key_takeaways: List[str] = Field(description="List of 4-7 key bullet point takeaways")
    highlights: List[HighlightClip] = Field(
        description="Chronologically ordered list of non-overlapping highlight segments that summarize the video"
    )


def get_installed_ollama_models() -> List[str]:
    """Queries local Ollama service for installed text models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = [
                m.get("name") for m in data.get("models", [])
                if not any(x in m.get("name", "").lower() for x in ["embed", "nomic", "bge"])
            ]
            if models:
                return models
    except Exception:
        pass
    return ["mistral:latest", "llama3.2:latest", "qwen2.5:latest", "gemma2:latest"]


def extract_json_from_text(text: str) -> dict:
    """Robustly extracts JSON dictionary from LLM response text."""
    # Strip <think>...</think> reasoning tags if present (e.g., DeepSeek-R1)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Try markdown json block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try finding outer braces
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return json.loads(cleaned[first_brace:last_brace + 1])

    return json.loads(cleaned)


def get_llm_instance(provider: str, api_key: str, model_name: Optional[str] = None, temperature: float = 0.2):
    """
    Initializes a LangChain ChatModel based on provider and user-supplied API key.
    """
    provider = provider.lower()

    if provider in ["ollama", "local"]:
        from langchain_openai import ChatOpenAI
        model = model_name or "mistral:latest"
        return ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model,
            temperature=temperature,
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        model = model_name or "meta-llama/llama-3.3-70b-instruct:free"
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or "free",
            model=model,
            temperature=temperature,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        model = model_name or "llama-3.3-70b-versatile"
        return ChatGroq(
            model_name=model,
            groq_api_key=api_key,
            temperature=temperature,
        )
    elif provider in ["google", "gemini"]:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model_name or "gemini-3.7-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def format_transcript_with_timestamps(segments: List[Dict[str, Any]], max_chars: int = 60000) -> str:
    """
    Formats transcript segments into a timestamped readable text block for LLM prompt.
    """
    lines = []
    total_len = 0
    for seg in segments:
        start_s = seg.get("start", 0.0)
        end_s = seg.get("end", 0.0)
        text = seg.get("text", "").strip()
        line = f"[{start_s:0.1f}s - {end_s:0.1f}s]: {text}"
        total_len += len(line) + 1
        if total_len > max_chars:
            lines.append("... [transcript truncated due to length limit]")
            break
        lines.append(line)
    return "\n".join(lines)


def optimize_highlight_boundaries(
    highlights: List[HighlightClip],
    total_duration: float,
    min_clip_duration: float = 3.0,
    buffer_seconds: float = 0.5,
    merge_gap_threshold: float = 2.0
) -> List[HighlightClip]:
    """
    Cleans up, pads, and merges overlapping or near-adjacent highlight intervals.
    """
    if not highlights:
        return []

    # Sort chronologically by start time
    sorted_hl = sorted(highlights, key=lambda h: h.start_time)
    processed = []

    for hl in sorted_hl:
        # Add small buffer padding for natural conversational flow
        s = max(0.0, hl.start_time - buffer_seconds)
        e = min(total_duration if total_duration > 0 else 999999.0, hl.end_time + buffer_seconds)

        # Ensure minimum duration
        if (e - s) < min_clip_duration:
            e = min(total_duration if total_duration > 0 else 999999.0, s + min_clip_duration)

        if s >= e:
            continue

        if not processed:
            processed.append(HighlightClip(
                start_time=round(s, 2),
                end_time=round(e, 2),
                title=hl.title,
                reason=hl.reason,
                importance_score=hl.importance_score
            ))
        else:
            prev = processed[-1]
            # Merge if overlapping or within merge_gap_threshold
            if s <= (prev.end_time + merge_gap_threshold):
                prev.end_time = round(max(prev.end_time, e), 2)
                prev.title = f"{prev.title} & {hl.title}"
                prev.reason = f"{prev.reason} | {hl.reason}"
                prev.importance_score = max(prev.importance_score, hl.importance_score)
            else:
                processed.append(HighlightClip(
                    start_time=round(s, 2),
                    end_time=round(e, 2),
                    title=hl.title,
                    reason=hl.reason,
                    importance_score=hl.importance_score
                ))

    return processed


def generate_video_summary(
    transcript_segments: List[Dict[str, Any]],
    total_duration: float,
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
    target_summary_ratio: float = 0.30,
    custom_focus_prompt: Optional[str] = None
) -> VideoSummaryResult:
    """
    Executes the LLM prompt to identify key moments and synthesize summary text.
    """
    target_seconds = total_duration * target_summary_ratio
    formatted_transcript = format_transcript_with_timestamps(transcript_segments)

    custom_focus = f"\nUser Special Focus: {custom_focus_prompt}" if custom_focus_prompt else ""

    json_schema_example = json.dumps({
        "title": "Concise Summary Title",
        "overview": "2-3 paragraph executive summary of the entire video.",
        "key_takeaways": ["Takeaway point 1", "Takeaway point 2", "Takeaway point 3"],
        "highlights": [
            {
                "start_time": 12.5,
                "end_time": 45.0,
                "title": "Key Concept Introduction",
                "reason": "Explains the foundational architecture and problem statement.",
                "importance_score": 9
            }
        ]
    }, indent=2)

    system_prompt = (
        "You are an expert AI video editor and summarizer.\n"
        "Your mission is to analyze a timestamped transcript of a video and select the MOST IMPORTANT "
        "segments to create an engaging, highly informative, and coherent condensed summary video.\n\n"
        "CRITICAL RULES FOR CLIP SELECTION:\n"
        f"1. Total video duration: {total_duration:.1f} seconds (~{total_duration/60:.1f} mins).\n"
        f"2. Target summary length: Approximately {target_seconds:.1f} seconds ({target_summary_ratio*100:.0f}% of original).\n"
        "3. Choose only the most essential key moments where crucial explanations, core insights, demos, or main conclusions happen.\n"
        "4. Exact Timestamps: Use the precise start_time and end_time (in seconds) matching the transcript segments.\n"
        "5. Flow & Coherence: Select complete thoughts and statements so the cut video sounds natural.\n"
        "6. Chronological Order: Ensure the highlights are ordered from beginning to end of the video.\n"
        "7. Provide a clear Executive Overview, bullet-point Takeaways, and informative Titles for each highlight.\n"
        "8. Output MUST be valid JSON adhering exactly to this structure:\n"
        f"```json\n{json_schema_example}\n```\n"
        f"{custom_focus}"
    )

    human_prompt = (
        f"Here is the timestamped video transcript:\n\n"
        f"```text\n{formatted_transcript}\n```\n\n"
        f"Analyze the transcript and return the JSON VideoSummaryResult with optimal highlight timestamps totaling around {target_seconds:.1f} seconds."
    )

    provider_clean = provider.lower()

    if provider_clean in ["google", "gemini"]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        models_to_try = [
            model_name or "gemini-3.7-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview"
        ]

        last_error = None
        result = None
        for m in list(dict.fromkeys(models_to_try)):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=[human_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VideoSummaryResult,
                        system_instruction=system_prompt,
                        temperature=0.2,
                    ),
                )
                if response and response.text:
                    result = VideoSummaryResult.model_validate_json(response.text)
                    break
            except Exception as e:
                last_error = e
                continue

        if result is None:
            raise RuntimeError(f"Google Gemini model execution failed: {last_error}")

    elif provider_clean in ["ollama", "local"]:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model_name or "mistral:latest",
            temperature=0.2,
        )
        messages = [
            SystemMessage(content=system_prompt + "\nRespond with raw JSON only."),
            HumanMessage(content=human_prompt),
        ]
        resp = llm.invoke(messages)
        parsed_data = extract_json_from_text(resp.content)
        result = VideoSummaryResult.model_validate(parsed_data)

    else:
        # Groq, OpenRouter, OpenAI
        llm = get_llm_instance(provider=provider, api_key=api_key, model_name=model_name)
        try:
            structured_llm = llm.with_structured_output(VideoSummaryResult)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
            result = structured_llm.invoke(messages)
        except Exception:
            messages = [
                SystemMessage(content=system_prompt + "\nRespond with valid JSON only."),
                HumanMessage(content=human_prompt),
            ]
            resp = llm.invoke(messages)
            parsed_data = extract_json_from_text(resp.content)
            result = VideoSummaryResult.model_validate(parsed_data)

    # Post-process and optimize highlight intervals
    result.highlights = optimize_highlight_boundaries(
        highlights=result.highlights,
        total_duration=total_duration
    )

    return result
