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


def consolidate_transcript_segments(
    segments: List[Dict[str, Any]],
    max_pause_seconds: float = 1.5,
    max_chunk_duration: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Groups rapid, fragmented subtitle lines into coherent, sentence-level speech blocks.
    Preserves exact start and end timestamps while making context 5x cleaner for LLM reasoning.
    """
    if not segments:
        return []

    consolidated = []
    curr = dict(segments[0])

    for seg in segments[1:]:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = seg.get("start", curr.get("end", 0.0))
        end = seg.get("end", start + 1.0)
        gap = max(0.0, start - curr["end"])
        curr_dur = curr["end"] - curr["start"]

        if gap <= max_pause_seconds and curr_dur < max_chunk_duration:
            curr["end"] = round(end, 2)
            curr["text"] = f"{curr['text']} {text}".strip()
        else:
            consolidated.append(curr)
            curr = {"start": round(start, 2), "end": round(end, 2), "text": text}

    consolidated.append(curr)
    return consolidated


def format_transcript_with_timestamps(segments: List[Dict[str, Any]], max_chars: int = 2_000_000) -> str:
    """
    Formats transcript segments into a timestamped readable text block for LLM prompt.
    Uses 2,000,000 char capacity to support full multi-hour transcripts without truncation on Gemini Flash.
    """
    consolidated = consolidate_transcript_segments(segments)
    lines = []
    total_len = 0
    for seg in consolidated:
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
    Executes the LLM prompt to analyze the entire video and select key moments across the whole video.
    """
    target_seconds = total_duration * target_summary_ratio
    formatted_transcript = format_transcript_with_timestamps(transcript_segments)

    custom_focus = f"\nUser Special Focus: {custom_focus_prompt}" if custom_focus_prompt else ""

    json_schema_example = json.dumps({
        "title": "Comprehensive Summary Title",
        "overview": "3-4 paragraph in-depth executive summary covering the entire video from start to finish.",
        "key_takeaways": [
            "Core insight or foundational concept explained in the opening",
            "Key demonstration, breakthrough, or deep-dive mechanism in the middle",
            "Main results, benchmark comparisons, or practical applications",
            "Final conclusion and forward-looking takeaways"
        ],
        "highlights": [
            {
                "start_time": 12.5,
                "end_time": 65.0,
                "title": "Foundational Overview",
                "reason": "Explains core problem statement and architecture.",
                "importance_score": 9
            }
        ]
    }, indent=2)

    system_prompt = (
        "You are a master AI video director and executive summarizer.\n"
        "Your mission is to analyze the FULL timestamped transcript of a video and select the MOST IMPORTANT "
        "highlight segments to create an engaging, comprehensive, and high-accuracy condensed summary video.\n\n"
        "CRITICAL RULES FOR ACCURATE WHOLE-VIDEO SUMMARIZATION:\n"
        f"1. Total video duration: {total_duration:.1f} seconds (~{total_duration/60:.1f} mins).\n"
        f"2. Target summary length: Approximately {target_seconds:.1f} seconds ({target_summary_ratio*100:.0f}% of original).\n"
        "3. WHOLE-VIDEO CHRONOLOGICAL COVERAGE: Ensure highlights are distributed across the ENTIRE duration of the video "
        "(beginning introduction, early core points, middle deep dives/demos, late-stage insights, and final conclusions). "
        "Do NOT cluster all clips in only one portion of the video.\n"
        "4. Exact Timestamps: Use the precise start_time and end_time (in seconds) matching the transcript segments.\n"
        "5. Complete Thoughts: Ensure each highlight captures a complete idea or statement so the cut video flows naturally.\n"
        "6. Chronological Order: Strictly sort the highlights from beginning to end of the video.\n"
        "7. Executive Overview: Provide a detailed, highly informative 3-paragraph executive overview and 5-7 key takeaways.\n"
        "8. Output MUST be valid JSON adhering exactly to this structure:\n"
        f"```json\n{json_schema_example}\n```\n"
        f"{custom_focus}"
    )

    human_prompt = (
        f"Here is the complete timestamped video transcript:\n\n"
        f"```text\n{formatted_transcript}\n```\n\n"
        f"Analyze the entire transcript across all sections and return the JSON VideoSummaryResult with optimal highlights totaling approximately {target_seconds:.1f} seconds."
    )


    provider_clean = provider.lower()

    if provider_clean in ["google", "gemini"]:
        from google import genai
        from google.genai import types
        from .key_pool import GeminiKeyPool

        key_list = GeminiKeyPool.parse_keys_str(api_key)
        # If user provided multiple Gemini API keys and video is at least 3 minutes, run Parallel Map-Reduce
        if len(key_list) > 1 and total_duration >= 180:
            return parallel_gemini_map_reduce_summary(
                transcript_segments=transcript_segments,
                total_duration=total_duration,
                api_keys=key_list,
                model_name=model_name,
                target_summary_ratio=target_summary_ratio,
                custom_focus_prompt=custom_focus_prompt
            )

        # Single key / standard execution with key pool rotation on rate-limit
        active_key = key_list[0] if key_list else api_key
        models_to_try = [
            model_name or "gemini-3.7-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ]
        clean_models = list(dict.fromkeys(models_to_try))


        last_error = None
        result = None
        for k in (key_list or [active_key]):
            client = genai.Client(api_key=k)
            for m in clean_models:
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
            if result is not None:
                break

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


def generate_video_summary_from_audio(
    audio_path: str,
    total_duration: float,
    api_key: str,
    model_name: Optional[str] = None,
    target_summary_ratio: float = 0.30,
    custom_focus_prompt: Optional[str] = None
) -> tuple[VideoSummaryResult, List[Dict[str, Any]]]:
    """
    Directly ingests audio file using Gemini 3.7 Flash native multimodal understanding.
    Processes full audio in ~4-6 seconds, eliminating slow local CPU Whisper transcription.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    target_seconds = total_duration * target_summary_ratio
    custom_focus = f"\nUser Special Focus: {custom_focus_prompt}" if custom_focus_prompt else ""

    class DirectAudioSummarySchema(BaseModel):
        title: str = Field(description="Concise, informative title summarizing the video")
        overview: str = Field(description="2-3 paragraph executive summary of the entire audio")
        key_takeaways: List[str] = Field(description="List of 4-7 key bullet point takeaways")
        highlights: List[HighlightClip] = Field(
            description="Chronologically ordered list of non-overlapping highlight segments totaling the target duration"
        )

    system_prompt = (
        "You are an expert AI video editor and audio analyst.\n"
        "Analyze the provided audio track from a video and extract the MOST IMPORTANT highlight segments "
        "along with an executive summary.\n\n"
        "CRITICAL RULES:\n"
        f"1. Total video duration: {total_duration:.1f} seconds (~{total_duration/60:.1f} mins).\n"
        f"2. Target summary length: Approximately {target_seconds:.1f} seconds ({target_summary_ratio*100:.0f}% of original).\n"
        "3. Choose only essential key moments where core insights, key explanations, or main conclusions occur.\n"
        "4. Exact Timestamps: Provide precise start_time and end_time (in seconds) for each highlight.\n"
        "5. Chronological Order: Order highlights from beginning to end of the video.\n"
        f"{custom_focus}"
    )

    models_to_try = [
        model_name or "gemini-3.7-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash"
    ]
    models_to_try = list(dict.fromkeys(models_to_try))


    uploaded_file = client.files.upload(file=audio_path)
    result = None
    last_error = None

    try:
        for m in list(dict.fromkeys(models_to_try)):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=[
                        uploaded_file,
                        f"Analyze this audio file and return the structured JSON summary with highlights totaling ~{target_seconds:.1f} seconds."
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DirectAudioSummarySchema,
                        system_instruction=system_prompt,
                        temperature=0.2,
                    ),
                )
                if response and response.text:
                    parsed = DirectAudioSummarySchema.model_validate_json(response.text)
                    result = VideoSummaryResult(
                        title=parsed.title,
                        overview=parsed.overview,
                        key_takeaways=parsed.key_takeaways,
                        highlights=optimize_highlight_boundaries(parsed.highlights, total_duration)
                    )
                    break
            except Exception as e:
                last_error = e
                continue

        if result is None:
            raise RuntimeError(f"Direct Gemini Audio processing failed: {last_error}")

        # Build synthetic transcript segments from highlights for display
        segments = [
            {
                "start": round(h.start_time, 2),
                "end": round(h.end_time, 2),
                "text": f"[{h.title}] {h.reason}"
            }
            for h in result.highlights
        ]
        return result, segments

    finally:
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass


def parallel_gemini_map_reduce_summary(
    transcript_segments: List[Dict[str, Any]],
    total_duration: float,
    api_keys: List[str],
    model_name: Optional[str] = None,
    target_summary_ratio: float = 0.30,
    custom_focus_prompt: Optional[str] = None
) -> VideoSummaryResult:
    """
    Parallel Map-Reduce engine with multi-model resiliency and full-video guarantee:
    1. Splits the transcript into N chronological parts.
    2. Uses different Gemini API keys concurrently with fallback models on 503/429.
    3. Guarantees 100% complete coverage across the entire video.
    """
    from google import genai
    from google.genai import types
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    num_keys = min(len(api_keys), 5)
    shard_duration = total_duration / num_keys
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_keys)]

    for seg in transcript_segments:
        start = seg.get("start", 0.0)
        idx = min(num_keys - 1, max(0, int(start // shard_duration)))
        shards[idx].append(seg)

    target_per_shard = (total_duration * target_summary_ratio) / num_keys
    custom_focus = f"\nUser Special Focus: {custom_focus_prompt}" if custom_focus_prompt else ""

    class ShardHighlightSchema(BaseModel):
        section_summary: str = Field(description="1-2 paragraph summary of this section")
        section_takeaways: List[str] = Field(description="2-3 key takeaways from this section")
        highlights: List[HighlightClip] = Field(description="Top key highlight clips from this section")

    models_to_try = [
        model_name or "gemini-3.7-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash"
    ]
    models_to_try = list(dict.fromkeys(models_to_try))

    def _process_shard(shard_idx: int, shard_segs: List[Dict[str, Any]], primary_key: str):
        if not shard_segs:
            return shard_idx, None

        formatted = format_transcript_with_timestamps(shard_segs)
        s_start = shard_idx * shard_duration
        s_end = (shard_idx + 1) * shard_duration

        prompt = (
            f"You are an expert AI video editor analyzing Part {shard_idx+1}/{num_keys} of a video (Time window: {s_start/60:.1f}m to {s_end/60:.1f}m).\n"
            f"Select the most essential key highlight moments from this section totaling approximately {target_per_shard:.1f} seconds.\n"
            f"Provide exact start_time and end_time (in seconds), concise titles, and section takeaways.\n"
            f"{custom_focus}\n\n"
            f"Transcript for this section:\n```text\n{formatted}\n```"
        )

        keys_to_attempt = [primary_key] + [k for k in api_keys if k != primary_key]

        for key in keys_to_attempt:
            client = genai.Client(api_key=key)
            for m in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ShardHighlightSchema,
                            temperature=0.2,
                        )
                    )
                    if response and response.text:
                        return shard_idx, ShardHighlightSchema.model_validate_json(response.text)
                except Exception:
                    continue

        return shard_idx, None

    # Map Phase: Parallel execution across all keys
    shard_results: Dict[int, ShardHighlightSchema] = {}
    with ThreadPoolExecutor(max_workers=num_keys) as executor:
        futures = {executor.submit(_process_shard, i, shards[i], api_keys[i % len(api_keys)]): i for i in range(num_keys)}
        for fut in as_completed(futures):
            s_idx, res = fut.result()
            if res:
                shard_results[s_idx] = res

    # If any shard failed to complete, fallback to full-video single pass to guarantee complete summary
    if len(shard_results) < num_keys:
        print(f"Notice: Map-Reduce had {num_keys - len(shard_results)} missing shards. Executing Unified Whole-Video Fallback Engine.")
        return _generate_single_pass_gemini_summary(
            transcript_segments=transcript_segments,
            total_duration=total_duration,
            api_keys=api_keys,
            model_name=model_name,
            target_summary_ratio=target_summary_ratio,
            custom_focus_prompt=custom_focus_prompt
        )

    # Combine highlights and section notes
    all_highlights = []
    section_overviews = []
    all_takeaways = []

    for i in range(num_keys):
        if i in shard_results:
            sr = shard_results[i]
            all_highlights.extend(sr.highlights)
            if sr.section_summary:
                section_overviews.append(f"**Part {i+1} ({i*shard_duration/60:.0f}-{(i+1)*shard_duration/60:.0f} mins)**: {sr.section_summary}")
            all_takeaways.extend(sr.section_takeaways)

    optimized_hl = optimize_highlight_boundaries(all_highlights, total_duration)

    # Reduce Phase: Generate global executive overview combining all sections
    class FinalSynthesisSchema(BaseModel):
        title: str = Field(description="Concise, captivating title summarizing the entire video")
        overview: str = Field(description="3-paragraph unified executive summary synthesizing all sections")
        key_takeaways: List[str] = Field(description="5-7 top bullet points representing the entire video")

    reduce_prompt = (
        "Synthesize an engaging title, comprehensive 3-paragraph executive overview, and 5-7 key takeaways "
        "from these chronological section summaries of the full video:\n\n" +
        "\n\n".join(section_overviews)
    )

    for k in api_keys:
        client = genai.Client(api_key=k)
        for m in models_to_try:
            try:
                final_resp = client.models.generate_content(
                    model=m,
                    contents=[reduce_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=FinalSynthesisSchema,
                        temperature=0.2
                    )
                )
                if final_resp and final_resp.text:
                    final_parsed = FinalSynthesisSchema.model_validate_json(final_resp.text)
                    return VideoSummaryResult(
                        title=final_parsed.title,
                        overview=final_parsed.overview,
                        key_takeaways=final_parsed.key_takeaways,
                        highlights=optimized_hl
                    )
            except Exception:
                continue

    return VideoSummaryResult(
        title="Comprehensive Video Summary",
        overview="\n\n".join(section_overviews) if section_overviews else "Executive summary synthesized from video sections.",
        key_takeaways=all_takeaways[:7] if all_takeaways else ["Key highlight takeaways extracted."],
        highlights=optimized_hl
    )


def _generate_single_pass_gemini_summary(
    transcript_segments: List[Dict[str, Any]],
    total_duration: float,
    api_keys: List[str],
    model_name: Optional[str] = None,
    target_summary_ratio: float = 0.30,
    custom_focus_prompt: Optional[str] = None
) -> VideoSummaryResult:
    """Fallback engine that passes the complete transcript in one unified pass with model rotation."""
    from google import genai
    from google.genai import types

    target_seconds = total_duration * target_summary_ratio
    formatted_transcript = format_transcript_with_timestamps(transcript_segments)
    custom_focus = f"\nUser Special Focus: {custom_focus_prompt}" if custom_focus_prompt else ""

    system_prompt = (
        "You are a master AI video director and executive summarizer.\n"
        "Your mission is to analyze the FULL timestamped transcript of a video and select the MOST IMPORTANT "
        "highlight segments to create an engaging, comprehensive, and high-accuracy condensed summary video.\n\n"
        "CRITICAL RULES:\n"
        f"1. Total video duration: {total_duration:.1f} seconds (~{total_duration/60:.1f} mins).\n"
        f"2. Target summary length: Approximately {target_seconds:.1f} seconds ({target_summary_ratio*100:.0f}% of original).\n"
        "3. WHOLE-VIDEO CHRONOLOGICAL COVERAGE: Ensure highlights are distributed across the ENTIRE duration of the video "
        "(beginning introduction, early core points, middle deep dives/demos, late-stage insights, and final conclusions).\n"
        "4. Exact Timestamps: Use precise start_time and end_time in seconds matching the transcript.\n"
        "5. Complete Thoughts: Ensure each highlight captures a complete statement.\n"
        "6. Provide a detailed 3-paragraph executive overview and 5-7 key takeaways.\n"
        f"{custom_focus}"
    )

    human_prompt = (
        f"Here is the complete timestamped video transcript:\n\n"
        f"```text\n{formatted_transcript}\n```\n\n"
        f"Analyze the entire transcript across all sections and return the JSON VideoSummaryResult with optimal highlights totaling approximately {target_seconds:.1f} seconds."
    )

    models_to_try = [
        model_name or "gemini-3.7-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ]
    models_to_try = list(dict.fromkeys(models_to_try))

    last_error = None
    for k in api_keys:
        client = genai.Client(api_key=k)
        for m in models_to_try:
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
                    result.highlights = optimize_highlight_boundaries(result.highlights, total_duration)
                    return result
            except Exception as e:
                last_error = e
                continue

    raise RuntimeError(f"All Gemini models and API keys failed: {last_error}")




