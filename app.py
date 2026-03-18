"""
Journalist Audio Copilot — MVP
Upload an audio file → get transcript, summary, key quotes, story brief, and angles.
Run: streamlit run app.py
Requires: pip install streamlit openai
"""

import streamlit as st
import tempfile
import os
from openai import OpenAI

# ── OpenAI client (reads key from Streamlit secrets or env var) ──────────────
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OpenAI API key. Add it to .streamlit/secrets.toml or set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=api_key)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Journalist Audio Copilot", page_icon="🎙️", layout="wide")

st.title("🎙️ Journalist Audio Copilot")
st.caption("Upload an audio file and get an instant transcript, summary, key quotes, story brief, and suggested angles.")

# ── Sidebar / inputs ────────────────────────────────────────────────────────
audio_file = st.file_uploader(
    "Upload audio (.mp3, .wav, .m4a)",
    type=["mp3", "wav", "m4a"],
)

context = st.text_area(
    "Context (optional — what is this recording about?)",
    placeholder="e.g. Interview with the mayor about the new housing policy",
)

process = st.button("Process", type="primary", disabled=audio_file is None)


MAX_WHISPER_BYTES = 25 * 1024 * 1024  # 25 MB Whisper limit


def _compress_audio(tmp_path: str) -> str:
    """Re-encode audio as 64kbps mono mp3 to fit under the Whisper size limit."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(tmp_path)
    compressed_path = tmp_path + ".mp3"
    audio.export(compressed_path, format="mp3", parameters=["-ac", "1", "-b:a", "64k"])
    return compressed_path


# ── Helper: transcribe audio via Whisper ─────────────────────────────────────
def transcribe_audio(audio_file) -> dict:
    """Send audio to OpenAI Whisper and return the verbose JSON response."""
    # Write uploaded file to a temp file (Whisper API needs a file path)
    suffix = os.path.splitext(audio_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    try:
        # Compress if file exceeds Whisper's 25 MB limit
        if os.path.getsize(tmp_path) > MAX_WHISPER_BYTES:
            st.info("File exceeds 25 MB — compressing audio…")
            compressed = _compress_audio(tmp_path)
            os.unlink(tmp_path)
            tmp_path = compressed

        with open(tmp_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        return response
    finally:
        os.unlink(tmp_path)


# ── Helper: build the analysis prompt ────────────────────────────────────────
def build_prompt(transcript: str, context: str) -> str:
    return f"""You are an experienced investigative journalist.

Given the following transcript and context:

Context: {context if context else "No additional context provided."}

Transcript:
{transcript}

Generate the following, using markdown formatting:

## Summary
A concise summary in 5–7 bullet points.

## Key Quotes
The most important direct quotes from the transcript. Include approximate timestamps if available. Format each quote so it is easy to copy.

## Story Brief
- **Headline**: a compelling headline
- **Intro**: 2–3 sentence introduction
- **Key points**: bulleted list of the most important facts

## Suggested Angles
3–5 possible story angles, including at least one unexpected or counter-intuitive angle.

Be specific, critical, and avoid generic phrasing."""


# ── Helper: run LLM analysis ────────────────────────────────────────────────
def analyze_transcript(transcript: str, context: str) -> str:
    """Send transcript + context to GPT-4o and return the analysis."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": build_prompt(transcript, context)}],
        temperature=0.4,
    )
    return response.choices[0].message.content


# ── Helper: format transcript with timestamps ───────────────────────────────
def format_transcript(whisper_response) -> str:
    """Turn Whisper verbose JSON into a readable timestamped transcript."""
    segments = getattr(whisper_response, "segments", None)
    if not segments:
        return whisper_response.text

    lines = []
    for seg in segments:
        start = seg.get("start", seg.get("Start", 0)) if isinstance(seg, dict) else getattr(seg, "start", 0)
        text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        mins, secs = divmod(int(start), 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {text.strip()}")
    return "\n".join(lines)


# ── Main processing flow ────────────────────────────────────────────────────
if process and audio_file is not None:
    # Step 1: Transcribe
    with st.spinner("Transcribing audio…"):
        try:
            whisper_result = transcribe_audio(audio_file)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

    transcript_text = format_transcript(whisper_result)
    plain_text = whisper_result.text  # plain version for LLM

    # Step 2: Analyze
    with st.spinner("Analyzing transcript…"):
        try:
            analysis = analyze_transcript(transcript_text, context)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ── Display results ──────────────────────────────────────────────────────
    st.divider()

    # Transcript section
    st.subheader("📝 Full Transcript")
    st.text_area("Transcript", transcript_text, height=300, label_visibility="collapsed")

    st.divider()

    # Analysis sections (rendered as markdown)
    st.markdown(analysis)

    st.divider()

    # Download button — combine everything into one text file
    full_output = f"JOURNALIST AUDIO COPILOT — REPORT\n{'=' * 40}\n\n"
    full_output += f"Audio file: {audio_file.name}\n"
    full_output += f"Context: {context or '(none)'}\n\n"
    full_output += f"TRANSCRIPT\n{'-' * 40}\n{transcript_text}\n\n"
    full_output += f"ANALYSIS\n{'-' * 40}\n{analysis}\n"

    st.download_button(
        label="Download full report as text",
        data=full_output,
        file_name="copilot_report.txt",
        mime="text/plain",
    )
