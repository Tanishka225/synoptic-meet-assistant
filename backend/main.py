# main.py - Synoptic backend (copy-paste ready)
# ----------------------------
# Features:
# - /meetings/live/start  -> create meeting
# - /ws/meetings/{id}     -> receive raw PCM16LE frames -> Vosk partials (streaming)
# - /meetings/{id}/end    -> write buffer -> Whisper final transcript -> summary + MoM
# - deduplication of partials + repeated phrase cleanup
# ----------------------------

import os
import json
import uuid
import tempfile
import shutil
import re
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# speech libs
from vosk import Model as VoskModel, KaldiRecognizer
import numpy as np
import soundfile as sf
import whisper

VOSK_MODEL_PATH = r"c:\Users\HP\Desktop\synoptic\vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = "base.en" 
MIN_PARTIAL_CHARS = 3       
OVERLAP_RATIO = 0.7
MAX_NGRAM_DEDUPE = 4
app = FastAPI(title="Synoptic Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- Data models ----------------
class MeetingStatus(str, Enum):
    LIVE = "live"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class Meeting:
    id: str
    title: str
    status: MeetingStatus
    transcript: Optional[str] = None
    summary: Optional[str] = None
    mom: Optional[str] = None
    live_buffer: Optional[List[str]] = None
    error: Optional[str] = None

# ---------------- In-memory stores ----------------
MEETINGS: Dict[str, Meeting] = {}
LIVE_AUDIO_BUFFERS: Dict[str, bytearray] = {}     # meeting_id -> raw PCM16LE bytes
VOSK_RECOGNIZERS: Dict[str, KaldiRecognizer] = {} # meeting_id -> KaldiRecognizer
LAST_PARTIALS: Dict[str, str] = {}                # meeting_id -> last partial sent (dedupe)

# ---------------- Load models ----------------
print("Loading Whisper model:", WHISPER_MODEL_NAME)
WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)
print("Whisper loaded.")

print("Loading Vosk model from:", VOSK_MODEL_PATH)
VOSK_MODEL = VoskModel(VOSK_MODEL_PATH)
print("Vosk loaded.")

# ---------------- Helpers ----------------
def create_meeting(title: str) -> Meeting:
    m_id = str(uuid.uuid4())
    meeting = Meeting(id=m_id, title=title, status=MeetingStatus.LIVE, live_buffer=[])
    MEETINGS[m_id] = meeting
    LIVE_AUDIO_BUFFERS[m_id] = bytearray()
    rec = KaldiRecognizer(VOSK_MODEL, SAMPLE_RATE)
    VOSK_RECOGNIZERS[m_id] = rec
    LAST_PARTIALS[m_id] = ""
    print(f"[MEETING] Created {m_id} title='{title}'")
    return meeting

def write_buffer_to_wav(meeting_id: str, out_path: str) -> bool:
    buf = LIVE_AUDIO_BUFFERS.get(meeting_id)
    if not buf or len(buf) == 0:
        return False
    try:
        arr = np.frombuffer(bytes(buf), dtype=np.int16)
        sf.write(out_path, arr, SAMPLE_RATE, subtype="PCM_16")
        return True
    except Exception as e:
        print("[ERROR] write_buffer_to_wav:", e)
        return False

def dedupe_repeated_phrases(text: str, max_ngram=MAX_NGRAM_DEDUPE) -> str:
    """
    Remove immediate repeated sequences (e.g., 'hello hello hello', 'we will we will').
    Scans for repeated adjacent n-grams up to max_ngram length.
    """
    if not text:
        return text
    words = re.findall(r'\S+', text)
    n = len(words)
    i = 0
    out = []
    while i < n:
        took = False
        for l in range(max_ngram, 0, -1):
            if i + 2*l <= n:
                seq1 = words[i:i+l]
                seq2 = words[i+l:i+2*l]
                if seq1 == seq2:
                    out.extend(seq1)
                    i += 2*l
                    took = True
                    break
        if not took:
            out.append(words[i])
            i += 1
    return " ".join(out)

def simple_summary_and_mom(transcript: str):
    """Fallback summarizer and MoM generator (replace with LLM/Gemini integration if desired)."""
    if not transcript or transcript.strip() == "":
        return ("", "")
    sents = [s.strip() for s in re.split(r'[.?!]\s*', transcript) if s.strip()]
    summary = ""
    if len(sents) >= 2:
        summary = sents[0] + ". " + sents[1] + "."
    else:
        summary = sents[0] + "." if sents else transcript[:200]
    # action item heuristic
    actions = []
    triggers = [" will ", " shall ", " should ", " to "]
    for s in sents:
        low = " " + s.lower() + " "
        for t in triggers:
            if t in low and len(s.split()) >= 3:
                actions.append(s.strip() + ".")
                break
    mom_lines = ["Title: Auto-generated MoM", "", "Summary:", summary, "", "Action Items:"]
    if actions:
        for i,a in enumerate(actions,1):
            mom_lines.append(f"{i}. {a}")
    else:
        mom_lines.append("No explicit action items detected.")
    mom_lines.extend(["", "Transcript excerpt:", transcript[:1200]])
    return (summary, "\n".join(mom_lines))

def summarize_and_create_mom(transcript: str):
    # placeholder: integrate Gemini/LLM here for high-quality MoM
    return simple_summary_and_mom(transcript)

# ---------------- API endpoints ----------------
@app.post("/meetings/live/start")
async def start_live_meeting(title: str = Form("Untitled Meeting")):
    meeting = create_meeting(title)
    return asdict(meeting)

@app.websocket("/ws/meetings/{meeting_id}")
async def websocket_live_meeting(websocket: WebSocket, meeting_id: str):
    """
    Receive raw PCM16LE frames from browser.
    Use Vosk for streaming partials. Deduplicate using LAST_PARTIALS.
    Append all bytes to LIVE_AUDIO_BUFFERS for final Whisper transcription.
    """
    if meeting_id not in MEETINGS:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    print(f"[WS] Client connected for meeting {meeting_id}")

    rec = VOSK_RECOGNIZERS.get(meeting_id)
    if rec is None:
        rec = KaldiRecognizer(VOSK_MODEL, SAMPLE_RATE)
        VOSK_RECOGNIZERS[meeting_id] = rec
        LAST_PARTIALS[meeting_id] = ""

    meeting = MEETINGS[meeting_id]

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            # store bytes for final transcript
            LIVE_AUDIO_BUFFERS[meeting_id].extend(data)

            # Feed data into Vosk recognizer
            try:
                accepted = rec.AcceptWaveform(data)
                if accepted:
                    # finalized segment
                    res = json.loads(rec.Result())
                    text = res.get("text", "").strip()
                    if text:
                        # avoid sending identical final twice
                        last_sent = LAST_PARTIALS.get(meeting_id, "")
                        if text != last_sent:
                            meeting.live_buffer.append(text)
                            MEETINGS[meeting_id] = meeting
                            await websocket.send_json({"type": "partial_transcript", "text": text, "final": True})
                            LAST_PARTIALS[meeting_id] = ""  # reset after final
                else:
                    # partial result
                    partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                    last = LAST_PARTIALS.get(meeting_id, "")
                    if not partial:
                        continue
                    # ignore very short partials
                    if len(partial) < MIN_PARTIAL_CHARS:
                        continue
                    # identical?
                    if partial == last:
                        continue
                    # overlap heuristic: if new partial contains last or vice versa, require it to be sufficiently longer
                    if last and (last in partial or partial in last):
                        if len(partial) <= len(last) * OVERLAP_RATIO:
                            continue
                    # send partial
                    await websocket.send_json({"type": "partial_transcript", "text": partial, "final": False})
                    LAST_PARTIALS[meeting_id] = partial

            except Exception as e:
                print("[WS] Vosk error:", e)
                try:
                    await websocket.send_json({"type":"error","message":f"Vosk error: {e}"})
                except:
                    pass

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected for meeting {meeting_id}")
        return
    except Exception as e:
        print("[WS] Unexpected error:", e)
        try:
            await websocket.send_json({"type":"error","message":str(e)})
        except:
            pass
        return

@app.post("/meetings/{meeting_id}/end")
def end_live_meeting(meeting_id: str):
    """
    On meeting end:
    - write PCM buffer to WAV
    - run Whisper final transcription
    - dedupe repeated phrases
    - generate summary & MoM
    """
    meeting = MEETINGS.get(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.status != MeetingStatus.LIVE:
        raise HTTPException(status_code=400, detail="Meeting is not LIVE")

    # write buffer to temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    ok = write_buffer_to_wav(meeting_id, wav_path)
    if not ok:
        meeting.status = MeetingStatus.ERROR
        meeting.error = "No audio buffer available"
        MEETINGS[meeting_id] = meeting
        try:
            os.remove(wav_path)
        except:
            pass
        return asdict(meeting)

    # final transcription with Whisper
    try:
        print(f"[WHISPER] Transcribing {wav_path} ...")
        result = WHISPER_MODEL.transcribe(wav_path, language="en", fp16=False)
        transcript = (result.get("text") or "").strip()
        # cleanup repeated phrases
        transcript = dedupe_repeated_phrases(transcript, max_ngram=MAX_NGRAM_DEDUPE)
        meeting.transcript = transcript
    except Exception as e:
        meeting.status = MeetingStatus.ERROR
        meeting.error = f"Whisper transcription failed: {e}"
        MEETINGS[meeting_id] = meeting
        try:
            os.remove(wav_path)
        except:
            pass
        return asdict(meeting)
    finally:
        try:
            os.remove(wav_path)
        except:
            pass

    # summarize & MoM
    try:
        summary, mom = summarize_and_create_mom(meeting.transcript)
        meeting.summary = summary
        meeting.mom = mom
        meeting.status = MeetingStatus.COMPLETED
    except Exception as e:
        meeting.status = MeetingStatus.ERROR
        meeting.error = f"Summarization failed: {e}"

    # cleanup
    try:
        LIVE_AUDIO_BUFFERS.pop(meeting_id, None)
        VOSK_RECOGNIZERS.pop(meeting_id, None)
        LAST_PARTIALS.pop(meeting_id, None)
    except:
        pass

    MEETINGS[meeting_id] = meeting
    return asdict(meeting)

@app.post("/meetings")
async def upload_offline_meeting(title: str = Form("Uploaded Recording"), audio: UploadFile = File(...)):
    """
    Upload an audio file; Whisper will transcribe and we generate summary+MoM.
    """
    m = create_meeting(title)
    tmp_dir = tempfile.mkdtemp(prefix="synoptic_")
    try:
        upload_path = os.path.join(tmp_dir, audio.filename)
        with open(upload_path, "wb") as f:
            f.write(await audio.read())
        try:
            result = WHISPER_MODEL.transcribe(upload_path, language="en", fp16=False)
            transcript = (result.get("text") or "").strip()
            transcript = dedupe_repeated_phrases(transcript, max_ngram=MAX_NGRAM_DEDUPE)
            m.transcript = transcript
            s, mom = summarize_and_create_mom(transcript)
            m.summary = s
            m.mom = mom
            m.status = MeetingStatus.COMPLETED
            MEETINGS[m.id] = m
            return asdict(m)
        except Exception as e:
            m.status = MeetingStatus.ERROR
            m.error = f"Whisper error: {e}"
            MEETINGS[m.id] = m
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.get("/meetings")
def list_meetings():
    return [asdict(m) for m in MEETINGS.values()]

@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    m = MEETINGS.get(meeting_id)
    if not m:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return asdict(m)

@app.get("/meetings/{meeting_id}/mom/download")
def download_mom(meeting_id: str):
    m = MEETINGS.get(meeting_id)
    if not m or not m.mom:
        raise HTTPException(status_code=404, detail="MoM not available")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        path = tmp.name
        tmp.write(m.mom.encode("utf-8"))
    return FileResponse(path, media_type="text/plain", filename=f"mom_{meeting_id}.txt")

@app.get("/health")
def health():
    return {"status":"ok"}
