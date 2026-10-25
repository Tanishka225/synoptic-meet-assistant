# Synoptic
**Live meeting transcription, turned into clear notes and minutes of meeting.**

A locally run web application that captures meeting audio, shows a live transcript as people speak, and produces a polished final transcript, summary, and MoM when the meeting ends.

**Why This Project Matters**

Meetings produce decisions, action items, and commitments, but most of that value disappears the moment the call ends. People struggle to listen closely and take good notes at the same time. Memory is unreliable, and writing minutes afterwards is slow and tedious.

The problem               	                                                      How Synoptic helps

Note-taking pulls attention away from the discussion	              ->              A live transcript captures the conversation while participants stay engaged.

Live speech recognition is fast but imperfect	                      ->              A second, more accurate pass re-transcribes the full recording after the meeting.

Raw transcripts are too long to be useful	                          ->              An automatic summary and MoM distill the key points.

Meeting audio can be confidential	                                  ->             The app is designed to run on your own computer, with no hosting service required.

**✨ Features**

 Live transcription: Vosk provides incremental transcript updates while audio is being streamed.
 
 Final transcription: Faster-Whisper processes the recorded audio when the meeting ends for a more accurate transcript.
 
 Meeting notes: the backend calls summarize_and_create_mom(...) to produce a summary and minutes of meeting, if that function is configured and working.
 
 Browser-based interface: a frontend communicates with the FastAPI backend over HTTP and WebSockets.
 
 Local execution: designed to run on your own Windows computer without deploying to a hosting service.
 
 Tab-audio capture: audio from a shared browser tab (such as a meeting tab) can be captured through the browser's sharing flow.

**How It Works**

Synoptic uses a two-pass transcription design. A lightweight model keeps the transcript moving in real time, and a stronger model refines the result when the meeting ends.

**Why it matters**

1	**Capture**	The browser captures audio from the microphone or from a shared tab. Browser capture works without installing extra audio drivers.

2	**Stream**	Audio is sent to the backend over a WebSocket. A persistent connection allows low-latency, continuous delivery.

3	**Live** recognition	Vosk transcribes the stream incrementally. Participants see the conversation as text almost immediately.

4	**Record**	The same audio is buffered and recorded. The full recording is needed for the accurate final pass.

5	**Final** recognition	When the meeting ends, Faster-Whisper transcribes the recorded audio.	A larger model on the complete audio is generally more accurate than live output.

6	**Summarize** Produces a summary and MoM from the final transcript. Turns a long transcript into something that can be read and shared quickly.

**🎯 Key Design Decisions**

Why two transcription engines? Live and final transcription have different goals. Vosk's small model is lightweight and responsive, which suits real-time display. Faster-Whisper takes more time and resources but can work from the complete recording, which is better for a finished transcript.

Why generate the notes from the final transcript? Summaries are only as good as the text they are built from. Using the more accurate final pass gives the summary and MoM a stronger foundation than the live output.

Why run locally? Meeting audio often contains sensitive information. The transcription pipeline runs on your own machine rather than on a hosting service you have to deploy and secure. The summary step depends on how summarize_and_create_mom(...) is implemented and on any services or credentials it requires.

Why a browser-based capture flow? It lets Synoptic work with whatever audio the browser can access, such as a microphone or a shared tab, without needing a direct meeting-platform integration.

**Requirements**

-Windows 10/11

-Python installed and working from Command Prompt

-Internet connection for installing Python packages and downloading the Faster-Whisper model the first time

-Enough free disk space for Python packages, model files, and temporary audio

-A modern browser such as Chrome

-A microphone or a browser tab that provides meeting audio
