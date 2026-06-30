# 🎙️ Doros Transcript Bot — تفريغ الصوتيات الآلي

An enterprise-grade Telegram bot that transcribes Arabic Islamic lectures (دروس) and reconstructs them into clean, publication-ready Arabic text — built to turn raw, noisy audio into something you could publish straight to a website or PDF.

🔗 **Live bot:** [@dorus_transcriptbot](https://t.me/dorus_transcriptbot)
🚀 **Hosted on:** Hugging Face Spaces (kept awake via the FastAPI dummy server trick described below)

It combines fast, accurate speech-to-text with an LLM-powered "reconstruction" pass that goes beyond simple cleanup: it rebuilds sentence structure, fixes phonetic mishearings, restores Qur'anic verses and Hadith to their correct wording, and properly formats poetry (شعر) — all while staying faithful to the speaker's intended meaning rather than Whisper's literal (and often broken) output.

---

## ✨ Features

- **🎧 Speech-to-Text** — Powered by Groq's `whisper-large-v3-turbo` for fast, high-accuracy Arabic transcription.
- **✍️ Intelligent Reconstruction** — Mistral Large rewrites the raw transcript as a "Reconstructor" (مُعيد بناء), not just a proofreader:
  - Rebuilds punctuation and sentence flow based on meaning, not Whisper's noisy pause-based periods.
  - Corrects phonetic/mishearing errors (e.g. مفردات مشوهة → الكلمة الصحيحة حسب السياق).
  - Restores Qur'anic ayat and Hadith to their known, correct text.
  - Properly formats poetry into paired hemistichs (`شطر *** شطر`).
  - Wraps Qur'an in `«»` and Hadith in `""`.
- **📦 Automatic Chunking** — Large audio files are compressed and split automatically to stay under API size limits; long transcripts are chunked for the LLM's context window.
- **🧵 FCFS Queue System** — A strict first-come-first-served async queue prevents concurrent processing crashes and rate-limit collisions, with live queue position updates sent to users.
- **🔄 Live Status Updates** — Users get real-time progress messages (downloading → transcribing → reconstructing → sending).
- **🧹 Whisper Artifact Cleanup** — Strips common transcription noise (subscribe prompts, channel mentions, leftover markdown fences, etc.) before formatting.
- **📨 Smart Message Splitting** — Long transcripts are intelligently split across multiple Telegram messages, respecting paragraph and sentence boundaries.
- **🌐 Always-On Trick** — A lightweight FastAPI server runs alongside the bot to prevent free-tier hosting platforms from idling it to sleep.

---

## 🏗️ Architecture

```
Telegram Audio/Voice Message
        │
        ▼
 ┌─────────────────┐
 │   FCFS Queue     │  (async, single worker — no race conditions)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │  Compress/Chunk   │  (pydub + ffmpeg, keeps files under 20MB)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │  Groq Whisper STT │  (whisper-large-v3-turbo)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │  Artifact Cleanup  │  (regex-based noise removal)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │  Mistral Large 2512 │  (system-prompted Reconstructor)
 └─────────────────┘
        │
        ▼
   Formatted Arabic Text → sent back to user on Telegram
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Telegram Client | [Pyrogram](https://docs.pyrogram.org/) (MTProto) |
| Speech-to-Text | [Groq API](https://groq.com/) — `whisper-large-v3-turbo` |
| Text Reconstruction | [Mistral AI](https://mistral.ai/) — `mistral-large-2512` |
| Audio Processing | `pydub` + `static-ffmpeg` |
| Keep-Alive Server | `FastAPI` + `uvicorn` |
| Concurrency | `asyncio` Queue (FCFS worker) |

---

## 📋 Prerequisites

- Python 3.10+
- A Telegram API ID/Hash and session string (via Pyrogram)
- A [Groq API key](https://console.groq.com/)
- A [Mistral AI API key](https://console.mistral.ai/)
- `ffmpeg` (auto-installed via `static-ffmpeg`)

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   TELEGRAM_SESSION_STRING=your_session_string

   GROQ_API_KEY=your_groq_api_key
   MISTRAL_API_KEY=your_mistral_api_key

   # Optional — defaults to "ar"
   TRANSCRIPT_LANGUAGE=ar
   ```

4. **Run the bot**
   ```bash
   python bot.py
   ```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_API_ID` | ✅ | Your Telegram app's API ID |
| `TELEGRAM_API_HASH` | ✅ | Your Telegram app's API hash |
| `TELEGRAM_SESSION_STRING` | ✅ | Pre-generated Pyrogram session string |
| `GROQ_API_KEY` | ✅ | API key for Groq Whisper transcription |
| `MISTRAL_API_KEY` | ✅ | API key for Mistral Large reconstruction |
| `TRANSCRIPT_LANGUAGE` | ❌ | Whisper language hint (default: `ar`) |

---

## 🚀 Deployment

The bot is currently live on **Hugging Face Spaces**. Since Spaces are built for web apps, the bot includes a built-in keep-alive trick: a minimal FastAPI server runs on port `7860` (HF's expected port) and exposes a health check at `/`, reporting bot status and current queue size — this keeps the Space from idling out even though the actual workload is a Telegram client, not a web server.

**Hugging Face Spaces:**
1. Create a new Space with the **Docker** (or Python) SDK.
2. Push this repo's contents to the Space.
3. Add the environment variables listed above as **Space secrets** (Settings → Repository secrets).
4. The Space builds and runs `bot.py`, which starts both the Pyrogram client and the FastAPI keep-alive server.

**Railway / other platforms** work the same way — push the repo, set the env vars, and deploy. `bot.py` handles both the bot and the keep-alive server together.

---

## 💬 Usage

1. Start a chat with your bot on Telegram and send `/start`.
2. Send an audio file, voice note, or audio document.
3. The bot replies with your queue position, then walks through:
   - ⬇️ Downloading
   - ⚙️ Compressing/analyzing
   - 🎙️ Transcribing
   - ✨ Reconstructing
   - 📤 Sending the final formatted text
4. Long lectures are automatically split into multiple messages.

---

## 📁 Project Structure

```
.
├── bot.py              # Main bot logic, queue system, processing pipeline
├── requirements.txt     # Python dependencies
├── .gitignore
└── readme.md
```

---

## 🤝 Contributing

Issues and pull requests are welcome — especially around improving the Arabic reconstruction prompt, handling additional audio formats, or hardening the queue system further.

## 📄 License

Add your preferred license here (e.g. MIT).
