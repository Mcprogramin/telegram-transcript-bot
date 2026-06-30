# 🎙️ Doros Transcript Bot

**AI-Powered Telegram Audio Transcription & Elite Arabic Formatting**

An enterprise-grade Telegram bot that transcribes long Arabic audio lectures and formats them into publication-ready, academically structured text. Built with a powerful AI stack, it handles massive files, corrects phonetic errors, and beautifully formats Islamic theology, Quranic verses, and poetry.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Telegram](https://img.shields.io/badge/Telegram-MTProto-0088cc?logo=telegram)
![Groq](https://img.shields.io/badge/STT-Groq_Whisper-F55036)
![Mistral](https://img.shields.io/badge/LLM-Mistral_Large-FF7000)

---

## ✨ Features

- 🎧 **Massive File Support:** Bypasses Telegram's 20MB limit using Pyrogram MTProto to download and process audio files up to **2GB**.
- 🧠 **Elite AI Stack:** 
  - **STT:** Groq `whisper-large-v3-turbo` for lightning-fast, highly accurate Arabic transcription.
  - **LLM:** Mistral Large 2512 (262K Context Window) for deep contextual understanding and editing.
- 📖 **The "Reconstructor" Prompt:** A custom-engineered system prompt that forces the LLM to act as an elite Arabic editor. It fixes Whisper's phonetic mishearings, adds Quranic diacritics (Tashkeel), and formats poetry correctly.
- ⏳ **Smart FCFS Queue:** A strict First-Come-First-Served `asyncio.Queue` system prevents server crashes, manages memory efficiently, and respects API rate limits when multiple users send files simultaneously.
- 🛡️ **Anti-Hallucination Engine:** Python-level Regex cleaning automatically strips common Whisper artifacts (e.g., "ترجمة نانسي قنقر", "اشتركوا في القناة") before the text ever reaches the LLM.
- ⚡ **Optimized Chunking:** Processes up to **50,000 words per chunk**, maximizing Mistral Large's 262K context window for blazing-fast processing speeds.
- 🆓 **Free Hosting Compatible:** Includes a built-in FastAPI dummy server trick to keep the bot alive 24/7 on free-tier platforms like Hugging Face Spaces.

---

## 🏗️ Architecture

1. **Download & Compress:** Pyrogram downloads the audio. `Pydub` and `FFmpeg` compress/slice it if it exceeds Groq's 25MB API limit.
2. **Transcription:** Audio chunks are sent to Groq Whisper in sequence.
3. **Pre-Cleaning:** Regex strips AI hallucinations and intro/outro junk.
4. **Reconstruction:** The raw text is chunked (up to 50K words) and sent to Mistral Large with the custom Arabic formatting prompt.
5. **Delivery:** The beautifully formatted text is split into Telegram-friendly 4096-character messages and sent back to the user.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- FFmpeg installed on your system
- Telegram API credentials ([my.telegram.org](https://my.telegram.org))
- Groq API Key ([console.groq.com](https://console.groq.com))
- Mistral API Key ([console.mistral.ai](https://console.mistral.ai))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/doros-transcript-bot.git
   cd doros-transcript-bot
