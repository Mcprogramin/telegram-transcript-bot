"""
Telegram MTProto Audio Transcript Bot (God Mode)
================================================
Bypasses all Telegram limits using Pyrogram.
Automatically slices massive files for Groq Whisper.
"""

import os
import re
import asyncio
import shutil
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from pyrogram import Client, filters
from pyrogram.enums import ParseMode

from groq import Groq
from google import genai

# Auto-install ffmpeg for pydub
import static_ffmpeg
static_ffmpeg.add_paths()
from pydub import AudioSegment
from pydub.utils import make_chunks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
SESSION_STRING = os.getenv("TELEGRAM_SESSION_STRING", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = "gemini-2.5-flash"
TRANSCRIPT_LANGUAGE = os.getenv("TRANSCRIPT_LANGUAGE", "ar").strip() or None

# Fixed Regex Patterns
_THINK_RE = re.compile(r"</think>")
_FENCE_RE = re.compile(r"^```[^\n]*\n?", re.MULTILINE)

# The Arabic Prompt
_FORMAT_SYSTEM = """أنت مُعيد بناء لغوي عربي نخبوي ومُحرر نصوص محادثات. هدفك الوحيد هو تحويل النصوص الخام الناتجة عن تحويل الكلام إلى نص (STT) إلى نثر بشري دقيق، منطقي، وطبيعي التدفق.

### بروتوكول المعالجة الرباعية (إلزامي)
**[المرحلة الأولى: تحليل السياق الشامل]** اقرأ النص كاملاً أولاً لفهم اللهجة والمجال.
**[المرحلة الثانية: التحقق من المنطق والسياق]** افحص كل جملة للتأكد من منطقيتها.
**[المرحلة الثالثة: التشريح الصوتي والنحوي]** أصلح هلوسات الذكاء الاصطناعي والأخطاء النحوية.
**[المرحلة الرابعة: بناء الجملة والإيقاع]** أضف علامات الترقيم واحذف التكرار.

### قوانين التصحيح الحاسمة
1. قانون المنطق السياقي: كل جملة يجب أن تكون منطقية.
2. قانون الحفاظ على اللهجة: لا تحول العامية إلى فصحى.
3. قانون الدقة المصطلحية: القرآن في «...»، الحديث في "...".
4. لا تلخص، لا تحذف، لا تضف عناوين.

أخرج النص المُعاد بناؤه فقط. لا مقدمات. لا ملاحظات."""

# Initialize Pyrogram Client using String Session
app = Client(
    "my_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    session_string=SESSION_STRING
)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = text.replace("```", "")
    return text.strip()

def _process_audio_chunks(input_path: str, task_dir: str) -> list[str]:
    """Slices audio into 10-minute chunks if it's too big for Groq."""
    max_size_bytes = 20 * 1024 * 1024  # 20MB safe limit for Groq
    
    if os.path.getsize(input_path) <= max_size_bytes:
        return [input_path]
        
    print(f"File is {os.path.getsize(input_path) / (1024*1024):.2f} MB. Slicing...")
    audio = AudioSegment.from_file(input_path)
    
    # 10 minutes in milliseconds
    chunk_length_ms = 10 * 60 * 1000
    chunks = make_chunks(audio, chunk_length_ms)
    
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(task_dir, f"chunk_{i}.mp3")
        # Export as 64kbps mono MP3 (approx 4.8MB per 10 mins)
        chunk.export(chunk_path, format="mp3", bitrate="64k", parameters=["-ac", "1"])
        chunk_paths.append(chunk_path)
        
    print(f"Split into {len(chunk_paths)} chunks.")
    return chunk_paths

def _transcribe_sync(client: Groq, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=GROQ_AUDIO_MODEL, file=f, response_format="text", language=TRANSCRIPT_LANGUAGE
        )
    return resp.strip() if isinstance(resp, str) else getattr(resp, "text", "").strip()

def _format_sync(client: genai.Client, raw_text: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[_FORMAT_SYSTEM, "---\n\nالنص:\n\n" + raw_text],
        config={"temperature": 0.3}
    )
    return _strip_code_fences(response.text or "")

# ---------------------------------------------------------------------------
# Core Processing Engine
# ---------------------------------------------------------------------------
async def process_audio_task(message, file_path: str, task_dir: str):
    status_msg = await message.reply("⚙️ Processing audio size...")
    
    try:
        # 1. Slice if needed
        chunk_paths = await asyncio.to_thread(_process_audio_chunks, file_path, task_dir)
        
        # 2. Transcribe each chunk
        groq_client = Groq(api_key=GROQ_API_KEY)
        raw_text_parts = []
        
        for i, path in enumerate(chunk_paths):
            if len(chunk_paths) > 1:
                await status_msg.edit_text(f"🎙️ Transcribing chunk {i+1}/{len(chunk_paths)}...")
            else:
                await status_msg.edit_text("🎙️ Transcribing with Groq Whisper...")
                
            part_text = await asyncio.to_thread(_transcribe_sync, groq_client, path)
            raw_text_parts.append(part_text)
            
        # Regroup the text
        raw_text = " ".join(raw_text_parts)
        
        # 3. Format with Gemini
        await status_msg.edit_text("✨ Formatting & fixing Arabic with Gemini...")
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        formatted_text = await asyncio.to_thread(_format_sync, gemini_client, raw_text)
        
        # 4. Send back in chunks
        await status_msg.edit_text("📤 Sending transcript...")
        
        limit = 4000
        for i in range(0, len(formatted_text), limit):
            await message.reply_text(formatted_text[i:i+limit], parse_mode=ParseMode.MARKDOWN)
            
        await status_msg.edit_text("✅ Finished!")

    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
        
    finally:
        try:
            shutil.rmtree(task_dir)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Pyrogram Handlers
# ---------------------------------------------------------------------------
@app.on_message(filters.command("start"))
async def start_cmd(client, message):
    await message.reply_text(
        "👋 Welcome to the God-Mode Audio Transcript Bot!\n\n"
        "I have NO file size limits. Send me any audio file or voice note (up to 2GB).\n"
        "If it's massive, I will automatically slice it, transcribe it, and stitch it back together.\n\n"
        "I will transcribe, format, and fix the Arabic text for you."
    )

@app.on_message(filters.audio | filters.voice | filters.document)
async def handle_audio(client, message):
    # Ignore non-audio documents
    if message.document and not message.document.mime_type.startswith("audio/"):
        return

    task_id = str(uuid.uuid4())
    task_dir = os.path.join("temp_audio", task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Determine file extension
    if message.voice:
        file_ext = "ogg"
    elif message.audio:
        file_ext = message.audio.file_name.split('.')[-1] if message.audio.file_name else "mp3"
    else:
        file_ext = message.document.file_name.split('.')[-1] if message.document.file_name else "mp3"
        
    file_path = os.path.join(task_dir, f"input.{file_ext}")
    
    status_msg = await message.reply("⬇️ Downloading from Telegram (No Limits)...")
    
    try:
        # Pyrogram download
        await message.download(file_name=file_path)
        await status_msg.edit_text("📥 Downloaded! Processing...")
        
        # Process
        await process_audio_task(message, file_path, task_dir)
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Download failed: {str(e)[:100]}")
        shutil.rmtree(task_dir)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not all([API_ID, API_HASH, SESSION_STRING, GROQ_API_KEY, GOOGLE_API_KEY]):
        print("ERROR: Missing environment variables!")
        return

    print("Starting Pyrogram Client...")
    app.run()

if __name__ == "__main__":
    main()