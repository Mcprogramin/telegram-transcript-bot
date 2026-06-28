"""
Telegram MTProto Audio Transcript Bot (Ultimate Groq Stack)
===========================================================
STT: Whisper-large-v3-turbo (Best Arabic accuracy)
Text: GPT OSS 120B (Elite intelligence) with Llama 3.1 8B Instant fallback.
"""

import os
import re
import math
import asyncio
import shutil
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from pyrogram import Client, filters
from pyrogram.enums import ParseMode
from groq import Groq, RateLimitError

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
GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"
GROQ_TEXT_MODEL = "openai/gpt-oss-120b"
FALLBACK_TEXT_MODEL = "llama-3.1-8b-instant"
TRANSCRIPT_LANGUAGE = os.getenv("TRANSCRIPT_LANGUAGE", "ar").strip() or None

# Fixed Regex Patterns (Exactly as you specified)
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

# Initialize Pyrogram Client
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
    """Compresses WAV/large files to MP3, then slices by size only if necessary."""
    max_size_bytes = 20 * 1024 * 1024  # 20MB safe limit (Groq max is 25MB)
    file_size = os.path.getsize(input_path)
    
    if file_size <= max_size_bytes:
        return [input_path]
        
    print(f"File is {file_size / (1024*1024):.2f} MB. Attempting compression...")
    
    compressed_path = os.path.join(task_dir, "compressed_input.mp3")
    audio = AudioSegment.from_file(input_path)
    audio.export(compressed_path, format="mp3", bitrate="64k", parameters=["-ac", "1"])
    
    compressed_size = os.path.getsize(compressed_path)
    print(f"Compressed to {compressed_size / (1024*1024):.2f} MB")
    
    if compressed_size <= max_size_bytes:
        return [compressed_path]
        
    print(f"Still too big. Slicing by size...")
    total_duration_ms = len(audio)
    num_chunks = math.ceil(compressed_size / max_size_bytes)
    chunk_length_ms = total_duration_ms // num_chunks
    
    chunks = make_chunks(audio, chunk_length_ms)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(task_dir, f"chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate="64k", parameters=["-ac", "1"])
        chunk_paths.append(chunk_path)
        
    print(f"Split into {len(chunk_paths)} chunks based on size.")
    return chunk_paths

def _transcribe_sync(groq_client: Groq, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            model=GROQ_AUDIO_MODEL, file=f, response_format="text", language=TRANSCRIPT_LANGUAGE
        )
    return resp.strip() if isinstance(resp, str) else getattr(resp, "text", "").strip()

def _chunk_text(text: str, max_words: int = 2000) -> list[str]:
    """Splits text into chunks to respect TPM limits."""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

async def _format_sync_async(groq_client: Groq, raw_text: str) -> str:
    """Formats text using GPT OSS 120B with automatic fallback to Llama 3.1 8B."""
    chunks = _chunk_text(raw_text, max_words=2000)
    formatted_parts = []
    
    for i, chunk in enumerate(chunks):
        current_model = GROQ_TEXT_MODEL
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Add a 2.5s delay to stay safely under the 30 RPM limit
                if i > 0 or attempt > 0:
                    await asyncio.sleep(2.5)
                    
                completion = groq_client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": _FORMAT_SYSTEM},
                        {"role": "user", "content": f"النص:\n\n{chunk}"}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                
                formatted_parts.append(completion.choices[0].message.content or "")
                break  # Success, move to next chunk
                
            except RateLimitError:
                print(f"️ Rate limit hit on {current_model}. Switching to fallback...")
                current_model = FALLBACK_TEXT_MODEL  # Switch to fallback for this chunk
                
        # If it somehow fails both, append the raw chunk to prevent crashing
        if len(formatted_parts) <= i:
            formatted_parts.append(chunk) 

    return _strip_code_fences(" ".join(formatted_parts))

# ---------------------------------------------------------------------------
# Core Processing Engine
# ---------------------------------------------------------------------------
async def process_audio_task(message, file_path: str, task_dir: str):
    status_msg = await message.reply("️ Processing audio size...")
    
    try:
        # 1. Smart size-based processing (Compress -> Slice if needed)
        chunk_paths = await asyncio.to_thread(_process_audio_chunks, file_path, task_dir)
        
        # 2. Transcribe each chunk with Groq Whisper
        groq_client = Groq(api_key=GROQ_API_KEY)
        raw_text_parts = []
        
        for i, path in enumerate(chunk_paths):
            if len(chunk_paths) > 1:
                await status_msg.edit_text(f"🎙️ Transcribing chunk {i+1}/{len(chunk_paths)} with Whisper...")
            else:
                await status_msg.edit_text("🎙️ Transcribing with Groq Whisper...")
                
            part_text = await asyncio.to_thread(_transcribe_sync, groq_client, path)
            raw_text_parts.append(part_text)
            
        raw_text = " ".join(raw_text_parts)
        
        # 3. Format with GPT OSS 120B (Falls back to Llama 3.1 8B if rate limited)
        await status_msg.edit_text("✨ Formatting with GPT OSS 120B...")
        formatted_text = await _format_sync_async(groq_client, raw_text)
        
        # 4. Send back to Telegram
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
        " Welcome to the Ultimate Groq Audio Bot!\n\n"
        "I use Groq Whisper for transcription and GPT OSS 120B for elite Arabic formatting.\n"
        "If limits are hit, I automatically fallback to Llama 3.1 8B Instant.\n"
        "I support massive files (up to 2GB)!"
    )

@app.on_message(filters.audio | filters.voice | filters.document)
async def handle_audio(client, message):
    if message.document and not message.document.mime_type.startswith("audio/"):
        return

    task_id = str(uuid.uuid4())
    task_dir = os.path.join("temp_audio", task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    if message.voice:
        file_ext = "ogg"
    elif message.audio:
        file_ext = message.audio.file_name.split('.')[-1] if message.audio.file_name else "mp3"
    else:
        file_ext = message.document.file_name.split('.')[-1] if message.document.file_name else "mp3"
        
    file_path = os.path.join(task_dir, f"input.{file_ext}")
    
    status_msg = await message.reply("⬇️ Downloading from Telegram (No Limits)...")
    
    try:
        await message.download(file_name=file_path)
        await status_msg.edit_text("📥 Downloaded! Processing...")
        await process_audio_task(message, file_path, task_dir)
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Download failed: {str(e)[:100]}")
        shutil.rmtree(task_dir)

def main():
    if not all([API_ID, API_HASH, SESSION_STRING, GROQ_API_KEY]):
        print("ERROR: Missing environment variables!")
        return

    print("Starting Pyrogram Client...")
    app.run()

if __name__ == "__main__":
    main()