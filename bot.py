"""
Telegram MTProto Audio Transcript Bot (Mistral Large Stack + Queue)
====================================================================
STT: Groq Whisper-large-v3-turbo
Text: Mistral Large 2512 (262K context)
Queue: Strict First-Come-First-Served (FCFS) to prevent crashes & rate limits.
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
from groq import Groq
from mistralai.client import Mistral

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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"
MISTRAL_TEXT_MODEL = "mistral-large-2512"
TRANSCRIPT_LANGUAGE = os.getenv("TRANSCRIPT_LANGUAGE", "ar").strip() or None

# Fixed Regex Patterns
_THINK_RE = re.compile(r"</think>")
_FENCE_RE = re.compile(r"^```[^\n]*\n?", re.MULTILINE)

# The Arabic Prompt (EXACTLY as you wrote it)
_FORMAT_SYSTEM = """أنت "مُعيد بناء" (Reconstructor)، لست محرراً ولا مدققاً. مدخلاتك خراطة صوتية أخرجها Whisper، وليست نصاً كتبه إنسان. مهمتك: استعادة ما أراد الشيخ قوله، لا حفظ ما أخطأ فيه Whisper.

━━━ مثال إلزامي (هذا يُعرّف عملك) ━━━

المدخل:
يقهر هذا. بهذا. وهذا. بهذا. حتى يؤول الأمر. إليه. فلهذا هو السعيد. السعيد. السعيد.

المخرج:
يقهر هذا بهذا، وهذا بهذا، حتى يؤول الأمر إليه، فلهذا هو السعيد.

━━━ القواعد ━━━

١. الترقيم — تجاهل كل نقطة في المدخل، هي ضجيج Whisper لا أكثر. ابنِ الجمل الطويلة المتدفقة من المعنى، لا من الوقفات الصوتية.

٢. التصحيح الصوتي — لك صلاحية كاملة لإعادة كتابة أي كلمة لا تناسب السياق:
   "المحدون" ← المُوَحِّدون | "الأسقياء" ← الأتقياء أو الأشقياء حسب السياق
   "الميشان" ← لِمَن يشاء | "الضده" ← ضِدَّه
   أي آية أو حديث مشوه ← أعد بناءه إلى النص الصحيح المعروف

٣. الشعر — البيت الشعري شطران في سطر واحد، مفصولان بـ (***):
   ✓ صحيح:  فلا تغتر بالدنيا   ***   فإنها دار فناء
   ✗ خاطئ:  فلا تغتر / بالدنيا / فإنها / دار فناء

٤. النصوص الشرعية: الآيات في «»، الأحاديث في ""

المخرج: النص المُعاد بناؤه فقط، بلا عناوين ولا ملاحظات."""

# Initialize Pyrogram Client
app = Client(
    "my_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    session_string=SESSION_STRING
)

# ---------------------------------------------------------------------------
# Queue System
# ---------------------------------------------------------------------------
task_queue = asyncio.Queue()
worker_started = False

async def queue_worker():
    """Background worker that processes tasks one by one (FCFS)."""
    while True:
        # Wait for the next task in the queue
        message, task_dir = await task_queue.get()
        try:
            # Process the task (Download -> Transcribe -> Format -> Send)
            await process_audio_task(message, task_dir)
        except Exception as e:
            print(f"Worker error processing task: {e}")
            try:
                await message.reply_text(f"❌ حدث خطأ غير متوقع في المعالجة: {str(e)[:100]}")
            except:
                pass
        finally:
            # Mark the task as done so the queue can move to the next one
            task_queue.task_done()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = text.replace("```", "")
    return text.strip()

def _clean_whisper_artifacts(text: str) -> str:
    """Physically removes common Whisper hallucinations before sending to LLM."""
    text = re.sub(r"ترجمة.*?قنقر", "", text, flags=re.IGNORECASE)
    text = re.sub(r"اشتركوا في القناة[،,]?\s*(?:وعلى|والحديث|وتابعوا| وعلى)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"تابعونا", "", text, flags=re.IGNORECASE)
    text = re.sub(r"النص المحسن:?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(وعلى|و|،|,)\s*", "", text)
    return text

def _process_audio_chunks(input_path: str, task_dir: str) -> list[str]:
    max_size_bytes = 20 * 1024 * 1024
    file_size = os.path.getsize(input_path)
    
    if file_size <= max_size_bytes:
        return [input_path]
        
    print(f"File is {file_size / (1024*1024):.2f} MB. Attempting compression...")
    compressed_path = os.path.join(task_dir, "compressed_input.mp3")
    audio = AudioSegment.from_file(input_path)
    audio.export(compressed_path, format="mp3", bitrate="64k", parameters=["-ac", "1"])
    
    compressed_size = os.path.getsize(compressed_path)
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
        
    return chunk_paths

def _transcribe_sync(groq_client: Groq, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            model=GROQ_AUDIO_MODEL, file=f, response_format="text", language=TRANSCRIPT_LANGUAGE
        )
    return resp.strip() if isinstance(resp, str) else getattr(resp, "text", "").strip()

def _chunk_text(text: str, max_words: int = 50000) -> list[str]:
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

async def _format_sync_async(mistral_client: Mistral, raw_text: str) -> str:
    chunks = _chunk_text(raw_text, max_words=50000)
    formatted_parts = []
    
    for i, chunk in enumerate(chunks):
        try:
            if i > 0:
                await asyncio.sleep(14)
                
            chat_response = mistral_client.chat.complete(
                model=MISTRAL_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": _FORMAT_SYSTEM},
                    {"role": "user", "content": f"النص:\n\n{chunk}"}
                ],
                temperature=0.3,
                max_tokens=262144
            )
            formatted_parts.append(chat_response.choices[0].message.content or "")
        except Exception as e:
            print(f"Error formatting chunk {i}: {e}")
            formatted_parts.append(chunk)

    return _strip_code_fences("\n\n".join(formatted_parts))

# ---------------------------------------------------------------------------
# Core Processing Engine (Now handles download sequentially)
# ---------------------------------------------------------------------------
async def process_audio_task(message, task_dir: str):
    # Determine file extension
    if message.voice:
        file_ext = "ogg"
    elif message.audio:
        file_ext = message.audio.file_name.split('.')[-1] if message.audio.file_name else "mp3"
    else:
        file_ext = message.document.file_name.split('.')[-1] if message.document.file_name else "mp3"
        
    file_path = os.path.join(task_dir, f"input.{file_ext}")
    
    # 1. Download (Sequential to save disk space)
    status_msg = await message.reply("⬇️ جاري تحميل الملف من تيليجرام...")
    try:
        await message.download(file_name=file_path)
        await status_msg.edit_text("⚙️ جاري تحليل حجم الملف وضغطه...")
        
        # 2. Compress/Slice Audio
        chunk_paths = await asyncio.to_thread(_process_audio_chunks, file_path, task_dir)
        
        # 3. Transcribe
        groq_client = Groq(api_key=GROQ_API_KEY)
        raw_text_parts = []
        for i, path in enumerate(chunk_paths):
            if len(chunk_paths) > 1:
                await status_msg.edit_text(f"🎙️ جاري التفريغ الصوتي للجزء {i+1}/{len(chunk_paths)}...")
            else:
                await status_msg.edit_text("🎙️ جاري التفريغ الصوتي عبر Whisper...")
            part_text = await asyncio.to_thread(_transcribe_sync, groq_client, path)
            raw_text_parts.append(part_text)
            
        raw_text = " ".join(raw_text_parts)
        raw_text = _clean_whisper_artifacts(raw_text)
        
        # 4. Format
        await status_msg.edit_text("✨ جاري إعادة البناء عبر Mistral Large 2512...")
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        formatted_text = await _format_sync_async(mistral_client, raw_text)
        
        # 5. Send back
        await status_msg.edit_text("📤 جاري إرسال النص النهائي...")
        paragraphs = formatted_text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > 3900:
                if current_chunk:
                    await message.reply_text(current_chunk.strip(), parse_mode=ParseMode.MARKDOWN)
                    current_chunk = para + "\n\n"
                else:
                    sentences = para.split('.')
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > 3900:
                            if current_chunk:
                                await message.reply_text(current_chunk.strip(), parse_mode=ParseMode.MARKDOWN)
                            current_chunk = sentence + ". "
                        else:
                            current_chunk += sentence + ". "
            else:
                current_chunk += para + "\n\n"
        
        if current_chunk:
            await message.reply_text(current_chunk.strip(), parse_mode=ParseMode.MARKDOWN)
            
        await status_msg.edit_text("✅ تمت العملية بنجاح!")

    except Exception as e:
        await status_msg.edit_text(f"❌ خطأ في المعالجة: {str(e)[:200]}")
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
        "👋 أهلاً بك في بوت التفريغ والتحرير الذكي!\n\n"
        "أرسل لي ملفاً صوتياً أو صوتاً (Voice) وسأقوم بتفريغه وتنقيحه بأعلى جودة.\n"
        "📌 *ملاحظة:* يعمل البوت بنظام الطابور (First-Come-First-Served) لضمان أعلى جودة وعدم استنفاد الموارد."
    )

@app.on_message(filters.audio | filters.voice | filters.document)
async def handle_audio(client, message):
    global worker_started
    
    # Start the background worker if it's not running yet
    if not worker_started:
        asyncio.create_task(queue_worker())
        worker_started = True

    if message.document and not message.document.mime_type.startswith("audio/"):
        return

    # Create task directory
    task_id = str(uuid.uuid4())
    task_dir = os.path.join("temp_audio", task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Add to Queue
    await task_queue.put((message, task_dir))
    
    # Notify user of their position
    position = task_queue.qsize()
    if position == 1:
        await message.reply_text("✅ **تم استلام طلبك!**\n🚀 أنت الأول في الطابور، جاري بدء العمل على ملفك الآن...")
    else:
        await message.reply_text(
            f"✅ **تم استلام طلبك وإضافته للطابور!**\n\n"
            f"📊 موقعك الحالي: **{position}**\n"
            f"⏳ سأبدأ العمل على ملفك فور انتهاء الملفات السابقة.\n"
            f"*(يرجى عدم إرسال ملفات أخرى حتى ينتهي هذا الملف لتجنب ازدحام الطابور)*"
        )

def main():
    if not all([API_ID, API_HASH, SESSION_STRING, GROQ_API_KEY, MISTRAL_API_KEY]):
        print("ERROR: Missing environment variables!")
        return

    print("Starting Pyrogram Client with Queue System...")
    app.run()

if __name__ == "__main__":
    main()