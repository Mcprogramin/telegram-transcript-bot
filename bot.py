"""
Telegram MTProto Audio Transcript Bot (Mistral Stack)
======================================================
STT: Groq Whisper-large-v3-turbo (Best Arabic accuracy)
Text: Mistral Small 3.1 (Excellent Arabic, 128K context, no TPM limits)
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
from mistralai.client import Mistral  # 👈 CORRECTED IMPORT

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
MISTRAL_TEXT_MODEL = "mistral-small-2506"
TRANSCRIPT_LANGUAGE = os.getenv("TRANSCRIPT_LANGUAGE", "ar").strip() or None

# Fixed Regex Patterns
_THINK_RE = re.compile(r"</think>")
_FENCE_RE = re.compile(r"^```[^\n]*\n?", re.MULTILINE)

# The Arabic Prompt
_FORMAT_SYSTEM = """أنت "مُعيد بناء" (Reconstructor)، لست محرراً ولا مدققاً. مدخلاتك ليست نصاً كتبه إنسان — هي خراطة صوتية (phonetic garbage) أخرجها نظام Whisper، وهي مليئة بالأخطاء الصوتية والهلوسات والترقيم العشوائي. مهمتك: استعادة ما أراد الشيخ قوله، لا حفظ ما قاله Whisper.

━━━━━━━━━━━━━━━━━━━━━━━━

القاعدة الأولى — إعادة بناء الترقيم من الصفر (الأهم):

كل نقطة (.) وكل فاصلة (،) في المدخل هي أثر Whisper، وهي خاطئة بالكامل. تجاهلها كأنها غير موجودة.
اقرأ المعنى، ثم ابنِ الترقيم من الصفر بناءً على الفكرة لا على الوقفة الصوتية.

أمثلة إلزامية — هذه الأنماط تُعالَج هكذا دائماً:
- "يقهر هذا. بهذا."        ←  "يقهر هذا بهذا"
- "ولو كان. ولو كان. ولو كان." ←  "ولو كان ولو كان ولو كان"
- "بعدا. بعدا. سحقا. سحقا." ←  "بُعدًا بُعدًا، سُحقًا سُحقًا"
- "تعبٌ. وعناء. وشقاء."    ←  "تعبٌ وعناءٌ وشقاء"

القاعدة: الوقفةُ الصوتية ليست نهاية جملة. الفكرةُ الكاملة هي نهاية الجملة.

━━━━━━━━━━━━━━━━━━━━━━━━

القاعدة الثانية — التصحيح الصوتي العميق:

كل كلمة غريبة أو لا تناسب السياق هي خطأ صوتي من Whisper — لا كلمة لكاتب يجب الحفاظ عليها.
لديك صلاحية كاملة لإعادة كتابة أي كلمة إذا كانت صوتياً مشابهة لمصطلح عربي أو إسلامي يناسب السياق.

اسأل نفسك عن كل كلمة مشكوك فيها: "ما الكلمة العربية أو المصطلح الإسلامي الذي يُشبه هذا الصوت ويناسب هذا السياق اللاهوتي؟"

أنماط الأخطاء الصوتية الشائعة:
- "المحدون"      ←  "المُوَحِّدون"
- "الأسقياء"     ←  "الأتقياء"
- "الميشان" / "لميشاء"  ←  "لِمَن يشاء"
- "الضده" / "بالضده"  ←  "ضِدَّه" / "بضِدِّه"
- أي عبارة قرآنية مشوهة صوتياً → أعد بناءها إلى الآية الصحيحة
- أي عبارة حديثية مشوهة صوتياً → أعد بناءها إلى النص المعروف

━━━━━━━━━━━━━━━━━━━━━━━━

القاعدة الثالثة — تنسيق الشعر:

البيت الشعري العربي يتكون من شطرين، لا من أسطر متقطعة.
كل بيت في سطر واحد مستقل، الشطر الأول والثاني مفصولان بمسافة واسعة أو بـ(   ***   ).

صحيح:
فلا تغتر بالدنيا   ***   فإنها دار فناء
وكم من صحيح مات من غير علة   ***   وكم من سقيم عاش حيناً من الدهر

خاطئ (لا تفعل هذا أبداً):
فلا تغتر
بالدنيا
فإنها
دار فناء

━━━━━━━━━━━━━━━━━━━━━━━━

القاعدة الرابعة — تنسيق النصوص الشرعية:
- الآيات القرآنية: «نص الآية»
- الأحاديث النبوية: "نص الحديث"

━━━━━━━━━━━━━━━━━━━━━━━━

القاعدة الخامسة — الحذف الجذري:
احذف بلا تردد: إشعارات الاشتراك، أسماء المترجمين، أي عبارة استهلالية أو ختامية أدرجها Whisper.
لا تُخرج أي عنوان مثل "النص المحسن" أو أي تعليق.

━━━━━━━━━━━━━━━━━━━━━━━━

القاعدة السادسة — الحفاظ على الروح:
احفظ كل فكرة ودقيقة عقدية وأسلوب الشيخ. لا تحذف أي مضمون.
تخلص فقط من التلعثم عديم المعنى.

━━━━━━━━━━━━━━━━━━━━━━━━

المخرج: النص المُعاد بناؤه فقط، بلا عناوين ولا ملاحظات ولا كود markdown."""
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

def _chunk_text(text: str, max_words: int = 3000) -> list[str]:
    """Splits text into chunks. Mistral has 128K context so we can use larger chunks."""
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
    """Formats text using Mistral Small 3.1 (no TPM limits, just 1 req/sec)."""
    chunks = _chunk_text(raw_text, max_words=3000)
    formatted_parts = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Add 1.5s delay to respect Mistral's 1 request/second limit
            if i > 0:
                await asyncio.sleep(1.5)
                
            chat_response = mistral_client.chat.complete(
                model=MISTRAL_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": _FORMAT_SYSTEM},
                    {"role": "user", "content": f"النص:\n\n{chunk}"}
                ],
                temperature=0.3,
                max_tokens=16000
            )
            
            formatted_parts.append(chat_response.choices[0].message.content or "")
            
        except Exception as e:
            print(f"Error formatting chunk {i}: {e}")
            # If Mistral fails, keep the raw text as fallback
            formatted_parts.append(chunk)

    return _strip_code_fences(" ".join(formatted_parts))

# ---------------------------------------------------------------------------
# Core Processing Engine
# ---------------------------------------------------------------------------
async def process_audio_task(message, file_path: str, task_dir: str):
    status_msg = await message.reply("⚙️ Processing audio size...")
    
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
        
        # 3. Format with Mistral Small 3.1
        await status_msg.edit_text("✨ Formatting with Mistral Small 3.1...")
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        formatted_text = await _format_sync_async(mistral_client, raw_text)
        
        # 4. Send back to Telegram (smart chunking by paragraphs)
        await status_msg.edit_text("📤 Sending transcript...")
        
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
        "👋 Welcome to the Mistral Audio Bot!\n\n"
        "I use Groq Whisper for transcription and Mistral Small 3.1 for elite Arabic formatting.\n"
        "I support massive files (up to 2GB) with excellent Arabic quality!"
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
    if not all([API_ID, API_HASH, SESSION_STRING, GROQ_API_KEY, MISTRAL_API_KEY]):
        print("ERROR: Missing environment variables!")
        return

    print("Starting Pyrogram Client...")
    app.run()

if __name__ == "__main__":
    main()