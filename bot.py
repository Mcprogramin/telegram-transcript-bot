"""
Telegram Bot → Obsidian Faithful Transcript Agent
=================================================
Users send a YouTube link. The bot queues it, processes it, 
and sends back the formatted .md file.
"""

import os
import re
import asyncio
import subprocess
import tempfile
import shutil
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

# Telegram & Async imports
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# AI & Tools imports
from groq import Groq
from google import genai
import yt_dlp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = "gemini-2.5-flash"
TRANSCRIPT_LANGUAGE = os.getenv("TRANSCRIPT_LANGUAGE", "ar").strip() or None

# Fixed Regex Patterns
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[^\n]*\n?", re.MULTILINE)

# The Arabic Prompt
_FORMAT_SYSTEM = """أنت مُعيد بناء لغوي عربي نخبوي ومُحرر نصوص محادثات. هدفك الوحيد هو تحويل النصوص الخام الناتجة عن تحويل الكلام إلى نص (STT) إلى نثر بشري دقيق، منطقي، وطبيعي التدفق.

### بروتوكول المعالجة الرباعية (إلزامي)
لا تُخرج النص فوراً. قم عقلياً بتشغيل هذه الخطوات الأربع على النص المدخل قبل كتابة ردك:
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

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name[:80].strip() or "transcript"

def _strip_code_fences(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = text.replace("```", "")
    return text.strip()

def _extract_audio_ffmpeg(src: str, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", out_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace')[-200:]}")

def _download_youtube_sync(url: str, out_dir: str) -> str:
    outtmpl = os.path.join(out_dir, "%(title).80s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "quiet": True, "no_warnings": True, "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not os.path.exists(filename):
            for ext in ("m4a", "mp3", "webm"):
                candidate = Path(filename).with_suffix(f".{ext}")
                if candidate.exists(): return str(candidate)
        return filename

def _transcribe_sync(client: Groq, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=GROQ_AUDIO_MODEL, file=f, response_format="text",
            language=TRANSCRIPT_LANGUAGE
        )
    return resp.strip() if isinstance(resp, str) else getattr(resp, "text", "").strip()

def _format_sync(client: genai.Client, raw_text: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[_FORMAT_SYSTEM, "---\n\nالنص:\n\n" + raw_text],
        config={"temperature": 0.3}
    )
    return _strip_code_fences(response.text or "")

async def send_long_text(bot, chat_id, text):
    limit = 4000
    for i in range(0, len(text), limit):
        await bot.send_message(chat_id=chat_id, text=text[i:i+limit], parse_mode="Markdown")

# ---------------------------------------------------------------------------
# The Core Processing Engine
# ---------------------------------------------------------------------------
async def process_video_task(chat_id: int, message_id: int, url: str, bot):
    status_msg_id = None
    
    async def send_status(text):
        nonlocal status_msg_id
        try:
            if status_msg_id:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text=text)
            else:
                msg = await bot.send_message(chat_id=chat_id, text=text)
                status_msg_id = msg.message_id
        except Exception:
            pass

    await send_status("📥 Added to queue. Starting...")

    try:
        await send_status("⬇️ Downloading audio from YouTube...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_file = await asyncio.to_thread(_download_youtube_sync, url, tmp_dir)
            
            await send_status("⚙️ Converting audio with ffmpeg...")
            mp3_path = os.path.join(tmp_dir, "converted.mp3")
            await asyncio.to_thread(_extract_audio_ffmpeg, audio_file, mp3_path)

            await send_status("🎙️ Transcribing with Groq Whisper...")
            groq_client = Groq(api_key=GROQ_API_KEY)
            raw_text = await asyncio.to_thread(_transcribe_sync, groq_client, mp3_path)

            await send_status("✨ Formatting & fixing Arabic with Gemini...")
            gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
            formatted_text = await asyncio.to_thread(_format_sync, gemini_client, raw_text)

            await send_status("📤 Sending transcript...")
            safe_name = _sanitize_filename(url)
            final_text = f"**Transcript for:** {url}\n\n{formatted_text}"
            
            local_out = Path("Transcripts")
            local_out.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            local_path = local_out / f"{ts}_{safe_name}.md"
            local_path.write_text(final_text, encoding="utf-8")

            await send_long_text(bot, chat_id, final_text)
            await bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text="✅ Finished!")

    except Exception as e:
        error_msg = f"❌ Error: {str(e)[:200]}"
        if status_msg_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text=error_msg)
        else:
            await bot.send_message(chat_id=chat_id, text=error_msg)

# ---------------------------------------------------------------------------
# Telegram Bot Handlers
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome! I am the Obsidian Transcript Bot.\n\n"
        "Send me a YouTube link, and I will transcribe, format, and fix the Arabic text for you."
    )

async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.text:
        return
        
    url = update.message.text.strip()
    if not url.startswith("http"):
        return
        
    chat_id = update.message.chat_id
    msg_id = update.message.message_id
    
    await update.message.reply_text(f"📥 Received link! Adding to queue...")
    
    # Use bot_data to store the queue safely
    context.application.bot_data['queue'].put_nowait((chat_id, msg_id, url))

async def post_init(application: Application):
    # Initialize the queue in bot_data
    application.bot_data['queue'] = asyncio.Queue()
    # Start the background worker
    application.create_task(worker_loop(application))
    print("Bot started. Background worker running.")

async def worker_loop(application: Application):
    while True:
        chat_id, msg_id, url = await application.bot_data['queue'].get()
        try:
            await process_video_task(chat_id, msg_id, url, application.bot)
        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            application.bot_data['queue'].task_done()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not GROQ_API_KEY or not GOOGLE_API_KEY:
        print("ERROR: Missing API keys in .env file!")
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
    
    if token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("Please set TELEGRAM_BOT_TOKEN in your .env file!")
        return

    application = Application.builder().token(token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))

    application.post_init = post_init

    print("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()