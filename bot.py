"""
Telegram Audio Transcript Bot
=============================
Users send an audio file or voice note. The bot queues it, 
transcribes it with Groq, formats it with Gemini, and sends it back.
"""

import os
import re
import asyncio
import shutil
import datetime
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from groq import Groq
from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = "gemini-2.5-flash"
TRANSCRIPT_LANGUAGE = os.getenv("TRANSCRIPT_LANGUAGE", "ar").strip() or None

# Fixed Regex Patterns
_THINK_RE = re.compile(r"
</think>
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

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = text.replace("```", "")
    return text.strip()

def _transcribe_sync(client: Groq, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=GROQ_AUDIO_MODEL, 
            file=f, 
            response_format="text",
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
# Core Processing Engine
# ---------------------------------------------------------------------------
async def process_audio_task(chat_id: int, message_id: int, file_path: str, task_dir: str, bot):
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

    try:
        # 1. Transcribe (Groq)
        await send_status("🎙️ Transcribing with Groq Whisper...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        raw_text = await asyncio.to_thread(_transcribe_sync, groq_client, file_path)

        # 2. Format (Gemini)
        await send_status("✨ Formatting & fixing Arabic with Gemini...")
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        formatted_text = await asyncio.to_thread(_format_sync, gemini_client, raw_text)

        # 3. Send to user
        await send_status("📤 Sending transcript...")
        final_text = f"**Transcript:**\n\n{formatted_text}"
        
        # Save locally for records
        local_out = Path("Transcripts")
        local_out.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        local_path = local_out / f"{ts}_{chat_id}.md"
        local_path.write_text(final_text, encoding="utf-8")

        await send_long_text(bot, chat_id, final_text)
        await bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text="✅ Finished!")

    except Exception as e:
        error_msg = f"❌ Error: {str(e)[:200]}"
        if status_msg_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text=error_msg)
        else:
            await bot.send_message(chat_id=chat_id, text=error_msg)
            
    finally:
        # Clean up the temporary audio file after processing
        try:
            shutil.rmtree(task_dir)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Telegram Bot Handlers
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to the Audio Transcript Bot!\n\n"
        "Send me an audio file or a voice note, and I will transcribe, format, and fix the Arabic text for you."
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Determine if it's a voice note or an audio file
    if update.message.voice:
        file = await update.message.voice.get_file()
        file_ext = "ogg"
    elif update.message.audio:
        file = await update.message.audio.get_file()
        file_ext = update.message.audio.file_name.split('.')[-1] if update.message.audio.file_name else "mp3"
    else:
        await update.message.reply_text("Please send a valid audio file or voice note.")
        return

    chat_id = update.message.chat_id
    
    # Create a unique ID for this file to avoid collisions
    task_id = str(uuid.uuid4())
    task_dir = os.path.join("temp_audio", task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    file_path = os.path.join(task_dir, f"input.{file_ext}")
    
    ack_msg = await update.message.reply_text("⬇️ Downloading audio from Telegram...")
    await file.download_to_drive(file_path)
    
    await ack_msg.edit_text(f"📥 Received audio! Adding to queue...")
    
    # Add to the background queue
    context.application.bot_data['queue'].put_nowait((chat_id, update.message.message_id, file_path, task_dir))

async def post_init(application: Application):
    application.bot_data['queue'] = asyncio.Queue()
    asyncio.create_task(worker_loop(application))
    print("Queue initialized and background worker started.")

async def worker_loop(application: Application):
    while True:
        chat_id, msg_id, file_path, task_dir = await application.bot_data['queue'].get()
        try:
            await process_audio_task(chat_id, msg_id, file_path, task_dir, application.bot)
        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            application.bot_data['queue'].task_done()

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"Exception while handling an update: {context.error}")

def main():
    if not GROQ_API_KEY or not GOOGLE_API_KEY:
        print("ERROR: Missing API keys in .env file!")
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
    if token == "YOUR_TELEGRAM_BOT_TOKEN_HERE": return

    application = Application.builder().token(token).build()
    
    # We only need the start command and the audio handler now!
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))
    
    application.post_init = post_init
    application.add_error_handler(error_handler)

    print("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()