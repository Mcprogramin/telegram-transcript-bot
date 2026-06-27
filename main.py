"""
Long Videos → Obsidian Faithful Transcript Agent
=================================================
Downloads from YouTube or processes local files (30min - 3h).
Creates FAITHFUL transcripts with Quran/Hadith formatting.
Uses Groq Whisper for transcription + Gemini 2.5 Flash for formatting.
"""

import os
import re
import time
import datetime
import threading
import traceback
import subprocess
import tempfile
import shutil
from pathlib import Path
from queue import Queue
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except ImportError:
    raise SystemExit("Run: pip install customtkinter")

try:
    from dotenv import load_dotenv
except ImportError:
    raise SystemExit("Run: pip install python-dotenv")

try:
    from groq import Groq
except ImportError:
    raise SystemExit("Run: pip install groq")

try:
    from google import genai
except ImportError:
    raise SystemExit("Run: pip install google-genai")

try:
    import yt_dlp
except ImportError:
    raise SystemExit("Run: pip install yt-dlp")

load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_AUDIO_MODEL: str = "whisper-large-v3-turbo"
GEMINI_MODEL: str = "gemini-2.5-flash"

MAX_FILE_SIZE_MB = 24
MAX_QUEUE_SIZE = 50
CHUNK_MINUTES = 14
AUDIO_BITRATE = "64k"
AUDIO_SAMPLE_RATE = 16000

# Read Obsidian path from .env, fallback to local folder
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
if not OBSIDIAN_VAULT_PATH:
    OBSIDIAN_VAULT_PATH = str(Path(__file__).resolve().parent / "Transcripts")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4a", ".mp3", ".wav", ".flac", ".ogg"}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE  = re.compile(r"^```[^\n]*\n?", re.MULTILINE)

def _strip_code_fences(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = text.replace("```", "")
    return text.strip()


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().strip(".")
    return name[:120] or "transcript"


# ---------------------------------------------------------------------------
# FFmpeg helpers (Windows UTF-8 safe)
# ---------------------------------------------------------------------------
def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH. Install it first.")
    return path


def _extract_audio(ffmpeg: str, src: str, out_path: str) -> None:
    """Convert any input to 16kHz mono mp3 at 64kbps."""
    cmd = [
        ffmpeg, "-y", "-i", src,
        "-vn", "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-b:a", AUDIO_BITRATE,
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg extract failed: {stderr[-500:]}")


def _get_duration(ffmpeg: str, path: str) -> float:
    cmd = [ffmpeg, "-i", path, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True)
    # Decode stderr as UTF-8 (ffmpeg outputs UTF-8, Windows default cp1252 crashes)
    stderr = result.stderr.decode("utf-8", errors="replace")
    m = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", stderr)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _split_audio(ffmpeg: str, src: str, chunk_dir: Path, chunk_minutes: int,
                 log) -> list[Path]:
    """Split into fixed-duration chunks for transcription."""
    dur = _get_duration(ffmpeg, src)
    if dur <= 0:
        raise RuntimeError(f"Could not determine audio duration of {src}")
    chunk_sec = chunk_minutes * 60
    n_chunks = int(dur // chunk_sec) + (1 if dur % chunk_sec > 1 else 0)
    log(f"Splitting into {n_chunks} chunks of {chunk_minutes} min each...")

    chunks: list[Path] = []
    pattern = str(chunk_dir / "chunk_%04d.mp3")
    cmd = [
        ffmpeg, "-y", "-i", src,
        "-f", "segment",
        "-segment_time", str(chunk_sec),
        "-c", "libmp3lame",
        "-b:a", AUDIO_BITRATE,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-reset_timestamps", "1",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg split failed: {stderr[-500:]}")

    for p in sorted(chunk_dir.glob("chunk_*.mp3")):
        chunks.append(p)
    if not chunks:
        raise RuntimeError("ffmpeg produced no chunks.")
    return chunks


# ---------------------------------------------------------------------------
# YouTube download
# ---------------------------------------------------------------------------
def _download_youtube(url: str, out_dir: Path, log) -> Path:
    log(f"Downloading: {url}")
    outtmpl = str(out_dir / "%(title).80s.%(ext)s")

    def progress(d):
        if d.get("status") == "downloading":
            pct = d.get("_percent_str", "?")
            log(f"  … downloading {pct}")
        elif d.get("status") == "finished":
            log(f"  ✓ download complete, processing…")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "progress_hooks": [progress],
        "nocheckcertificate": True,
        "retries": 10,
        "fragment_retries": 10,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if not Path(filename).exists():
                for ext in ("m4a", "mp3", "webm", "opus", "wav"):
                    candidate = Path(filename).with_suffix(f".{ext}")
                    if candidate.exists():
                        return candidate
            return Path(filename)
    except Exception as e:
        log(f"Download error: {e}")
        raise


# ---------------------------------------------------------------------------
# Groq: transcription (audio only)
# ---------------------------------------------------------------------------
def _transcribe_file(client: Groq, path: Path, language: str | None) -> str:
    with open(path, "rb") as f:
        kwargs = dict(model=GROQ_AUDIO_MODEL, file=f, response_format="text")
        if language:
            kwargs["language"] = language
        resp = client.audio.transcriptions.create(**kwargs)
    if isinstance(resp, str):
        return resp.strip()
    return getattr(resp, "text", str(resp)).strip()


def _transcribe_long(client: Groq, audio_path: Path, language: str | None,
                     log) -> str:
    ffmpeg = _require_ffmpeg()
    size_mb = audio_path.stat().st_size / (1024 * 1024)

    if size_mb <= MAX_FILE_SIZE_MB:
        log(f"Transcribing ({size_mb:.1f} MB, single pass)…")
        return _transcribe_file(client, audio_path, language)

    with tempfile.TemporaryDirectory(prefix="lt_chunks_") as td:
        chunk_dir = Path(td)
        chunks = _split_audio(ffmpeg, str(audio_path), chunk_dir,
                              CHUNK_MINUTES, log)
        parts: list[str] = []
        for i, c in enumerate(chunks, 1):
            log(f"  chunk {i}/{len(chunks)} ({c.stat().st_size/1024/1024:.1f} MB)…")
            parts.append(_transcribe_file(client, c, language))
        return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Gemini: formatting (text only)
# ---------------------------------------------------------------------------
_FORMAT_SYSTEM = """أنت مُعيد بناء لغوي عربي نخبوي ومُحرر نصوص محادثات. هدفك الوحيد هو تحويل النصوص الخام الناتجة عن تحويل الكلام إلى نص (STT) إلى نثر بشري دقيق، منطقي، وطبيعي التدفق.

### بروتوكول المعالجة الرباعية (إلزامي)
لا تُخرج النص فوراً. قم عقلياً بتشغيل هذه الخطوات الأربع على النص المدخل قبل كتابة ردك:

**[المرحلة الأولى: تحليل السياق الشامل]**
اقرأ النص كاملاً أولاً لفهم:
- الموضوع الرئيسي والهدف من الحديث
- اللهجة المستخدمة (خليجية، مصرية، شامية، مغاربية، عراقية، فصحى)
- المجال (ديني، سياسي، تقني، اجتماعي، تعليمي، حديث عادي)
- الشخصيات المتحدث عنها وعلاقاتهم
- التسلسل الزمني والمنطقي للأفكار

**[المرحلة الثانية: التحقق من المنطق والسياق]**
افحص كل جملة للتأكد من:
- هل الجملة منطقية في سياق الفقرة؟
- هل ترتبط بما قبلها وما بعدها؟
- هل المعنى العام واضح ومتماسك؟
- هل هناك تناقضات أو قفزات غير مبررة؟

إذا وجدت جملة لا معنى لها أو تناقض السياق:
- أعد قراءتها مع الجمل المحيطة
- استنتج ما كان المتحدث يقوله فعلاً بناءً على السياق
- أعد بناء الجملة لتتناسب منطقياً مع السياق

**[المرحلة الثالثة: التشريح الصوتي والنحوي]**
الذكاء الاصطناعي يرتكب "هلوسات صوتية" (يكتب كلمات عربية حقيقية تشبه الكلمة المنطوقة صوتياً لكنها لا معنى لها في السياق).
- ابحث عن الكلمات التي تبدو غريبة في سياقها
- انطقها صوتياً في "عقلك" وابحث عن الكلمة الصحيحة
- أصلح الأخطاء النحوية والصرفية
- أصلح الالتصاق (والليصار -> واللي صار) والفصل الخاطئ (ال محامي -> المحامي)

**[المرحلة الرابعة: بناء الجملة والإيقاع]**
الكلام المنطوق لا يحتوي على علامات ترقيم؛ النصوص الخام تبدو كجدار نصي متشنج.
- احذف البدايات الكاذبة، التأتأة، وتكرارات الذكاء الاصطناعي
- أدخل علامات الترقيم الطبيعية (فواصل، نقاط، علامات استفهام) لتمنح النص إيقاع تنفس بشري
- تأكد من أن كل فقرة لها فكرة واحدة متماسكة

***

### قوانين التصحيح الحاسمة

**1. قانون المنطق السياقي:**
كل جملة يجب أن تكون منطقية في سياقها. إذا كانت الجملة تبدو "آمنة" لغوياً لكنها لا معنى لها في السياق، يجب إعادة بنائها.

**2. قانون الحفاظ على اللهجة:**
التصحيح العدواني ينطبق على الأخطاء، وليس على الأسلوب. لا "ترقّ" الحديث العامي إلى الفصحى.
- حافظ على نكهة اللهجة الإقليمية 100%
- أصلح القواعد المكسورة، لكن حافظ على قواعد اللهجة

**3. قانون التماسك المنطقي:**
الأفكار يجب أن تتدفق بشكل منطقي. إذا وجدت قفزة غير مبررة:
- ابحث عن الجملة المفقودة أو المشوهة
- أعد بناء الانتقال المنطقي
- تأكد من أن القارئ يمكنه متابعة الفكرة دون ارتباك

**4. قانون الدقة المصطلحية:**
- المصطلحات الدينية: استخدم المصطلحات الدقيقة
- المصطلحات السياسية: استخدم المصطلحات الصحيحة
- المصطلحات التقنية: استخدم التحويل العربي المقبول
- القرآن في «...»، الحديث في "..."

**5. قانون التنسيق البصري:**
قسّم النص إلى فقرات موضوعية نظيفة. كل مرة يتحول المتحدث إلى:
- موضوع فرعي جديد
- نقطة منطقية جديدة
- تحول في الزمن أو المكان
اضغط Enter لبدء فقرة جديدة.

***

### قيود الإخراج الحاسمة
- لا تُخرج أفكارك في المراحل الأربع
- لا تقل "إليك النص المصحح:" أو أي مقدمة
- لا تلخص، لا تقصّر، لا تحذف، ولا تترك ملاحظات بديلة مثل [غير مسموع]
- لا تضف عناوين أو تعليقات أو شروحات
- أخرج النص العربي المُعاد بناؤه والمُصحح فقط
- حافظ على كل فكرة قالها المتحدث مع تصحيح الأخطاء المنطقية واللغوية فقط
- لا تضف محتوى أو تحذف محتوى

أخرج النص المُعاد بناؤه فقط. لا مقدمات. لا ملاحظات."""


def _format_transcript(gemini_client, raw: str, log) -> str:
    """Single-pass formatting with Gemini's massive context window."""
    if not raw.strip():
        return raw

    char_count = len(raw)
    log(f"Formatting transcript ({char_count:,} chars, single pass with Gemini)…")

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                _FORMAT_SYSTEM,
                "---\n\nالنص المطلوب معالجته:\n\n" + raw,
            ],
            config={
                "temperature": 0.3,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        out = response.text or ""
        return _strip_code_fences(out)
    except Exception as e:
        log(f"Gemini error: {e}. Retrying without thinking config…")
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                _FORMAT_SYSTEM,
                "---\n\nالنص المطلوب معالجته:\n\n" + raw,
            ],
            config={"temperature": 0.3},
        )
        out = response.text or ""
        return _strip_code_fences(out)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.queue: Queue = Queue()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def log(self, msg: str):
        self.app.after(0, self.app._append_log, msg)

    def set_status(self, msg: str):
        self.app.after(0, self.app._set_status, msg)

    def set_progress(self, value: float):
        self.app.after(0, self.app._set_progress, value)

    def run(self):
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        google_key = os.getenv("GOOGLE_API_KEY", "").strip()

        if not groq_key:
            self.log("ERROR: GROQ_API_KEY not set in .env")
            self.set_status("Missing GROQ_API_KEY")
            return
        if not google_key:
            self.log("ERROR: GOOGLE_API_KEY not set in .env")
            self.set_status("Missing GOOGLE_API_KEY")
            return

        groq_client = Groq(api_key=groq_key)
        gemini_client = genai.Client(api_key=google_key)
        language = os.getenv("TRANSCRIPT_LANGUAGE", "").strip() or None

        out_root = Path(OBSIDIAN_VAULT_PATH)
        out_root.mkdir(parents=True, exist_ok=True)
        self.log(f"Output folder: {out_root}")
        self.log(f"Audio: {GROQ_AUDIO_MODEL} (Groq) | Format: {GEMINI_MODEL} (Google)")

        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except Exception:
                continue
            if item is None:
                break
            try:
                self._process(groq_client, gemini_client, item, out_root, language)
            except Exception as e:
                self.log(f"ERROR on '{item.get('label')}': {e}")
                traceback.print_exc()
            finally:
                self.queue.task_done()
                self.app.after(0, self.app._pop_queue_head)

        self.set_status("Idle")
        self.set_progress(0)

    def _process(self, groq_client: Groq, gemini_client, item: dict,
                 out_root: Path, language: str | None):
        label = item["label"]
        source = item["source"]
        kind = item["kind"]

        self.set_status(f"Working: {label}")
        self.set_progress(0.05)

        with tempfile.TemporaryDirectory(prefix="lt_src_") as td:
            tmp = Path(td)
            if kind == "url":
                self.log(f"[{label}] Downloading from YouTube…")
                local = _download_youtube(source, tmp, self.log)
            else:
                local = Path(source)
                if not local.exists():
                    raise FileNotFoundError(local)

            self.log(f"[{label}] Extracting audio…")
            audio_full = tmp / "full_audio.mp3"
            ffmpeg = _require_ffmpeg()
            _extract_audio(ffmpeg, str(local), str(audio_full))
            self.set_progress(0.25)

            self.log(f"[{label}] Transcribing with Groq Whisper (may take a while)…")
            raw = _transcribe_long(groq_client, audio_full, language, self.log)
            self.set_progress(0.65)

            self.log(f"[{label}] Formatting with Gemini 2.5 Flash (single pass)…")
            formatted = _format_transcript(gemini_client, raw, self.log)
            self.set_progress(0.95)

            # Save
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            safe = _sanitize_filename(label)
            out_path = out_root / f"{ts} - {safe}.md"

            header = (
                f"# {label}\n\n"
                f"- Source: {source}\n"
                f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                f"- Audio model: {GROQ_AUDIO_MODEL} (Groq)\n"
                f"- Format model: {GEMINI_MODEL} (Google)\n\n---\n\n"
            )
            out_path.write_text(header + formatted, encoding="utf-8")
            self.log(f"[{label}] ✓ Saved: {out_path}")
            self.set_progress(1.0)
            self.set_status(f"Done: {label}")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Long Videos → Obsidian Faithful Transcript")
        self.geometry("820x640")
        self.minsize(720, 560)

        self._build_ui()
        self.worker = Worker(self)
        self.worker.start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        frm_url = ctk.CTkFrame(self)
        frm_url.pack(fill="x", **pad)
        ctk.CTkLabel(frm_url, text="YouTube URL:").pack(side="left", padx=(8, 6))
        self.ent_url = ctk.CTkEntry(frm_url, placeholder_text="https://youtube.com/…")
        self.ent_url.pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkButton(frm_url, text="Add URL", width=110,
                      command=self._add_url).pack(side="left", padx=(6, 8))

        frm_file = ctk.CTkFrame(self)
        frm_file.pack(fill="x", **pad)
        ctk.CTkButton(frm_file, text="Add local file(s)…", width=180,
                      command=self._add_files).pack(side="left", padx=8)
        ctk.CTkLabel(frm_file,
                     text=f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}"
                     ).pack(side="left", padx=8)

        frm_q = ctk.CTkFrame(self)
        frm_q.pack(fill="both", expand=True, **pad)
        ctk.CTkLabel(frm_q, text="Queue",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.lst_queue = ctk.CTkTextbox(frm_q, height=120, state="disabled")
        self.lst_queue.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        ctk.CTkButton(frm_q, text="Remove selected", width=150,
                      command=self._remove_selected).pack(side="left", padx=8, pady=(0, 6))
        ctk.CTkButton(frm_q, text="Clear queue", width=130,
                      command=self._clear_queue).pack(side="left", padx=4, pady=(0, 6))

        frm_p = ctk.CTkFrame(self)
        frm_p.pack(fill="x", **pad)
        self.lbl_status = ctk.CTkLabel(frm_p, text="Idle")
        self.lbl_status.pack(anchor="w", padx=8)
        self.bar = ctk.CTkProgressBar(frm_p)
        self.bar.pack(fill="x", padx=8, pady=(4, 8))
        self.bar.set(0)

        frm_log = ctk.CTkFrame(self)
        frm_log.pack(fill="both", expand=True, **pad)
        ctk.CTkLabel(frm_log, text="Log",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.txt_log = ctk.CTkTextbox(frm_log, state="disabled",
                                      font=ctk.CTkFont(family="Consolas", size=12))
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._items: list[dict] = []

    def _add_url(self):
        url = self.ent_url.get().strip()
        if not url:
            return
        if self.worker.queue.qsize() >= MAX_QUEUE_SIZE:
            messagebox.showwarning("Queue full", f"Max {MAX_QUEUE_SIZE} items.")
            return
        item = {"kind": "url", "source": url, "label": url}
        self._enqueue(item)
        self.ent_url.delete(0, "end")

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select video/audio files",
            filetypes=[("Media", " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))),
                       ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            if self.worker.queue.qsize() >= MAX_QUEUE_SIZE:
                messagebox.showwarning("Queue full",
                                       f"Stopped at {MAX_QUEUE_SIZE} items.")
                break
            item = {"kind": "file", "source": p, "label": Path(p).name}
            self._enqueue(item)

    def _enqueue(self, item: dict):
        self._items.append(item)
        self.worker.queue.put(item)
        self._refresh_queue_view()
        self._append_log(f"+ Queued: {item['label']}")

    def _remove_selected(self):
        messagebox.showinfo("Info", "Use 'Clear queue' to remove pending items.")

    def _clear_queue(self):
        cleared = 0
        while True:
            try:
                self.worker.queue.get_nowait()
                self.worker.queue.task_done()
                cleared += 1
            except Exception:
                break
        self._items.clear()
        self._refresh_queue_view()
        if cleared:
            self._append_log(f"Cleared {cleared} pending item(s).")

    def _pop_queue_head(self):
        if self._items:
            self._items.pop(0)
        self._refresh_queue_view()

    def _refresh_queue_view(self):
        self.lst_queue.configure(state="normal")
        self.lst_queue.delete("1.0", "end")
        if not self._items:
            self.lst_queue.insert("end", "(queue empty)")
        else:
            for i, it in enumerate(self._items, 1):
                tag = "YT" if it["kind"] == "url" else "FILE"
                self.lst_queue.insert("end", f"{i:2d}. [{tag}] {it['label']}\n")
        self.lst_queue.configure(state="disabled")

    def _append_log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"[{ts}] {msg}\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _set_status(self, msg: str):
        self.lbl_status.configure(text=msg)

    def _set_progress(self, v: float):
        self.bar.set(max(0.0, min(1.0, v)))

    def _on_close(self):
        self.worker.stop()
        try:
            self.worker.queue.put_nowait(None)
        except Exception:
            pass
        self.destroy()


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in .env")
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not found in .env")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()