"""Gradio app for dots.tts SOAR: voice cloning and narration, saved by name.

Run it with `python app.py`. The Colab notebook sets these first (all optional):

  DOTS_OUTPUT_ROOT  where project folders are created (a Google Drive folder on Colab)
  DOTS_PROJECT      default project name shown in the app
  DOTS_PRECISION    auto | bfloat16 | float16 | float32
  DOTS_OPTIMIZE     1 to use torch.compile (Ampere or newer GPUs only)
  DOTS_SHARE        1 to create a public gradio.live link (always on in Colab)
  DOTS_PASSWORD     if set, the app asks for username "dots" and this password
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
import torch

import engine
import narration
import tts_utils
from engine import SynthesisError

MODEL_ID = "dots-studio/dots.tts-soar"
APP_DIR = Path(__file__).resolve().parent
# Trimmed references, voice-lock clips, and copies of takes shown in the UI.
# It lives in the system temp dir on purpose: Gradio may copy files from there
# into its cache (served under unguessable hashed paths) but never serves the
# folder itself. The Drive output folder is never exposed to the share link.
WORK_DIR = Path(tempfile.gettempdir()) / "dots_tts_work"
AUTH_USERNAME = "dots"
PRECISIONS = ("bfloat16", "float16", "float32")

CLONE_FULL = "Clone: reference audio + transcript (best)"
CLONE_TIMBRE = "Clone: reference audio only"
RANDOM_VOICE = "Random voice (no reference)"

logger = logging.getLogger("dots_colab")


# ---------------------------------------------------------------------------
# Configuration and model loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    output_root: Path
    default_project: str
    precision: str
    optimize: bool
    share: bool
    password: str = ""


def pick_precision(requested: str | None, capability: tuple[int, int] | None) -> str:
    """Resolve "auto" for the GPU at hand; `capability` is None without a GPU.

    bfloat16 is the model's native precision, but GPUs older than Ampere (the
    Colab T4) only emulate it, while their tensor cores run float16 natively.
    float16 output was checked against bfloat16 and transcribes identically.
    """
    if capability is None:
        return "float32"  # dots.tts refuses half precision on CPU
    requested = (requested or "auto").strip().lower()
    if requested in PRECISIONS:
        return requested
    return "bfloat16" if capability >= (8, 0) else "float16"


def load_config(env: dict[str, str] | None = None) -> AppConfig:
    env = dict(os.environ if env is None else env)
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    wants_optimize = env.get("DOTS_OPTIMIZE", "0") == "1"
    # dots.tts's torch.compile path is tuned for Ampere (compute capability
    # 8.0) and newer; don't attempt it on a T4.
    optimize = wants_optimize and capability is not None and capability >= (8, 0)
    if wants_optimize and not optimize:
        print("ℹ️  torch.compile needs an Ampere-or-newer GPU (A100, L4, H100) — skipping it.")
    return AppConfig(
        output_root=Path(env.get("DOTS_OUTPUT_ROOT") or APP_DIR / "outputs"),
        default_project=env.get("DOTS_PROJECT") or narration.DEFAULT_PROJECT,
        precision=pick_precision(env.get("DOTS_PRECISION"), capability),
        optimize=optimize,
        share=env.get("DOTS_SHARE") == "1" or "COLAB_RELEASE_TAG" in env,
        password=env.get("DOTS_PASSWORD", "").strip(),
    )


def print_device_notice(config: AppConfig) -> None:
    if not torch.cuda.is_available():
        print(
            "\n" + "=" * 72 + "\n"
            "⚠️  NO GPU DETECTED — dots.tts is a 2B model; on CPU a single sentence\n"
            "    can take many minutes.\n\n"
            "    In Colab:  Runtime → Change runtime type → T4 GPU → Save,\n"
            "               then run the cells again.\n" + "=" * 72 + "\n"
        )
        return
    name = torch.cuda.get_device_name()
    print(f"🖥️  GPU: {name} · precision: {config.precision} · torch.compile: {config.optimize}")
    if torch.cuda.get_device_capability() < (8, 0) and config.precision == "bfloat16":
        print("ℹ️  This GPU has no native bfloat16, so PyTorch emulates it (slower than 'auto').")


def load_runtime(config: AppConfig):
    from dots_tts.runtime import DotsTtsRuntime  # heavy; keep out of module import

    print(f"⏳ Loading {MODEL_ID} ...")
    runtime = DotsTtsRuntime.from_pretrained(
        MODEL_ID, precision=config.precision, optimize=config.optimize
    )
    print("✅ Model loaded")
    return runtime


# ---------------------------------------------------------------------------
# Whisper (reference transcripts and subtitles) via subtitle.py
# ---------------------------------------------------------------------------


def transcribe(audio_path: str, language: str) -> str:
    """Exact words spoken in `audio_path`, or "" if Whisper fails."""
    try:
        from subtitle import subtitle_maker

        result = subtitle_maker(audio_path, tts_utils.whisper_code(language))
        return (result[7] or "").strip()
    except Exception:
        logger.exception("Transcription failed for %s", audio_path)
        return ""


def make_subtitles(wav_path: Path, language: str) -> dict[str, str | None]:
    """SRT files for a take, keyed by the suffix they are saved with."""
    try:
        from subtitle import subtitle_maker

        result = subtitle_maker(str(wav_path), tts_utils.whisper_code(language))
    except Exception:
        logger.exception("Subtitle generation failed for %s", wav_path)
        return {}
    if result[0] is None:
        logger.warning("Subtitle generation failed: %s", result[-1])
        return {}
    return {"": result[1], "_words": result[2], "_shorts": result[3]}


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def make_settings(language, steps, cfg, speaker, normalize, auto_split, chunk_chars, pause):
    return engine.Settings(
        language=language or tts_utils.LANGUAGE_NOT_SET,
        num_steps=int(steps),
        guidance_scale=float(cfg),
        speaker_scale=float(speaker),
        normalize_text=bool(normalize),
        auto_split=bool(auto_split),
        chunk_chars=int(chunk_chars),
        chunk_pause=float(pause),
    )


def build_voice(mode: str, ref_audio: str | None, ref_text: str, language: str):
    """Returns (voice, transcript to show in the UI, notes for the status box)."""
    if mode == RANDOM_VOICE:
        return engine.Voice(), ref_text, []
    if not ref_audio:
        raise SynthesisError("Upload a reference clip, or choose 'Random voice'.")

    path = engine.shorten_reference(ref_audio, WORK_DIR)
    notes = []
    if path != ref_audio:
        notes.append(f"✂️ Reference cut to {engine.audio_seconds(path):.1f} s (15 s max).")
    if mode == CLONE_TIMBRE:
        return engine.Voice(audio_path=path), ref_text, notes

    transcript = (ref_text or "").strip()
    if path != ref_audio or not transcript:
        transcript = transcribe(path, language)
        if not transcript:
            raise SynthesisError(
                "Couldn't transcribe the reference clip. Type exactly what is said "
                "in it into 'Reference transcript', or use 'Clone: reference audio only'."
            )
        notes.append("📝 Reference transcript filled in by Whisper — check it matches.")
    return engine.Voice(audio_path=path, transcript=transcript), transcript, notes


def describe_folder(folder: Path) -> str:
    parts = folder.parts
    if "MyDrive" in parts:
        return "Google Drive › " + " › ".join(parts[parts.index("MyDrive") + 1:])
    return str(folder)


def skipped_lines_note(lines: list[str], shown: int = 6) -> str:
    """One status line listing script lines that won't be read aloud."""
    quoted = [f"“{line[:40]}{'…' if len(line) > 40 else ''}”" for line in lines[:shown]]
    more = f" and {len(lines) - shown} more" if len(lines) > shown else ""
    return f"🙈 Not read aloud (headings / text before the first id): {', '.join(quoted)}{more}"


def take_summary(take: engine.Take) -> str:
    chunks = f" · {take.chunk_count} chunks" if take.chunk_count > 1 else ""
    return (
        f"⏱️ {take.duration:.1f} s of audio in {take.elapsed:.1f} s"
        f" · seed {take.seed}{chunks}"
    )


def for_display(paths: list[Path]) -> list[str]:
    """Copy saved files somewhere Gradio may serve them from.

    The saved originals stay in the output folder (Google Drive on Colab),
    which is deliberately not in Gradio's allowed paths. A file that can't be
    copied is left out of the UI; it is still saved.
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=WORK_DIR))
    copies = []
    for path in paths:
        target = folder / path.name
        try:
            shutil.copyfile(path, target)
        except OSError:
            logger.exception("Could not copy %s for display", path)
            continue
        copies.append(str(target))
    return copies


def save_with_subtitles(folder: Path, name: str, take: engine.Take, language: str, want_subs: bool):
    """Save the take, then (optionally) its subtitles. Returns (path, replaced, files).

    A subtitle failure never turns a saved take into a reported failure; the
    caller sees it as `files` holding only the .wav.
    """
    path, replaced = narration.save_take(folder, name, take.audio, take.sample_rate)
    files = [path]
    if want_subs:
        try:
            files += narration.copy_subtitles(folder, name, make_subtitles(path, language))
        except OSError:
            logger.exception("Saving subtitles for %s failed", path)
    return path, replaced, files


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


class Handlers:
    """Every Gradio callback, bound to one loaded runtime."""

    def __init__(self, runtime: Any, config: AppConfig):
        self.runtime = runtime
        self.config = config

    def folder(self, project: str, date_text: str) -> Path:
        return narration.project_folder(self.config.output_root, project, date_text)

    def folder_label(self, project: str, date_text: str) -> str:
        return f"📁 Saving to **{describe_folder(self.folder(project, date_text))}**"

    def on_mode_change(self, mode: str):
        return gr.Audio(visible=mode != RANDOM_VOICE), gr.Textbox(visible=mode == CLONE_FULL)

    def on_reference_upload(self, path: str | None, language: str):
        if not path:
            return gr.skip(), gr.skip(), ""
        try:
            short = engine.shorten_reference(path, WORK_DIR)
        except SynthesisError as exc:
            return gr.skip(), "", f"⚠️ {exc}"
        except Exception:
            logger.exception("Could not read reference %s", path)
            return gr.skip(), "", "⚠️ Couldn't read that audio file. Try a WAV or MP3."

        notes = []
        if short != path:
            notes.append(f"✂️ Cut to {engine.audio_seconds(short):.1f} s — 5–15 s clones best.")
        transcript = transcribe(short, language)
        notes.append(
            "📝 Transcript filled in by Whisper — fix any mistakes; it must match the audio word for word."
            if transcript
            else "⚠️ Whisper couldn't transcribe this clip — type the transcript yourself."
        )
        return short, transcript, "\n".join(notes)

    def on_generate(self, text, name, mode, ref_audio, ref_text, language, project, date_text,
                    want_subs, steps, cfg, speaker, normalize, seed, auto_split, chunk_chars,
                    pause, progress=gr.Progress()):
        typed_name = narration.clean_name(name)
        save_name = typed_name or f"take_{time.strftime('%Y%m%d_%H%M%S')}"
        folder = self.folder(project, date_text)
        settings = make_settings(language, steps, cfg, speaker, normalize, auto_split, chunk_chars, pause)
        try:
            voice, transcript, notes = build_voice(mode, ref_audio, ref_text, language)
            take = engine.synthesize(
                self.runtime, text, voice, settings,
                seed=tts_utils.resolve_seed(seed), work_dir=WORK_DIR,
                on_chunk=lambda i, n: progress(i / n, desc=f"Chunk {i + 1} of {n}"),
            )
            path, replaced, files = save_with_subtitles(folder, save_name, take, language, want_subs)
        except SynthesisError as exc:
            return None, f"❌ {exc}", gr.skip(), gr.skip(), gr.skip(), gr.skip()
        except Exception as exc:
            logger.exception("Generation failed")
            return None, f"❌ Unexpected error — {type(exc).__name__}: {exc}", gr.skip(), gr.skip(), gr.skip(), gr.skip()

        lines = [
            f"✅ Saved {path.name}" + (" (replaced the previous take)" if replaced else ""),
            f"📁 {describe_folder(folder)}",
            take_summary(take),
            *notes,
        ]
        if want_subs and len(files) == 1:
            lines.append("⚠️ Subtitles could not be made for this take.")
        next_name = narration.next_name(typed_name) if typed_name else ""
        shown = for_display(files)
        return (shown[0] if shown else None), "\n".join(lines), shown, next_name, take, transcript

    def on_batch(self, script, skip_existing, mode, ref_audio, ref_text, language, project,
                 date_text, want_subs, steps, cfg, speaker, normalize, seed, auto_split,
                 chunk_chars, pause, progress=gr.Progress()):
        try:
            parsed = narration.parse_script(script)
            voice, transcript, notes = build_voice(mode, ref_audio, ref_text, language)
        except (narration.ScriptError, SynthesisError) as exc:
            yield None, f"❌ {exc}", gr.skip(), gr.skip(), gr.skip()
            return
        items = parsed.narrations
        if parsed.ignored:
            notes = [*notes, skipped_lines_note(parsed.ignored)]

        settings = make_settings(language, steps, cfg, speaker, normalize, auto_split, chunk_chars, pause)
        used_seed = tts_utils.resolve_seed(seed)
        folder = self.folder(project, date_text)
        log = [f"📁 {describe_folder(folder)} · {len(items)} narrations · seed {used_seed}", *notes]
        files: list[str] = []
        last_path, last_take = None, None
        counts = {"saved": 0, "failed": 0, "skipped": 0}

        for index, item in enumerate(items):
            if skip_existing and (folder / f"{item.name}.wav").exists():
                counts["skipped"] += 1
                log.append(f"⏭️ {item.name}.wav already exists — skipped")
                continue
            progress(index / len(items), desc=f"Narration {item.name} ({index + 1} of {len(items)})")
            yield last_path, "\n".join(log + [f"⏳ Generating {item.name} …"]), files or None, last_take, transcript

            line, shown, take = self._batch_item(item, voice, settings, used_seed, folder, language, want_subs)
            log.append(line)
            if take is None:
                counts["failed"] += 1
                continue
            counts["saved"] += 1
            files += shown
            last_path, last_take = (shown[0] if shown else last_path), take
            if voice.audio_path is None:
                # Random voice: keep the first narration's voice for the rest.
                voice = engine.voice_from_take(take, WORK_DIR)

        log.append(
            f"🏁 Done — {counts['saved']} saved, {counts['failed']} failed, {counts['skipped']} skipped."
        )
        yield last_path, "\n".join(log), files or None, last_take, transcript

    def _batch_item(self, item, voice, settings, seed, folder, language, want_subs):
        """Generate and save one narration. Returns (log line, files for the UI, take or None)."""
        try:
            take = engine.synthesize(self.runtime, item.text, voice, settings, seed=seed, work_dir=WORK_DIR)
            path, _, saved = save_with_subtitles(folder, item.name, take, language, want_subs)
        except SynthesisError as exc:
            return f"❌ {item.name}: {exc}", [], None
        except Exception as exc:
            logger.exception("Narration %s failed", item.name)
            return f"❌ {item.name}: unexpected error — {type(exc).__name__}: {exc}", [], None
        line = f"✅ {path.name} · {take.duration:.1f} s"
        if want_subs and len(saved) == 1:
            line += " · ⚠️ subtitles could not be made"
        return line, for_display(saved), take

    def on_use_take(self, take: engine.Take | None):
        if take is None:
            return gr.skip(), gr.skip(), gr.skip(), "Generate a take first."
        voice = engine.voice_from_take(take, WORK_DIR)
        mode = CLONE_FULL if voice.transcript else CLONE_TIMBRE
        return (
            voice.audio_path,
            voice.transcript,
            mode,
            "📌 That take's voice is now the reference — every new narration will use it.",
        )


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

HEADER = """
<div style="text-align:center; margin:12px auto 4px; max-width:820px;">
  <h1 style="font-size:2.2em; margin-bottom:4px;">🎙️ dots.tts SOAR</h1>
  <p>2B continuous autoregressive TTS · 48 kHz · zero-shot voice cloning in 100+ languages</p>
</div>
"""

TIPS = """
* **Reference clip:** 5–15 s of clean speech from one speaker. Longer clips are cut automatically.
* **Transcript:** must match the reference word for word. Whisper fills it in; fix its mistakes.
* **Seed:** each seed gives a different rhythm and intonation. The seed used is shown in the status box; type it back in to repeat a take.
* **Quality:** raise *Inference steps* to 16–32 for slightly better audio at the cost of speed.
* **Random voice:** when you like a voice, click **Use this take as the reference voice** to keep it for every narration.
* **Batch script:** start each narration with its id — `1.1: text`, `1.1 — text`, or the id alone on its line with the text below. Lines without an id continue the narration above. Headings like `CHAPTER 1` are skipped, anything in `[square brackets]` is treated as a note and not read aloud, and markers like ★ are ignored.
* **Chinese polyphones:** write tone-marked pinyin to force a reading, e.g. `我生平不hào此道`.
"""

BATCH_PLACEHOLDER = """CHAPTER 1

1.1 — [COLD OPEN]
The storm had been building all afternoon.

1.2: By nightfall, the harbour was empty.
2.1: Morning brought an eerie calm."""

THEME = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
CSS = """
.gradio-container {max-width: 100% !important;}
#status textarea {font-family: ui-monospace, monospace; font-size: 0.9em;}
"""
BROWSER_DATE_JS = "() => new Date().toLocaleDateString('en-CA')"  # YYYY-MM-DD, local time


def _settings_panel():
    with gr.Accordion("Generation settings", open=False):
        steps = gr.Slider(4, 32, value=10, step=1, label="Inference steps",
                          info="10 is the default; 16–32 is slightly better and slower.")
        cfg = gr.Slider(0.5, 3.0, value=1.2, step=0.1, label="Guidance scale (CFG)")
        speaker = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Speaker scale",
                            info="How strongly the reference speaker's timbre is applied.")
        normalize = gr.Checkbox(value=False, label="Normalize numbers and dates (Chinese and English only)")
        seed = gr.Number(value=-1, precision=0, label="Seed",
                         info="-1 picks a new one each run. The seed used is shown in the status box.")
        with gr.Accordion("Long text", open=False):
            auto_split = gr.Checkbox(value=True, label="Split long text on sentences")
            chunk_chars = gr.Slider(80, 400, value=250, step=10, label="Max characters per chunk",
                                    info="Chinese, Japanese, and Korean characters count as three.")
            pause = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="Pause between chunks (seconds)")
    return [steps, cfg, speaker, normalize, seed, auto_split, chunk_chars, pause]


def _voice_panel() -> SimpleNamespace:
    mode = gr.Radio([CLONE_FULL, CLONE_TIMBRE, RANDOM_VOICE], value=CLONE_FULL, label="Voice")
    ref_audio = gr.Audio(label="Reference audio (5–15 s of clean speech)", type="filepath",
                         sources=["upload", "microphone"])
    ref_text = gr.Textbox(label="Reference transcript", lines=2,
                          placeholder="Filled in by Whisper after you upload — must match the audio exactly.")
    language = gr.Dropdown(tts_utils.language_choices(), value=tts_utils.LANGUAGE_NOT_SET,
                           label="Language", info="Optional. Useful for mixed-language text and dialects.")
    return SimpleNamespace(mode=mode, ref_audio=ref_audio, ref_text=ref_text, language=language)


def _text_tabs() -> SimpleNamespace:
    with gr.Tabs():
        with gr.Tab("Single narration"):
            name = gr.Textbox(label="Narration name", value="1.1",
                              info="Saved as <name>.wav, then moves on to the next number (1.1 → 1.2).")
            text = gr.Textbox(label="Text", lines=6, placeholder="Type what should be said…")
            generate = gr.Button("🎙️ Generate & save", variant="primary")
        with gr.Tab("Batch script"):
            script = gr.Textbox(label="Script — one narration per line, starting with its id",
                                lines=12, placeholder=BATCH_PLACEHOLDER)
            skip_existing = gr.Checkbox(value=False,
                                        label="Skip narrations already saved (resume an interrupted batch)")
            with gr.Row():
                batch = gr.Button("🎬 Generate all", variant="primary")
                stop = gr.Button("⏹ Stop", variant="stop")
    return SimpleNamespace(name=name, text=text, generate=generate, script=script,
                           skip_existing=skip_existing, batch=batch, stop=stop)


def _output_panel() -> SimpleNamespace:
    audio = gr.Audio(label="Latest take", type="filepath", interactive=False)
    status = gr.Textbox(label="Status", lines=8, elem_id="status")
    use_take = gr.Button("📌 Use this take as the reference voice")
    want_subs = gr.Checkbox(value=False, label="Also save subtitles (SRT) next to each take")
    files = gr.File(label="Saved files", file_count="multiple", interactive=False)
    return SimpleNamespace(audio=audio, status=status, use_take=use_take, want_subs=want_subs, files=files)


def _wire_events(demo: gr.Blocks, handlers: Handlers, ui: SimpleNamespace) -> None:
    voice, texts, out = ui.voice, ui.texts, ui.out
    shared = [voice.mode, voice.ref_audio, voice.ref_text, voice.language,
              ui.project, ui.date, out.want_subs, *ui.settings]
    # Gradio's concurrency limit is per event; one shared id makes every GPU
    # job (dots.tts or Whisper) wait for the previous one, whichever button started it.
    gpu = {"concurrency_id": "gpu", "concurrency_limit": 1}
    for trigger in (ui.project.change, ui.date.change):
        trigger(handlers.folder_label, [ui.project, ui.date], ui.folder_info)
    demo.load(None, None, ui.date, js=BROWSER_DATE_JS)
    voice.mode.change(handlers.on_mode_change, voice.mode, [voice.ref_audio, voice.ref_text])
    for trigger in (voice.ref_audio.upload, voice.ref_audio.stop_recording):
        trigger(handlers.on_reference_upload, [voice.ref_audio, voice.language],
                [voice.ref_audio, voice.ref_text, out.status], **gpu)
    texts.generate.click(handlers.on_generate, [texts.text, texts.name, *shared],
                         [out.audio, out.status, out.files, texts.name, ui.take, voice.ref_text], **gpu)
    batch_event = texts.batch.click(handlers.on_batch, [texts.script, texts.skip_existing, *shared],
                                    [out.audio, out.status, out.files, ui.take, voice.ref_text], **gpu)
    texts.stop.click(None, None, None, cancels=[batch_event])
    out.use_take.click(handlers.on_use_take, ui.take,
                       [voice.ref_audio, voice.ref_text, voice.mode, out.status])


def build_demo(runtime: Any, config: AppConfig) -> gr.Blocks:
    handlers = Handlers(runtime, config)
    today = time.strftime("%Y-%m-%d")

    with gr.Blocks(title="dots.tts SOAR") as demo:
        gr.HTML(HEADER)
        with gr.Accordion("💡 Tips", open=False):
            gr.Markdown(TIPS)
        with gr.Row():
            project = gr.Textbox(label="Project name", value=config.default_project, scale=3)
            date_box = gr.Textbox(label="Date", value=today, scale=1,
                                  info="From your browser's clock — edit it if you like.")
        folder_info = gr.Markdown(handlers.folder_label(config.default_project, today))
        with gr.Row():
            with gr.Column(scale=1):
                voice = _voice_panel()
                texts = _text_tabs()
                settings = _settings_panel()
            with gr.Column(scale=1):
                out = _output_panel()
        ui = SimpleNamespace(project=project, date=date_box, folder_info=folder_info, voice=voice,
                             texts=texts, settings=settings, out=out, take=gr.State(None))
        _wire_events(demo, handlers, ui)
    return demo


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    config = load_config()
    config.output_root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK_DIR, ignore_errors=True)  # scratch files from an earlier run
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print_device_notice(config)
    runtime = load_runtime(config)
    print(f"📁 Projects are saved under {config.output_root}")
    demo = build_demo(runtime, config)
    if config.password:
        print(f"🔒 The app asks for username '{AUTH_USERNAME}' and the password you set.")
    # No allowed_paths on purpose: see WORK_DIR and for_display().
    demo.queue(max_size=16).launch(
        share=config.share,
        debug=config.share,
        theme=THEME,
        css=CSS,
        auth=(AUTH_USERNAME, config.password) if config.password else None,
    )


if __name__ == "__main__":
    main()
