from pathlib import Path

import gradio as gr
import pytest
import soundfile as sf

import app
import engine
from conftest import FakeRuntime, tone

NO_PROGRESS = lambda *args, **kwargs: None  # noqa: E731
# steps, cfg, speaker, normalize, seed, auto_split, chunk_chars, pause
DEFAULT_SETTINGS = (10, 1.2, 1.5, False, 5, True, 250, 0.3)


@pytest.fixture
def handlers(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(app, "transcribe", lambda path, language: "words from whisper")
    config = app.AppConfig(tmp_path / "out", "Story", "float16", False, False)
    return app.Handlers(FakeRuntime(), config)


def reference(tmp_path, seconds=5.0) -> str:
    path = tmp_path / f"ref_{seconds}.wav"
    sf.write(str(path), tone(seconds, 16000), 16000)
    return str(path)


def is_skip(value) -> bool:
    return value == gr.skip()


@pytest.mark.parametrize(
    ("requested", "capability", "expected"),
    [
        ("auto", (7, 5), "float16"),
        ("auto", (8, 9), "bfloat16"),
        (None, (9, 0), "bfloat16"),
        ("BFloat16", (7, 5), "bfloat16"),
        ("float32", (8, 0), "float32"),
        ("nonsense", (7, 5), "float16"),
        ("float16", None, "float32"),
    ],
)
def test_pick_precision(requested, capability, expected):
    assert app.pick_precision(requested, capability) == expected


def test_load_config_reads_environment(tmp_path):
    config = app.load_config({"DOTS_OUTPUT_ROOT": str(tmp_path), "DOTS_PROJECT": "Tales",
                              "DOTS_SHARE": "1", "DOTS_OPTIMIZE": "1"})
    assert config.output_root == tmp_path and config.default_project == "Tales" and config.share
    assert not config.optimize or app.torch.cuda.get_device_capability() >= (8, 0)


def test_load_config_defaults():
    config = app.load_config({})
    assert config.output_root == app.APP_DIR / "outputs"
    assert config.default_project == "My Project" and not config.share
    assert config.password == ""


def test_load_config_password():
    assert app.load_config({"DOTS_PASSWORD": " secret "}).password == "secret"


def test_load_config_shares_on_colab():
    assert app.load_config({"COLAB_RELEASE_TAG": "release-colab"}).share


def test_describe_folder_on_drive_and_locally():
    drive = Path("/content/drive/MyDrive/Dots TTS/Story_2026-09-25")
    assert app.describe_folder(drive) == "Google Drive › Dots TTS › Story_2026-09-25"
    assert app.describe_folder(Path("out/Story")) == str(Path("out/Story"))


def test_build_voice_random_needs_no_reference():
    voice, transcript, notes = app.build_voice(app.RANDOM_VOICE, None, "kept", "none")
    assert voice == engine.Voice() and transcript == "kept" and notes == []


def test_build_voice_clone_requires_audio():
    with pytest.raises(engine.SynthesisError, match="Upload a reference"):
        app.build_voice(app.CLONE_FULL, None, "", "none")


def test_build_voice_uses_typed_transcript(handlers, tmp_path):
    path = reference(tmp_path)
    voice, transcript, notes = app.build_voice(app.CLONE_FULL, path, " typed words ", "EN")
    assert voice == engine.Voice(path, "typed words") and notes == []


def test_build_voice_transcribes_when_transcript_missing(handlers, tmp_path):
    voice, transcript, notes = app.build_voice(app.CLONE_FULL, reference(tmp_path), "", "EN")
    assert voice.transcript == transcript == "words from whisper" and notes


def test_build_voice_retranscribes_a_cut_reference(handlers, tmp_path):
    voice, transcript, notes = app.build_voice(app.CLONE_FULL, reference(tmp_path, 30), "for the long clip", "EN")
    assert transcript == "words from whisper"
    assert any("cut" in note for note in notes)


def test_build_voice_fails_when_whisper_fails(handlers, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "transcribe", lambda path, language: "")
    with pytest.raises(engine.SynthesisError, match="Couldn't transcribe"):
        app.build_voice(app.CLONE_FULL, reference(tmp_path), "", "EN")


def test_build_voice_timbre_only_skips_whisper(handlers, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "transcribe", lambda *args: pytest.fail("should not transcribe"))
    voice, _, _ = app.build_voice(app.CLONE_TIMBRE, reference(tmp_path), "", "EN")
    assert voice.transcript == ""


def test_on_generate_saves_named_take_and_advances_name(handlers, tmp_path):
    audio, status, files, next_name, take, transcript = handlers.on_generate(
        "Hello there.", "1.1", app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25",
        False, *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    )
    saved = tmp_path / "out" / "Story_2026-09-25" / "1.1.wav"
    assert saved.exists() and Path(audio).name == "1.1.wav"
    assert files == [audio] and next_name == "1.2"
    assert "✅ Saved 1.1.wav" in status and "seed 5" in status
    assert isinstance(take, engine.Take)


def test_ui_gets_copies_outside_the_output_folder(handlers, tmp_path):
    """The Drive folder is never handed to Gradio, so the share link can't reach it."""
    audio, _, files, *_ = handlers.on_generate(
        "Hello.", "1.1", app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25", False,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    )
    output_root = (tmp_path / "out").resolve()
    for path in [audio, *files]:
        assert Path(path).exists()
        assert output_root not in Path(path).resolve().parents


def test_on_generate_reports_replacing_a_take(handlers):
    args = ("Hello.", "2.1", app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25", False, *DEFAULT_SETTINGS)
    handlers.on_generate(*args, progress=NO_PROGRESS)
    _, status, *_ = handlers.on_generate(*args, progress=NO_PROGRESS)
    assert "replaced the previous take" in status


def test_on_generate_without_name_uses_a_timestamp(handlers):
    audio, _, _, next_name, _, _ = handlers.on_generate(
        "Hello.", "  ", app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25", False,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    )
    assert Path(audio).name.startswith("take_") and next_name == ""


def test_on_generate_error_keeps_other_outputs(handlers):
    audio, status, files, next_name, take, transcript = handlers.on_generate(
        "Hello.", "1.1", app.CLONE_FULL, None, "", "none", "Story", "2026-09-25", False,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    )
    assert audio is None and status.startswith("❌")
    assert all(is_skip(value) for value in (files, next_name, take, transcript))


def test_on_generate_unexpected_error_is_reported(handlers, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "synthesize", explode)
    _, status, *_ = handlers.on_generate(
        "Hello.", "1.1", app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25", False,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    )
    assert "RuntimeError: boom" in status


def test_on_generate_saves_subtitles_when_asked(handlers, tmp_path, monkeypatch):
    srt = tmp_path / "made.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    monkeypatch.setattr(app, "make_subtitles", lambda path, language: {"": str(srt)})
    _, _, files, *_ = handlers.on_generate(
        "Hello.", "1.1", app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25", True,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    )
    assert [Path(f).name for f in files] == ["1.1.wav", "1.1.srt"]


def _broken_subtitle_copy(folder, name, srt_paths):
    raise OSError("Drive went away")


def test_subtitle_copy_failure_still_reports_a_saved_take(handlers, monkeypatch):
    monkeypatch.setattr(app, "make_subtitles", lambda path, language: {})
    monkeypatch.setattr(app.narration, "copy_subtitles", _broken_subtitle_copy)
    audio, status, files, *_ = handlers.on_generate(
        "Hello.", "1.1", app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25", True,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    )
    assert "✅ Saved 1.1.wav" in status and "Subtitles could not be made" in status
    assert [Path(f).name for f in files] == ["1.1.wav"]


def test_batch_subtitle_copy_failure_still_counts_as_saved(handlers, monkeypatch):
    monkeypatch.setattr(app, "make_subtitles", lambda path, language: {})
    monkeypatch.setattr(app.narration, "copy_subtitles", _broken_subtitle_copy)
    updates = list(handlers.on_batch(
        "1.1: First.", False, app.RANDOM_VOICE, None, "", "none", "Story", "2026-09-25", True,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    ))
    status = updates[-1][1]
    assert "1 saved, 0 failed" in status and "subtitles could not be made" in status


def test_for_display_skips_files_it_cannot_copy(handlers, tmp_path):
    real = tmp_path / "real.wav"
    real.write_bytes(b"x")
    assert [Path(p).name for p in app.for_display([tmp_path / "missing.wav", real])] == ["real.wav"]


def run_batch(handlers, script, skip_existing=False, mode=app.RANDOM_VOICE, ref_audio=None):
    updates = list(handlers.on_batch(
        script, skip_existing, mode, ref_audio, "", "none", "Story", "2026-09-25", False,
        *DEFAULT_SETTINGS, progress=NO_PROGRESS,
    ))
    return updates[-1]


def test_on_batch_saves_each_narration_by_id_in_one_voice(handlers, tmp_path):
    audio, status, files, take, _ = run_batch(handlers, "1.1: First.\n1.2: Second.\n2.1: Third.")
    folder = tmp_path / "out" / "Story_2026-09-25"
    assert [Path(f).name for f in files] == ["1.1.wav", "1.2.wav", "2.1.wav"]
    assert all((folder / name).exists() for name in ("1.1.wav", "1.2.wav", "2.1.wav"))
    assert Path(audio).name == "2.1.wav" and folder.resolve() not in Path(audio).resolve().parents
    assert "3 saved, 0 failed, 0 skipped" in status
    calls = handlers.runtime.calls
    assert calls[0]["prompt_audio_path"] is None
    assert calls[1]["prompt_audio_path"] == calls[2]["prompt_audio_path"] is not None


def test_on_batch_reads_a_production_script_and_reports_skipped_lines(handlers, tmp_path):
    script = (
        "CHAPTER 1\n\n1.1 — [OPENING] [CHAR: 694]\nFirst line.\n\n"
        "CHAPTER 2\n\n★ 2.4 — [ASK 1 of 3] [CHAR: 557]\nSecond line.\n"
    )
    _, status, files, _, _ = run_batch(handlers, script)
    assert [Path(f).name for f in files] == ["1.1.wav", "2.4.wav"]
    assert "“CHAPTER 1”, “CHAPTER 2”" in status and "2 saved, 0 failed" in status
    assert [call["text"] for call in handlers.runtime.calls] == ["First line.", "Second line."]


def test_skipped_lines_note_truncates_long_lists():
    note = app.skipped_lines_note([f"CHAPTER {i}" for i in range(1, 10)] + ["x" * 60])
    assert "“CHAPTER 6”" in note and "“CHAPTER 7”" not in note and "and 4 more" in note
    assert "…" in app.skipped_lines_note(["y" * 60])


def test_on_batch_skips_existing_files(handlers, tmp_path):
    run_batch(handlers, "1.1: First.")
    _, status, files, _, _ = run_batch(handlers, "1.1: First.\n1.2: Second.", skip_existing=True)
    assert "1 saved, 0 failed, 1 skipped" in status
    assert [Path(f).name for f in files] == ["1.2.wav"]


def test_on_batch_continues_after_a_failure(handlers):
    handlers.runtime.audio_fn = lambda text: tone(0.0) if "Bad" in text else tone(0.5)
    _, status, files, _, _ = run_batch(handlers, "1.1: Bad one.\n1.2: Good one.")
    assert "❌ 1.1" in status and "1 saved, 1 failed" in status
    assert [Path(f).name for f in files] == ["1.2.wav"]


def test_on_batch_reports_script_errors(handlers):
    audio, status, *rest = run_batch(handlers, "1.1: A\n1.1: B")
    assert audio is None and "more than once" in status and all(is_skip(v) for v in rest)


def test_on_batch_clone_voice_used_for_every_narration(handlers, tmp_path):
    path = reference(tmp_path)
    run_batch(handlers, "1.1: First.\n1.2: Second.", mode=app.CLONE_FULL, ref_audio=path)
    assert {call["prompt_audio_path"] for call in handlers.runtime.calls} == {path}
    assert {call["prompt_text"] for call in handlers.runtime.calls} == {"words from whisper"}


def test_on_use_take_turns_a_take_into_the_reference(handlers):
    audio = tone(4)
    take = engine.Take(48000, audio, 1, 1, 0.0, audio, "exact words")
    path, transcript, mode, status = handlers.on_use_take(take)
    assert Path(path).exists() and transcript == "exact words" and mode == app.CLONE_FULL
    assert "📌" in status


def test_on_use_take_without_a_take(handlers):
    *outputs, status = handlers.on_use_take(None)
    assert all(is_skip(v) for v in outputs) and "Generate a take first" in status


def test_on_reference_upload_cuts_and_transcribes(handlers, tmp_path):
    short, transcript, status = handlers.on_reference_upload(reference(tmp_path, 30), "EN")
    assert engine.audio_seconds(short) <= engine.MAX_REFERENCE_SECONDS
    assert transcript == "words from whisper" and "Cut to" in status


def test_on_reference_upload_rejects_tiny_clip(handlers, tmp_path):
    _, transcript, status = handlers.on_reference_upload(reference(tmp_path, 0.2), "EN")
    assert transcript == "" and status.startswith("⚠️")


def test_on_reference_upload_handles_unreadable_file(handlers, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    _, _, status = handlers.on_reference_upload(str(bad), "EN")
    assert "Couldn't read" in status


def test_on_reference_upload_cleared(handlers):
    audio, transcript, status = handlers.on_reference_upload(None, "EN")
    assert is_skip(audio) and is_skip(transcript) and status == ""


def test_on_mode_change_toggles_reference_inputs(handlers):
    audio, text = handlers.on_mode_change(app.RANDOM_VOICE)
    assert audio.visible is False and text.visible is False
    audio, text = handlers.on_mode_change(app.CLONE_TIMBRE)
    assert audio.visible is True and text.visible is False


def test_build_demo_constructs_the_interface(handlers):
    demo = app.build_demo(handlers.runtime, handlers.config)
    assert isinstance(demo, gr.Blocks)
    labels = {getattr(block, "label", None) for block in demo.blocks.values()}
    assert {"Narration name", "Project name", "Reference transcript", "Saved files"} <= labels


def test_every_gpu_job_shares_one_queue(handlers):
    demo = app.build_demo(handlers.runtime, handlers.config)
    gpu_jobs = {"on_generate", "on_batch", "on_reference_upload"}
    fns = [fn for fn in demo.fns.values() if getattr(fn, "name", "") in gpu_jobs]
    assert {fn.name for fn in fns} == gpu_jobs
    assert {fn.concurrency_id for fn in fns} == {"gpu"}
    assert {fn.concurrency_limit for fn in fns} == {1}
