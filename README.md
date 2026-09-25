# 🎙️ Run dots.tts SOAR On Google Colab

Clone a voice and generate narration with **dots.tts SOAR** on Google Colab. Every take is saved to your **Google Drive** under the narration name you choose (`1.1.wav`, `1.2.wav`, …), in a folder named after your project and the date.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hirannalaka19/Dots-Google-Colab/blob/main/Dots_TTS_Colab.ipynb)

---

## 🧠 About

A Google Colab app for [dots.tts](https://github.com/studio-dots-ai/dots.tts) by the dots.tts team. It runs the [`dots-studio/dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar) checkpoint: a 2B-parameter continuous autoregressive TTS model producing 48 kHz audio, with the highest speaker similarity of the dots.tts checkpoints.

It uses the official `dots.tts` pip package, so there's no fork to maintain.

---

## 🔹 What it can do

* **Voice cloning** from 5–15 s of reference audio. Whisper writes the transcript for you.
* **Timbre-only cloning** when you don't have a transcript.
* **Random voices**: keep one you like with **📌 Use this take as the reference voice**.
* **Named narrations**: type `1.1`, get `1.1.wav`. The name then moves to `1.2` by itself.
* **Batch scripts**: paste a whole script (`1.1: …`, `1.2: …`, `2.1: …`) and every line is saved under its id, in one consistent voice.
* **Google Drive saving**: `MyDrive/Dots TTS/<Project>_<YYYY-MM-DD>/1.1.wav`, dated from your browser's clock.
* **Long text**: split on sentence boundaries and joined back with a short pause.
* **Subtitles (SRT)**: sentence level, word level and Shorts style, saved next to each take.
* **Reproducible takes**: the seed used is shown, so you can repeat a take exactly.
* **100+ languages and Chinese dialects** via the language tag.

---

## 🚀 How to use

1. Click the **Open In Colab** badge above.
2. Choose a GPU: `Runtime → Change runtime type → T4 GPU`. The free T4 is enough; peak GPU memory is about 6 GB.
3. Run **1️⃣ Install dots.tts** (about 2 minutes).
4. In **2️⃣ Run the app**, set your *project name* (and the Drive folder, if you want a different one), then run the cell. Allow Google Drive access when asked.
5. Open the public `*.gradio.live` link it prints.

Optional extras:
* **Lock the link:** anyone who has your `gradio.live` link can use the app. Set `app_password` in the Run cell to require a login (username `dots`). Your Drive folder itself is never served over the link; the app only shows copies of the takes it just made.
* **Faster download:** add a Hugging Face token as a Colab **Secret** named `HF_TOKEN` (🔑 icon in the sidebar, then enable *Notebook access*). The model is public, so the token only speeds up the 5 GB download.

---

## 🎬 Batch script format

Start each narration with its id. The text can be on the same line or on the lines below:

```
CHAPTER 1

1.1 — [COLD OPEN] [CHAR: 694]
The storm had been building all afternoon.

★ 1.2 — [THE WARNING]
By nightfall, the harbour was empty.

2.1: Morning brought an eerie calm.
```

* Ids like `1.1:`, `1.1 —`, `[1.1]`, `3)` and `12 |` all work, with or without a marker like `★` in front.
* Lines without an id continue the narration above them. In a script that uses ids like `1.1`, a line starting with a plain number (a list item like `1.` or `3:`, or a time like `6:19`) stays part of the narration.
* Anything in `[square brackets]` is a production note and is **not read aloud**.
* Headings like `CHAPTER 1` or `Part II`, and any text before the first id, are skipped. The status box lists what was skipped.
* A script with no ids at all gets one narration per line, numbered `1`, `2`, `3`.
* Tick **Skip narrations already saved** to resume a batch after Colab disconnects.
* In **Random voice** mode the first narration's voice is reused for the rest.

Generating a name that already exists replaces that file, and the status box tells you when it did.

---

## ⚙️ Settings worth knowing

| Setting | Default | Notes |
|---|---|---|
| Inference steps | 10 | 16–32 is slightly better and slower. |
| Guidance scale | 1.2 | The dots.tts recommended value. |
| Speaker scale | 1.5 | How strongly the reference timbre is applied. |
| Seed | -1 | -1 picks a new one each run; the one used is shown in *Status*. |
| Max characters per chunk | 250 | Chinese, Japanese and Korean characters count as three. |
| Precision (notebook) | auto | float16 on T4, bfloat16 on A100 / L4 / H100. |

Whisper (reference transcripts and subtitles) runs on the GPU when its CUDA libraries load. If they don't (Colab's CUDA 13 runtime may not provide the CUDA 12 libraries it needs), it switches to the CPU automatically: slower, but it works.

---

## 💻 Run locally

Linux with an NVIDIA GPU and Python 3.10–3.12. The text normalizer dots.tts depends on (`pynini`) has no Windows wheels.

```bash
git clone https://github.com/hirannalaka19/Dots-Google-Colab.git
cd Dots-Google-Colab
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python app.py            # set DOTS_OUTPUT_ROOT to choose where projects are saved
python -m pytest tests   # unit tests; no GPU or model needed
```

---

## 🙌 Credit

* 👨‍💻 Colab notebook & Gradio app by [HiranNalaka](https://github.com/hirannalaka19)
* 👉 [studio-dots-ai/dots.tts](https://github.com/studio-dots-ai/dots.tts), the original dots.tts model and code (Apache-2.0)
* 🗣️ Transcripts and subtitles by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3-turbo)

---

## ⚠️ Disclaimer

Please use these models responsibly. Do not use them for harmful, misleading, or unethical purposes such as unauthorized voice cloning, impersonation, fraud, scams, deceptive deepfakes, or any illegal content. Clearly label AI-generated audio. You are responsible for what you generate.
