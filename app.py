import atexit
import json
import os
import re
import tempfile
from typing import Dict, Optional, Tuple

import gradio as gr
import pyttsx3
from faster_whisper import WhisperModel
from openai import OpenAI


# ============================================================
# OpenAI API settings
# ============================================================
# PowerShell:
#   $env:OPENAI_API_KEY="sk-..."
#   $env:OPENAI_MODEL="gpt-5.4-mini"
#   python app.py
#
# Install:
#   pip install gradio faster-whisper pyttsx3 openai
# ============================================================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# Local LLM settings
# ============================================================
# 默认使用 OpenAI。UI 里也可以切换到 Local LM Studio。
# LM Studio OpenAI-compatible endpoint 默认一般是：
#   http://localhost:1234/v1/chat/completions
#
# 也可以用环境变量覆盖：
#   $env:LOCAL_LLM_URL="http://localhost:1234/v1/chat/completions"
#   $env:LOCAL_LLM_MODEL="qwen2.5-1.5b-instruct-unsloth-bnb-thinker"
# ============================================================
import requests

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1/chat/completions")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-1.5b-instruct-unsloth-bnb-thinker")


# ============================================================
# Whisper settings
# ============================================================
# "small" 比 "base" 慢一点，但中英日识别更稳。
# 如果你的 Windows 电脑太慢，可以改回 "base"。
whisper_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)

DEFAULT_WORDS = [
    "reluctant",
    "compromise",
    "eventually",
    "practical",
    "wreak",
    "appall",
    "balk",
    "blemish",
    "allude",
    "backlash",
]

TEMP_AUDIO_FILES = []


# ============================================================
# Avatar CSS + JS
# ============================================================
AVATAR_CSS = r"""
:root {
  --teacher-mouth-open: 8px;
  --teacher-mouth-width: 36px;
  --teacher-mouth-radius: 0 0 20px 20px;
  --teacher-mouth-inner-opacity: 0.55;
}

.teacher-panel {
  width: 100%;
  min-height: 460px;
  border-radius: 28px;
  border: 1px solid rgba(255,255,255,.14);
  background:
    radial-gradient(circle at 50% 0%, rgba(255,255,255,.22), transparent 22%),
    radial-gradient(circle at 15% 15%, rgba(255,255,255,.12), transparent 18%),
    linear-gradient(180deg, #22345b 0%, #16223f 42%, #121a32 100%);
  box-shadow:
    0 24px 60px rgba(0,0,0,.28),
    inset 0 1px 0 rgba(255,255,255,.05);
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px 10px 24px;
}

.teacher-wrap {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  z-index: 2;
}

.teacher-stage {
  position: relative;
  width: 310px;
  height: 360px;
}

.teacher-shadow {
  position: absolute;
  left: 50%;
  bottom: 7px;
  transform: translateX(-50%);
  width: 210px;
  height: 28px;
  border-radius: 50%;
  background: rgba(0,0,0,.28);
  filter: blur(12px);
}

.teacher-hair-back {
  position: absolute;
  left: 50%;
  top: 8px;
  transform: translateX(-50%);
  width: 232px;
  height: 262px;
  background: linear-gradient(180deg, #f7e0a8 0%, #e8b35f 40%, #af6d28 100%);
  border-radius: 120px 120px 88px 88px;
  box-shadow: inset -18px -18px 0 rgba(111,62,13,.16);
}

.teacher-shoulders {
  position: absolute;
  left: 50%;
  bottom: 24px;
  transform: translateX(-50%);
  width: 240px;
  height: 116px;
  background: linear-gradient(180deg, #d7e6ff 0%, #9ab9e9 100%);
  border-radius: 90px 90px 28px 28px;
  border: 1px solid rgba(255,255,255,.24);
  box-shadow: inset 0 10px 22px rgba(255,255,255,.18);
}

.teacher-neck {
  position: absolute;
  left: 50%;
  top: 216px;
  transform: translateX(-50%);
  width: 58px;
  height: 58px;
  background: linear-gradient(180deg, #f8d7c3 0%, #e9b99a 100%);
  border-radius: 18px;
  z-index: 4;
}

.teacher-collar-left,
.teacher-collar-right {
  position: absolute;
  top: 256px;
  width: 50px;
  height: 64px;
  background: rgba(255,255,255,.95);
  z-index: 5;
  box-shadow: 0 4px 10px rgba(0,0,0,.06);
}
.teacher-collar-left {
  left: 116px;
  clip-path: polygon(0 0, 100% 0, 58% 100%, 22% 82%);
  transform: rotate(3deg);
}
.teacher-collar-right {
  right: 116px;
  clip-path: polygon(0 0, 100% 0, 78% 82%, 42% 100%);
  transform: rotate(-3deg);
}

.teacher-head {
  position: absolute;
  left: 50%;
  top: 40px;
  transform: translateX(-50%);
  width: 190px;
  height: 225px;
  z-index: 6;
}

.teacher-face {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, #ffe0cf 0%, #efbf9e 92%);
  border-radius: 42% 42% 46% 46% / 36% 36% 54% 54%;
  box-shadow:
    inset -10px -12px 0 rgba(141,76,45,.08),
    0 16px 26px rgba(0,0,0,.17);
}

.teacher-hair-front {
  position: absolute;
  left: 50%;
  top: -6px;
  transform: translateX(-50%);
  width: 204px;
  height: 102px;
  background: linear-gradient(180deg, #f7dfa2 0%, #d89a43 58%, #ac6f2d 100%);
  border-radius: 84px 84px 44px 44px;
  z-index: 8;
  box-shadow: inset -15px -8px 0 rgba(112,68,17,.18);
}

.teacher-fringe-left,
.teacher-fringe-right {
  position: absolute;
  top: 24px;
  width: 52px;
  height: 110px;
  background: linear-gradient(180deg, #edc973 0%, #bf7a2f 100%);
  z-index: 8;
}
.teacher-fringe-left {
  left: 14px;
  border-radius: 20px 0 40px 70px;
  transform: rotate(-8deg);
}
.teacher-fringe-right {
  right: 20px;
  width: 64px;
  height: 116px;
  border-radius: 0 20px 70px 40px;
  transform: rotate(14deg);
}

.teacher-ear {
  position: absolute;
  top: 108px;
  width: 20px;
  height: 34px;
  background: #efbd9e;
  border-radius: 50%;
  z-index: 5;
}
.teacher-ear.left { left: -2px; }
.teacher-ear.right { right: -2px; }

.teacher-eye {
  position: absolute;
  top: 90px;
  width: 28px;
  height: 16px;
  z-index: 10;
}
.teacher-eye.left { left: 44px; }
.teacher-eye.right { right: 44px; }

.teacher-eye-lid {
  position: absolute;
  inset: 0;
  border-top: 4px solid #6a483e;
  border-radius: 50%;
}
.teacher-eye-ball {
  position: absolute;
  left: 5px;
  top: 6px;
  width: 18px;
  height: 10px;
  background: linear-gradient(180deg, #6ca5d5 0%, #335b80 100%);
  border-radius: 50%;
  overflow: hidden;
}
.teacher-eye-ball::before {
  content: "";
  position: absolute;
  left: 5px;
  top: 0px;
  width: 8px;
  height: 10px;
  background: #1f2e40;
  border-radius: 50%;
}
.teacher-eye-ball::after {
  content: "";
  position: absolute;
  right: 3px;
  top: 1px;
  width: 4px;
  height: 4px;
  background: rgba(255,255,255,.95);
  border-radius: 50%;
}

.teacher-avatar.speaking .teacher-eye-ball,
.teacher-avatar.waiting .teacher-eye-ball {
  animation: teacherBlink 5.3s infinite;
}

.teacher-brow {
  position: absolute;
  top: 72px;
  width: 34px;
  height: 8px;
  border-top: 4px solid rgba(105,65,41,.55);
  border-radius: 50%;
  z-index: 10;
}
.teacher-brow.left { left: 40px; transform: rotate(-6deg); }
.teacher-brow.right { right: 40px; transform: rotate(6deg); }

.teacher-nose {
  position: absolute;
  left: 50%;
  top: 113px;
  transform: translateX(-50%);
  width: 18px;
  height: 28px;
  z-index: 10;
}
.teacher-nose::before {
  content: "";
  position: absolute;
  inset: 0;
  border-right: 3px solid rgba(156,92,68,.22);
  border-bottom: 3px solid rgba(156,92,68,.16);
  border-radius: 50%;
}

.teacher-cheek {
  position: absolute;
  top: 138px;
  width: 28px;
  height: 16px;
  background: rgba(255,132,153,.22);
  border-radius: 50%;
  z-index: 9;
}
.teacher-cheek.left { left: 32px; }
.teacher-cheek.right { right: 32px; }

.teacher-mouth {
  position: absolute;
  left: 50%;
  top: 154px;
  transform: translateX(-50%);
  width: var(--teacher-mouth-width);
  height: var(--teacher-mouth-open);
  background: linear-gradient(180deg, #8c2c4e 0%, #5d1734 100%);
  border-radius: var(--teacher-mouth-radius);
  z-index: 11;
  box-shadow:
    inset 0 -2px 0 rgba(45,7,18,.38),
    0 2px 6px rgba(0,0,0,.12);
  transition:
    width .08s linear,
    height .08s linear,
    border-radius .08s linear,
    transform .08s linear;
  overflow: hidden;
}

.teacher-mouth::before {
  content: "";
  position: absolute;
  left: 50%;
  bottom: 0;
  transform: translateX(-50%);
  width: calc(var(--teacher-mouth-width) * .62);
  height: calc(var(--teacher-mouth-open) * .42);
  background: rgba(255,179,195,var(--teacher-mouth-inner-opacity));
  border-radius: 50%;
}

.teacher-mouth::after {
  content: "";
  position: absolute;
  left: 50%;
  top: -1px;
  transform: translateX(-50%);
  width: calc(var(--teacher-mouth-width) * .95);
  height: 3px;
  background: rgba(255,220,226,.45);
  border-radius: 50%;
}

.teacher-avatar.speaking .teacher-head {
  animation: teacherHeadBob 1.3s ease-in-out infinite;
}

.teacher-bust-highlight {
  position: absolute;
  left: 50%;
  bottom: 44px;
  transform: translateX(-50%);
  width: 112px;
  height: 20px;
  background: rgba(255,255,255,.45);
  border-radius: 50%;
  filter: blur(12px);
  opacity: .45;
}

.teacher-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 13px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.05);
  color: #cfd8ef;
  font-size: 13px;
  margin-top: 10px;
}
.teacher-indicator::before {
  content: "";
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #64748b;
}

.teacher-avatar.speaking .teacher-indicator::before {
  background: #18c964;
  box-shadow: 0 0 0 5px rgba(24,201,100,.12);
}
.teacher-avatar.waiting .teacher-indicator::before {
  background: #8aa0c8;
}

.teacher-caption {
  color: #e6ecfa;
  text-align: center;
  font-size: 14px;
  line-height: 1.55;
  max-width: 290px;
  margin-top: 12px;
  opacity: .95;
}

.teacher-name {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: .2px;
  color: white;
}

@keyframes teacherBlink {
  0%, 92%, 100% { transform: scaleY(1); }
  94%, 96% { transform: scaleY(0.08); }
}

@keyframes teacherHeadBob {
  0%, 100% { transform: translateX(-50%) translateY(0px); }
  50% { transform: translateX(-50%) translateY(2px); }
}
"""

AVATAR_JS = r"""
function() {
  const state = {
    audioContext: null,
    analyser: null,
    dataArray: null,
    currentAudio: null,
    rafId: null,
    sourceMap: new WeakMap(),
    lastSpeakingAt: 0
  };

  function getAvatar() {
    return document.getElementById("teacher-avatar");
  }

  function setWaiting() {
    const avatar = getAvatar();
    if (!avatar) return;
    avatar.classList.remove("speaking");
    avatar.classList.add("waiting");
    avatar.style.setProperty("--teacher-mouth-open", "8px");
    avatar.style.setProperty("--teacher-mouth-width", "36px");
    avatar.style.setProperty("--teacher-mouth-radius", "0 0 20px 20px");
    avatar.style.setProperty("--teacher-mouth-inner-opacity", "0.55");
    const indicator = document.getElementById("teacher-indicator-text");
    if (indicator) indicator.textContent = "Waiting";
  }

  function setSpeaking(level) {
    const avatar = getAvatar();
    if (!avatar) return;
    avatar.classList.remove("waiting");
    avatar.classList.add("speaking");

    const px = Math.max(10, Math.min(34, 9 + level * 130));
    const w = Math.max(30, Math.min(48, 34 + level * 48));
    avatar.style.setProperty("--teacher-mouth-open", px + "px");
    avatar.style.setProperty("--teacher-mouth-width", w + "px");
    avatar.style.setProperty("--teacher-mouth-radius", level > 0.12 ? "40%" : "0 0 18px 18px");
    avatar.style.setProperty("--teacher-mouth-inner-opacity", String(Math.min(0.95, 0.45 + level * 2.6)));

    const indicator = document.getElementById("teacher-indicator-text");
    if (indicator) indicator.textContent = "Speaking";
  }

  function ensureAudioContext() {
    if (!state.audioContext) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      state.audioContext = new Ctx();
    }
    if (state.audioContext.state === "suspended") {
      state.audioContext.resume().catch(() => {});
    }
    return state.audioContext;
  }

  function attachToAudio(audio) {
    if (!audio || state.sourceMap.has(audio)) return;

    const ctx = ensureAudioContext();
    if (!ctx) return;

    try {
      const source = ctx.createMediaElementSource(audio);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.72;
      source.connect(analyser);
      analyser.connect(ctx.destination);
      state.sourceMap.set(audio, { source, analyser });
    } catch (e) {
      // Same audio element cannot be attached twice.
    }
  }

  function startTracking(audio) {
    if (!audio) return;
    attachToAudio(audio);
    const node = state.sourceMap.get(audio);
    if (!node) return;

    state.currentAudio = audio;
    state.analyser = node.analyser;
    state.dataArray = new Uint8Array(node.analyser.fftSize);

    if (!state.rafId) {
      loop();
    }
  }

  function findActiveAudio() {
    const audios = Array.from(document.querySelectorAll("audio"));
    for (const audio of audios) {
      if (audio && !audio.paused && !audio.ended && audio.currentTime > 0) {
        return audio;
      }
    }
    return null;
  }

  function loop() {
    const avatar = getAvatar();
    if (!avatar) {
      state.rafId = null;
      return;
    }

    const activeAudio = findActiveAudio();
    if (activeAudio && activeAudio !== state.currentAudio) {
      startTracking(activeAudio);
    }

    if (state.analyser && state.currentAudio && !state.currentAudio.paused && !state.currentAudio.ended) {
      state.analyser.getByteTimeDomainData(state.dataArray);
      let sum = 0;
      for (let i = 0; i < state.dataArray.length; i++) {
        const v = (state.dataArray[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / state.dataArray.length);

      if (rms > 0.012) {
        state.lastSpeakingAt = Date.now();
        setSpeaking(rms);
      } else {
        if (Date.now() - state.lastSpeakingAt < 120) {
          setSpeaking(0.05);
        } else {
          setSpeaking(0.01);
        }
      }

      state.rafId = requestAnimationFrame(loop);
      return;
    }

    setWaiting();
    state.currentAudio = null;
    state.analyser = null;
    state.dataArray = null;
    state.rafId = requestAnimationFrame(loop);
  }

  function bindAudio(audio) {
    if (!audio || audio.dataset.teacherBound === "1") return;
    audio.dataset.teacherBound = "1";

    attachToAudio(audio);

    audio.addEventListener("play", () => startTracking(audio));
    audio.addEventListener("playing", () => startTracking(audio));
    audio.addEventListener("pause", () => {
      setTimeout(() => {
        if (!findActiveAudio()) setWaiting();
      }, 60);
    });
    audio.addEventListener("ended", () => {
      setTimeout(() => {
        const nextAudio = findActiveAudio();
        if (nextAudio) startTracking(nextAudio);
        else setWaiting();
      }, 60);
    });
  }

  function scanAudios() {
    const audios = Array.from(document.querySelectorAll("audio"));
    audios.forEach(bindAudio);
  }

  const observer = new MutationObserver(() => {
    scanAudios();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  scanAudios();
  setWaiting();
  loop();

  document.addEventListener("click", () => {
    ensureAudioContext();
    scanAudios();
  }, true);
}
"""


# ============================================================
# Cleanup
# ============================================================
def cleanup_temp_audio_files() -> None:
    global TEMP_AUDIO_FILES
    remaining = []

    for path in TEMP_AUDIO_FILES:
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print("[TTS] Deleted old audio:", path)
        except Exception as e:
            print("[TTS] Failed to delete old audio:", path, e)
            remaining.append(path)

    TEMP_AUDIO_FILES = remaining


atexit.register(cleanup_temp_audio_files)


def delete_input_audio_file(audio_path: Optional[str]) -> None:
    if not audio_path:
        return

    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("[Audio Input] Deleted input audio:", audio_path)
    except Exception as e:
        print("[Audio Input] Failed to delete input audio:", audio_path, e)


# ============================================================
# Text helpers
# ============================================================
def clean_markdown_stars(text: str) -> str:
    """
    Remove markdown bold/list asterisks from model output.
    """
    return (text or "").replace("*", "").strip()


def extract_json_object(text: str) -> Optional[Dict[str, str]]:
    """
    Try to parse a JSON object from the model output.
    """
    if not text:
        return None

    raw = text.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def format_explanation(data: Dict[str, str]) -> str:
    """
    Format word explanation without markdown stars.
    """
    lines = [
        f"单词：{data.get('word', '')}",
        f"中文意思：{data.get('meaning_cn', '')}",
        f"词性：{data.get('part_of_speech_cn', '')}",
        f"常见用法：{data.get('usage_cn', '')}",
        f"英文例句：{data.get('example_en', '')}",
        f"中文翻译：{data.get('example_cn', '')}",
        f"练习问题：{data.get('practice_question_en', '')} {data.get('practice_question_cn', '')}",
    ]
    return clean_markdown_stars("\n".join(lines))


def make_explanation_tts_texts(data: Dict[str, str]) -> Tuple[str, str]:
    """
    Return (Chinese TTS text, English TTS text).
    Chinese and English are separated so they can be read by different voices.
    """
    word = data.get("word", "")
    zh_text = (
        f"{word}。"
        f"中文意思：{data.get('meaning_cn', '')}。"
        f"词性：{data.get('part_of_speech_cn', '')}。"
        f"常见用法：{data.get('usage_cn', '')}。"
        f"中文翻译：{data.get('example_cn', '')}。"
    )

    en_text = (
        f"{word}. "
        f"Example: {data.get('example_en', '')} "
        f"Practice question: {data.get('practice_question_en', '')}"
    )

    return clean_markdown_stars(zh_text), clean_markdown_stars(en_text)


# ============================================================
# Whisper
# ============================================================
def transcribe_audio(audio_path: Optional[str]) -> str:
    if not audio_path:
        print("[Whisper] No audio_path received")
        return ""

    print("[Whisper] Transcribing audio:", audio_path)

    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            # 不固定 language="en"，让 Whisper 自动识别英语 / 汉语 / 日语。
            # 这样你可以用中文问问题，也可以用英语练习。
            vad_filter=False,
            beam_size=5,
        )

        texts = []
        print("[Whisper] Detected language:", info.language)
        print("[Whisper] Duration:", info.duration)

        for seg in segments:
            print("[Whisper] Segment:", seg.text)
            texts.append(seg.text.strip())

        text = " ".join(texts).strip()
        print("[Whisper] Final text:", text)
        return text

    except Exception as e:
        print("[Whisper] Error:", e)
        return ""

    finally:
        delete_input_audio_file(audio_path)


# ============================================================
# OpenAI API
# ============================================================
def call_openai_text(target_word: str, transcript: str, mode: str) -> str:
    if client is None:
        return (
            "OpenAI API Key 未设置。\n\n"
            "请在 PowerShell 中这样启动：\n"
            '$env:OPENAI_API_KEY="sk-你的key"\n'
            "python app.py"
        )

    target_word = target_word.strip() if target_word else ""

    if mode == "conversation":
        system_prompt = """
You are a friendly English conversation partner and vocabulary tutor.

The user is a Chinese/Japanese bilingual learner practicing English.
The user may speak English, Chinese, or Japanese.

Very important:
- Answer ONLY the current speech transcript.
- Do NOT reuse previous explanations, previous examples, or previous answers.
- The current transcript is the user's latest message.
- If the current transcript is Chinese or Japanese, understand the meaning and help the user express it naturally in English.
- If the current transcript asks a new question, answer that new question directly.
- If the current transcript is an English sentence attempt, respond to that sentence.
- If the transcript says "this word", "the word", "new word", "这个单词", "这个词", "这个英语单词", "この単語", or a similar-sounding wrong word, understand it as the target word.
- Keep your reply fresh and specific to the current transcript.
- Encourage the user to use the target word naturally.
- Keep your English reply short: 1-3 sentences.
- Add one short Chinese note if useful.
- Do not use markdown asterisks.

Preferred response style:
1. If the user asks in Chinese/Japanese, answer briefly in Chinese first if needed.
2. Then give a natural English expression or reply.
3. Ask one simple English follow-up question when appropriate.
"""
        user_prompt = f"""
TARGET_WORD:
{target_word}

CURRENT_USER_TRANSCRIPT:
{transcript}

Instruction:
The transcript may be English, Chinese, or Japanese.
Reply to CURRENT_USER_TRANSCRIPT only.
If the user used Chinese or Japanese, help convert their idea into natural spoken English.
Do not repeat a previous answer.
"""

    elif mode == "correction":
        system_prompt = """
You are a friendly English sentence correction and translation tutor.

The user is a Chinese/Japanese bilingual learner practicing English.
The user may speak English, Chinese, or Japanese.

Very important:
- Work ONLY on the current speech transcript.
- Do NOT reuse previous corrections, previous examples, or previous answers.
- The current transcript is the user's latest sentence attempt or latest idea.
- If the current transcript is English, correct it naturally.
- If the current transcript is Chinese or Japanese, translate the user's intended meaning into natural spoken English.
- Try to include the target word in the corrected English sentence if appropriate.
- Explain the correction or translation briefly in Chinese.
- Ask one simple follow-up question in English using the target word.
- Keep the response concise.
- Do not use markdown asterisks.
"""
        user_prompt = f"""
TARGET_WORD:
{target_word}

CURRENT_USER_TRANSCRIPT:
{transcript}

Instruction:
The transcript may be English, Chinese, or Japanese.
If it is Chinese or Japanese, translate the intended meaning into natural spoken English.
Correct or translate CURRENT_USER_TRANSCRIPT only.
Do not repeat a previous answer.
"""

    else:
        system_prompt = "You are a helpful English tutor. Reply briefly and clearly. Do not use markdown asterisks."
        user_prompt = transcript

    print("[OpenAI] Calling API")
    print("[OpenAI] Model:", OPENAI_MODEL)
    print("[OpenAI] Mode:", mode)
    print("[OpenAI] Target word:", target_word)
    print("[OpenAI] Transcript:", transcript)

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.5,
            max_output_tokens=220,
        )

        reply = clean_markdown_stars(response.output_text or "")
        print("[OpenAI] Reply:", reply)
        return reply

    except Exception as e:
        print("[OpenAI] Error:", e)
        return f"OpenAI API 调用失败：{e}"


def call_openai_explain_json(target_word: str) -> Dict[str, str]:
    """
    Ask OpenAI to return JSON for word explanation.
    This makes Chinese/English TTS separation stable.
    """
    if client is None:
        return {
            "word": target_word,
            "meaning_cn": "OpenAI API Key 未设置。",
            "part_of_speech_cn": "",
            "usage_cn": "请在 PowerShell 中设置 OPENAI_API_KEY 后重新启动。",
            "example_en": "",
            "example_cn": "",
            "practice_question_en": "",
            "practice_question_cn": "",
        }

    system_prompt = """
You are a friendly English vocabulary tutor for a Chinese learner.

Return strict JSON only. No markdown. No asterisks.

JSON schema:
{
  "word": "target word",
  "meaning_cn": "Chinese meaning",
  "part_of_speech_cn": "part of speech in Chinese, with English term in parentheses if useful",
  "usage_cn": "common usage explained in Chinese",
  "example_en": "one short natural English example sentence",
  "example_cn": "Chinese translation of the example sentence",
  "practice_question_en": "one short English practice question using the word",
  "practice_question_cn": "Chinese translation of the practice question"
}
"""

    user_prompt = f"Target word: {target_word}"

    print("[OpenAI] Calling API for word explanation JSON")
    print("[OpenAI] Model:", OPENAI_MODEL)
    print("[OpenAI] Target word:", target_word)

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.2,
            max_output_tokens=260,
        )

        output_text = clean_markdown_stars(response.output_text or "")
        print("[OpenAI] Explanation raw:", output_text)

        data = extract_json_object(output_text)

        if not data:
            # Fallback if model does not return JSON.
            return {
                "word": target_word,
                "meaning_cn": output_text,
                "part_of_speech_cn": "",
                "usage_cn": "",
                "example_en": "",
                "example_cn": "",
                "practice_question_en": "",
                "practice_question_cn": "",
            }

        # Clean every field.
        cleaned = {}
        for key, value in data.items():
            cleaned[key] = clean_markdown_stars(str(value))

        cleaned.setdefault("word", target_word)
        cleaned.setdefault("meaning_cn", "")
        cleaned.setdefault("part_of_speech_cn", "")
        cleaned.setdefault("usage_cn", "")
        cleaned.setdefault("example_en", "")
        cleaned.setdefault("example_cn", "")
        cleaned.setdefault("practice_question_en", "")
        cleaned.setdefault("practice_question_cn", "")

        return cleaned

    except Exception as e:
        print("[OpenAI] Error:", e)
        return {
            "word": target_word,
            "meaning_cn": f"OpenAI API 调用失败：{e}",
            "part_of_speech_cn": "",
            "usage_cn": "",
            "example_en": "",
            "example_cn": "",
            "practice_question_en": "",
            "practice_question_cn": "",
        }



# ============================================================
# Local LLM API
# ============================================================
def call_local_llm_text(target_word: str, transcript: str, mode: str) -> str:
    """
    调用本地 LM Studio / OpenAI-compatible API。
    只有在 UI 里选择 Local LM Studio 时才使用。
    """
    target_word = target_word.strip() if target_word else ""

    if mode == "conversation":
        system_prompt = """
You are a friendly English conversation partner and vocabulary tutor.

The user is a Chinese/Japanese bilingual learner practicing English.
The user may speak English, Chinese, or Japanese.

Very important:
- Answer ONLY the current speech transcript.
- Do NOT reuse previous explanations, previous examples, or previous answers.
- The current transcript is the user's latest message.
- If the current transcript is Chinese or Japanese, understand the meaning and help the user express it naturally in English.
- If the current transcript asks a new question, answer that new question directly.
- If the current transcript is an English sentence attempt, respond to that sentence.
- If the transcript says "this word", "the word", "new word", "这个单词", "这个词", "这个英语单词", "この単語", or a similar-sounding wrong word, understand it as the target word.
- Keep your reply fresh and specific to the current transcript.
- Encourage the user to use the target word naturally.
- Keep your English reply short: 1-3 sentences.
- Add one short Chinese note if useful.
- Do not use markdown asterisks.
"""
        user_prompt = f"""
TARGET_WORD:
{target_word}

CURRENT_USER_TRANSCRIPT:
{transcript}

Instruction:
The transcript may be English, Chinese, or Japanese.
Reply to CURRENT_USER_TRANSCRIPT only.
If the user used Chinese or Japanese, help convert their idea into natural spoken English.
Do not repeat a previous answer.
"""

    elif mode == "correction":
        system_prompt = """
You are a friendly English sentence correction and translation tutor.

The user is a Chinese/Japanese bilingual learner practicing English.
The user may speak English, Chinese, or Japanese.

Very important:
- Work ONLY on the current speech transcript.
- Do NOT reuse previous corrections, previous examples, or previous answers.
- The current transcript is the user's latest sentence attempt or latest idea.
- If the current transcript is English, correct it naturally.
- If the current transcript is Chinese or Japanese, translate the user's intended meaning into natural spoken English.
- Try to include the target word in the corrected English sentence if appropriate.
- Explain the correction or translation briefly in Chinese.
- Ask one simple follow-up question in English using the target word.
- Keep the response concise.
- Do not use markdown asterisks.
"""
        user_prompt = f"""
TARGET_WORD:
{target_word}

CURRENT_USER_TRANSCRIPT:
{transcript}

Instruction:
The transcript may be English, Chinese, or Japanese.
If it is Chinese or Japanese, translate the intended meaning into natural spoken English.
Correct or translate CURRENT_USER_TRANSCRIPT only.
Do not repeat a previous answer.
"""

    else:
        system_prompt = "You are a helpful English tutor. Reply briefly and clearly. Do not use markdown asterisks."
        user_prompt = transcript

    print("[Local LLM] Calling local API")
    print("[Local LLM] URL:", LOCAL_LLM_URL)
    print("[Local LLM] Model:", LOCAL_LLM_MODEL)
    print("[Local LLM] Mode:", mode)
    print("[Local LLM] Target word:", target_word)
    print("[Local LLM] Transcript:", transcript)

    try:
        response = requests.post(
            LOCAL_LLM_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                "temperature": 0.5,
                "max_tokens": 220,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        reply = clean_markdown_stars(reply)
        print("[Local LLM] Reply:", reply)
        return reply

    except Exception as e:
        print("[Local LLM] Error:", e)
        return (
            "本地 LLM 调用失败："
            f"{e}\\n\\n"
            "请确认 LM Studio Server 已启动，并且 LOCAL_LLM_URL / LOCAL_LLM_MODEL 设置正确。"
        )


def call_local_llm_explain_json(target_word: str) -> Dict[str, str]:
    """
    用本地 LM Studio 返回单词解释 JSON。
    """
    system_prompt = """
You are a friendly English vocabulary tutor for a Chinese learner.

Return strict JSON only. No markdown. No asterisks.

JSON schema:
{
  "word": "target word",
  "meaning_cn": "Chinese meaning",
  "part_of_speech_cn": "part of speech in Chinese, with English term in parentheses if useful",
  "usage_cn": "common usage explained in Chinese",
  "example_en": "one short natural English example sentence",
  "example_cn": "Chinese translation of the example sentence",
  "practice_question_en": "one short English practice question using the word",
  "practice_question_cn": "Chinese translation of the practice question"
}
"""
    user_prompt = f"Target word: {target_word}"

    print("[Local LLM] Calling local API for word explanation JSON")
    print("[Local LLM] URL:", LOCAL_LLM_URL)
    print("[Local LLM] Model:", LOCAL_LLM_MODEL)
    print("[Local LLM] Target word:", target_word)

    try:
        response = requests.post(
            LOCAL_LLM_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                "temperature": 0.2,
                "max_tokens": 260,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        output_text = clean_markdown_stars(data["choices"][0]["message"]["content"].strip())
        print("[Local LLM] Explanation raw:", output_text)

        parsed = extract_json_object(output_text)
        if not parsed:
            return {
                "word": target_word,
                "meaning_cn": output_text,
                "part_of_speech_cn": "",
                "usage_cn": "",
                "example_en": "",
                "example_cn": "",
                "practice_question_en": "",
                "practice_question_cn": "",
            }

        cleaned = {}
        for key, value in parsed.items():
            cleaned[key] = clean_markdown_stars(str(value))

        cleaned.setdefault("word", target_word)
        cleaned.setdefault("meaning_cn", "")
        cleaned.setdefault("part_of_speech_cn", "")
        cleaned.setdefault("usage_cn", "")
        cleaned.setdefault("example_en", "")
        cleaned.setdefault("example_cn", "")
        cleaned.setdefault("practice_question_en", "")
        cleaned.setdefault("practice_question_cn", "")
        return cleaned

    except Exception as e:
        print("[Local LLM] Error:", e)
        return {
            "word": target_word,
            "meaning_cn": f"本地 LLM 调用失败：{e}",
            "part_of_speech_cn": "",
            "usage_cn": "请确认 LM Studio Server 已启动，并且 LOCAL_LLM_URL / LOCAL_LLM_MODEL 设置正确。",
            "example_en": "",
            "example_cn": "",
            "practice_question_en": "",
            "practice_question_cn": "",
        }


def call_llm_text(llm_provider: str, target_word: str, transcript: str, mode: str) -> str:
    """
    根据 UI 选择调用 OpenAI 或本地 LLM。
    """
    if llm_provider == "Local LM Studio":
        return call_local_llm_text(target_word, transcript, mode)
    return call_openai_text(target_word, transcript, mode)


def call_llm_explain_json(llm_provider: str, target_word: str) -> Dict[str, str]:
    """
    根据 UI 选择调用 OpenAI 或本地 LLM 生成单词解释。
    """
    if llm_provider == "Local LM Studio":
        return call_local_llm_explain_json(target_word)
    return call_openai_explain_json(target_word)


# ============================================================
# TTS
# ============================================================
def select_voice(engine: pyttsx3.Engine, language: str) -> None:
    """
    Select voice by language:
    - language="en": English female first
    - language="zh": Chinese voice first
    """
    voices = engine.getProperty("voices")
    language = language.lower().strip()

    if language == "zh":
        preferred_keywords = [
            "huihui", "yaoyao", "xiaoxiao", "xiaoyi", "xiaobei",
            "kangkang", "chinese", "mandarin", "zh-cn", "zh_"
        ]
        fallback_keywords = ["zh", "china", "chinese", "mandarin"]

        for voice in voices:
            name = (voice.name or "").lower()
            voice_id = (voice.id or "").lower()
            combined = name + " " + voice_id

            if any(keyword in combined for keyword in preferred_keywords):
                engine.setProperty("voice", voice.id)
                print("[TTS] Selected Chinese voice:", voice.name)
                return

        for voice in voices:
            name = (voice.name or "").lower()
            voice_id = (voice.id or "").lower()
            combined = name + " " + voice_id

            if any(keyword in combined for keyword in fallback_keywords):
                engine.setProperty("voice", voice.id)
                print("[TTS] Selected fallback Chinese voice:", voice.name)
                return

        print("[TTS] No Chinese voice found. Using default voice.")
        return

    female_keywords = [
        "zira", "aria", "jenny", "sonia", "hazel",
        "susan", "heather", "samantha", "female"
    ]

    english_keywords = [
        "english", "en-us", "en-gb", "en_", "en-"
    ]

    for voice in voices:
        name = (voice.name or "").lower()
        voice_id = (voice.id or "").lower()
        combined = name + " " + voice_id

        is_female = any(keyword in combined for keyword in female_keywords)
        is_english = any(keyword in combined for keyword in english_keywords)

        if is_female and is_english:
            engine.setProperty("voice", voice.id)
            print("[TTS] Selected female English voice:", voice.name)
            return

    for voice in voices:
        name = (voice.name or "").lower()
        voice_id = (voice.id or "").lower()
        combined = name + " " + voice_id

        is_english = any(keyword in combined for keyword in english_keywords)

        if is_english:
            engine.setProperty("voice", voice.id)
            print("[TTS] Selected English voice:", voice.name)
            return

    print("[TTS] No English voice found. Using default voice.")


def create_tts_file(text: str, language: str = "en") -> Optional[str]:
    """
    Create one TTS wav file. Does not clean old files.
    Caller should call cleanup_temp_audio_files() before creating a new group of TTS files.
    """
    global TEMP_AUDIO_FILES

    text = clean_markdown_stars(text)

    if not text:
        return None

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
    ).name

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165 if language == "en" else 175)
        engine.setProperty("volume", 1.0)
        select_voice(engine, language)

        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()

        TEMP_AUDIO_FILES.append(output_path)
        print(f"[TTS] Saved {language} audio:", output_path)
        return output_path

    except Exception as e:
        print("[TTS] Error:", e)

        try:
            if os.path.exists(output_path):
                os.remove(output_path)
                print("[TTS] Deleted failed audio:", output_path)
        except Exception as delete_error:
            print("[TTS] Failed to delete failed audio:", delete_error)

        return None


def text_to_speech_file(text: str, language: str = "en") -> Optional[str]:
    """
    Cleanup old TTS files and create a new single TTS audio.
    """
    cleanup_temp_audio_files()
    return create_tts_file(text, language)


# ============================================================
# Gradio event functions
# ============================================================
def explain_word(llm_provider: str, target_word: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Word explanation:
    - display clean explanation without *
    - generate Chinese TTS audio
    - generate English TTS audio
    """
    if not target_word:
        return "请先选择或输入一个单词。", None, None

    data = call_llm_explain_json(llm_provider, target_word)
    display_text = format_explanation(data)
    zh_tts_text, en_tts_text = make_explanation_tts_texts(data)

    # For explanation, generate two audio files as one group.
    # Do not clean between Chinese and English generation.
    cleanup_temp_audio_files()
    zh_audio = create_tts_file(zh_tts_text, language="zh")
    en_audio = create_tts_file(en_tts_text, language="en")

    return display_text, zh_audio, en_audio


def voice_conversation(llm_provider: str, target_word: str, audio_path: Optional[str]):
    print("[App] voice_conversation audio_path =", audio_path)

    if not audio_path:
        return (
            "没有收到录音文件。请先在麦克风控件里录音，停止录音后再点击按钮。",
            "",
            None,
            gr.update(value=None),
        )

    transcript = transcribe_audio(audio_path)

    if not transcript:
        return (
            "Whisper 没有识别到语音。请说长一点、声音大一点，并确认麦克风权限正常。",
            "",
            None,
            gr.update(value=None),
        )

    print("[App] Current conversation transcript sent to OpenAI:", transcript)
    reply = call_llm_text(llm_provider, target_word, transcript, "conversation")
    audio_reply = text_to_speech_file(reply, language="en")

    return transcript, reply, audio_reply, gr.update(value=None)


def correct_sentence(llm_provider: str, target_word: str, audio_path: Optional[str]):
    print("[App] correct_sentence audio_path =", audio_path)

    if not audio_path:
        return (
            "没有收到录音文件。请先在麦克风控件里录音，停止录音后再点击按钮。",
            "",
            None,
            gr.update(value=None),
        )

    transcript = transcribe_audio(audio_path)

    if not transcript:
        return (
            "Whisper 没有识别到语音。请说长一点、声音大一点，并确认麦克风权限正常。",
            "",
            None,
            gr.update(value=None),
        )

    print("[App] Current correction transcript sent to OpenAI:", transcript)
    reply = call_llm_text(llm_provider, target_word, transcript, "correction")
    audio_reply = text_to_speech_file(reply, language="en")

    return transcript, reply, audio_reply, gr.update(value=None)


def update_words_from_text(words_text: str):
    if not words_text:
        return gr.update(choices=DEFAULT_WORDS, value=DEFAULT_WORDS[0])

    separators = [",", "，", "、", ";", "；", "\n", "\t"]
    normalized = words_text

    for sep in separators:
        normalized = normalized.replace(sep, " ")

    words = []
    for item in normalized.split(" "):
        word = item.strip()
        if word and word not in words:
            words.append(word)

    if not words:
        words = DEFAULT_WORDS

    return gr.update(choices=words, value=words[0])


TEACHER_HTML = r"""
<div id="teacher-avatar" class="teacher-avatar teacher-panel waiting">
  <div class="teacher-wrap">
    <div class="teacher-stage">
      <div class="teacher-shadow"></div>
      <div class="teacher-hair-back"></div>
      <div class="teacher-shoulders"></div>
      <div class="teacher-neck"></div>
      <div class="teacher-collar-left"></div>
      <div class="teacher-collar-right"></div>
      <div class="teacher-bust-highlight"></div>

      <div class="teacher-head">
        <div class="teacher-ear left"></div>
        <div class="teacher-ear right"></div>
        <div class="teacher-face"></div>
        <div class="teacher-hair-front"></div>
        <div class="teacher-fringe-left"></div>
        <div class="teacher-fringe-right"></div>

        <div class="teacher-brow left"></div>
        <div class="teacher-brow right"></div>

        <div class="teacher-eye left">
          <div class="teacher-eye-lid"></div>
          <div class="teacher-eye-ball"></div>
        </div>
        <div class="teacher-eye right">
          <div class="teacher-eye-lid"></div>
          <div class="teacher-eye-ball"></div>
        </div>

        <div class="teacher-nose"></div>
        <div class="teacher-cheek left"></div>
        <div class="teacher-cheek right"></div>
        <div class="teacher-mouth"></div>
      </div>
    </div>

    <div class="teacher-indicator">
      <span id="teacher-indicator-text">Waiting</span>
    </div>
    <div class="teacher-caption">
      <div class="teacher-name">Emily · Your English Teacher</div>
      <div>When the audio plays, her mouth changes with the voice volume.</div>
    </div>
  </div>
</div>
"""


with gr.Blocks(
    title="LLM Voice Tutor",
    css=AVATAR_CSS,
    js=AVATAR_JS,
) as demo:
    gr.Markdown("# LLM Voice Tutor")
    gr.Markdown("Python + Gradio + faster-whisper + OpenAI API / Local LM Studio 本地英语口语练习（支持中文 / 英文 / 日文语音输入）")

    llm_provider = gr.Radio(
        choices=["OpenAI API", "Local LM Studio"],
        value="OpenAI API",
        label="LLM 调用方式",
        info="默认使用 OpenAI API；如果选择 Local LM Studio，请先启动 LM Studio Server。",
    )

    with gr.Accordion("本地 LLM 连接信息（只读，修改请用环境变量）", open=False):
        gr.Markdown(
            f"""
当前本地 LLM URL：`{LOCAL_LLM_URL}`

当前本地模型名：`{LOCAL_LLM_MODEL}`

如需修改，在 PowerShell 启动前设置：

`$env:LOCAL_LLM_URL="http://localhost:1234/v1/chat/completions"`

`$env:LOCAL_LLM_MODEL="你的本地模型名"`
"""
        )

    with gr.Row():
        with gr.Column(scale=1):
            words_text = gr.Textbox(
                label="新单词列表",
                value="\n".join(DEFAULT_WORDS),
                lines=10,
                placeholder="每行一个单词，或用逗号/空格分隔",
            )

            update_words_btn = gr.Button("更新单词列表")

            target_word = gr.Dropdown(
                choices=DEFAULT_WORDS,
                value=DEFAULT_WORDS[0],
                label="当前练习单词",
                allow_custom_value=True,
            )

            explain_btn = gr.Button("单词解释")

        with gr.Column(scale=1):
            gr.HTML(TEACHER_HTML)

        with gr.Column(scale=2):
            explain_output = gr.Textbox(
                label="单词解释（已去掉 *，中英文分离朗读）",
                lines=10,
            )

            with gr.Row():
                explain_audio_zh_output = gr.Audio(
                    label="中文解释朗读",
                    type="filepath",
                    autoplay=False,
                )

                explain_audio_en_output = gr.Audio(
                    label="英文例句朗读",
                    type="filepath",
                    autoplay=True,
                )

    update_words_btn.click(
        fn=update_words_from_text,
        inputs=[words_text],
        outputs=[target_word],
    )

    explain_btn.click(
        fn=explain_word,
        inputs=[llm_provider, target_word],
        outputs=[explain_output, explain_audio_zh_output, explain_audio_en_output],
    )

    gr.Markdown("## 语音练习")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="录音输入（可说英语 / 中文 / 日语）",
    )

    with gr.Row():
        chat_btn = gr.Button("和 LLM 对话")
        correction_btn = gr.Button("纠正我的造句")

    transcript_output = gr.Textbox(
        label="Whisper 识别结果（自动识别中/英/日）",
        lines=4,
    )

    reply_output = gr.Textbox(
        label="LLM 回答",
        lines=10,
    )

    audio_reply_output = gr.Audio(
        label="回答朗读",
        type="filepath",
        autoplay=True,
    )

    chat_btn.click(
        fn=voice_conversation,
        inputs=[llm_provider, target_word, audio_input],
        outputs=[transcript_output, reply_output, audio_reply_output, audio_input],
    )

    correction_btn.click(
        fn=correct_sentence,
        inputs=[llm_provider, target_word, audio_input],
        outputs=[transcript_output, reply_output, audio_reply_output, audio_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
