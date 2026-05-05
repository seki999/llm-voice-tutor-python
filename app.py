import atexit
import os
import tempfile
from typing import Optional, Tuple

import gradio as gr
import pyttsx3
import requests
from faster_whisper import WhisperModel


LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct-unsloth-bnb-thinker"

# Windows CPU 推荐先用 base。
# 如果识别不准，再改成 "small"。
# 第一次运行会自动下载模型，需要等一会儿。
whisper_model = WhisperModel(
    "base",
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

# 保存本程序生成的临时 TTS 音频文件路径。
# 策略：生成新音频前删除旧音频；程序退出时也尝试删除剩余音频。
TEMP_AUDIO_FILES = []


AVATAR_CSS = """
.teacher-panel {
  width: 100%;
  min-height: 390px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 24px;
  background:
    radial-gradient(circle at 20% 10%, rgba(255,255,255,.24), transparent 28%),
    linear-gradient(145deg, #1b2446 0%, #11182f 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  box-shadow: 0 18px 45px rgba(0,0,0,.25);
  overflow: hidden;
  position: relative;
}

.teacher-stage {
  position: relative;
  width: 270px;
  height: 310px;
}

.teacher-hair-back {
  position: absolute;
  left: 50%;
  top: 6px;
  transform: translateX(-50%);
  width: 205px;
  height: 245px;
  background: linear-gradient(160deg, #f3d58e, #c58d3a);
  border-radius: 95px 95px 75px 75px;
  box-shadow: inset -16px -12px 0 rgba(92,50,18,.12);
}

.teacher-neck {
  position: absolute;
  left: 50%;
  top: 174px;
  transform: translateX(-50%);
  width: 54px;
  height: 54px;
  background: #f1c7aa;
  border-radius: 18px;
}

.teacher-body {
  position: absolute;
  left: 50%;
  bottom: 0;
  transform: translateX(-50%);
  width: 220px;
  height: 105px;
  background: linear-gradient(180deg, #eaf2ff, #95b7ec);
  border-radius: 70px 70px 20px 20px;
  border: 1px solid rgba(255,255,255,.35);
}

.teacher-shirt {
  position: absolute;
  left: 50%;
  bottom: 0;
  transform: translateX(-50%);
  width: 104px;
  height: 88px;
  background: white;
  clip-path: polygon(0 0, 100% 0, 78% 100%, 22% 100%);
  opacity: .96;
}

.teacher-face {
  position: absolute;
  left: 50%;
  top: 34px;
  transform: translateX(-50%);
  width: 156px;
  height: 178px;
  background: linear-gradient(180deg, #ffd9c0, #f0b995);
  border-radius: 48% 48% 46% 46%;
  box-shadow:
    inset -8px -10px 0 rgba(157,87,52,.08),
    0 12px 22px rgba(0,0,0,.18);
}

.teacher-bang {
  position: absolute;
  left: 54px;
  top: 24px;
  width: 165px;
  height: 78px;
  background: linear-gradient(150deg, #f7dc98, #bf8130);
  border-radius: 75px 75px 38px 38px;
  transform: rotate(-3deg);
  z-index: 3;
}

.teacher-bang::after {
  content: "";
  position: absolute;
  right: 4px;
  top: 34px;
  width: 58px;
  height: 84px;
  background: linear-gradient(160deg, #d69a42, #a76727);
  border-radius: 40px 20px 60px 20px;
  transform: rotate(-18deg);
}

.teacher-eye {
  position: absolute;
  top: 92px;
  width: 20px;
  height: 13px;
  background: #304062;
  border-radius: 50%;
  z-index: 4;
}

.teacher-eye.left { left: 91px; }
.teacher-eye.right { right: 91px; }

.teacher-eye::after {
  content: "";
  position: absolute;
  right: 4px;
  top: 2px;
  width: 5px;
  height: 5px;
  background: white;
  border-radius: 50%;
  opacity: .85;
}

.teacher-avatar.speaking .teacher-eye {
  animation: teacher-blink 5.2s infinite;
}

.teacher-brow {
  position: absolute;
  top: 79px;
  width: 28px;
  height: 6px;
  border-top: 4px solid rgba(85,51,30,.48);
  border-radius: 50%;
  z-index: 4;
}

.teacher-brow.left { left: 86px; transform: rotate(-4deg); }
.teacher-brow.right { right: 86px; transform: rotate(4deg); }

.teacher-nose {
  position: absolute;
  left: 50%;
  top: 112px;
  transform: translateX(-50%);
  width: 15px;
  height: 23px;
  border-right: 3px solid rgba(150,85,58,.22);
  border-bottom: 3px solid rgba(150,85,58,.18);
  border-radius: 50%;
  z-index: 4;
}

.teacher-mouth {
  position: absolute;
  left: 50%;
  top: 149px;
  transform: translateX(-50%);
  width: 42px;
  height: 10px;
  background: #a54c5e;
  border-radius: 0 0 25px 25px;
  z-index: 4;
  overflow: hidden;
  box-shadow: inset 0 -4px 0 rgba(77,18,32,.32);
}

.teacher-mouth::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -3px;
  transform: translateX(-50%);
  width: 30px;
  height: 8px;
  background: #f0a4ae;
  border-radius: 50%;
  opacity: .75;
}

.teacher-avatar.speaking .teacher-mouth {
  animation: teacher-talk-mouth .18s infinite alternate ease-in-out;
}

.teacher-avatar.speaking .teacher-head {
  animation: teacher-head-bob 1.2s infinite ease-in-out;
}

.teacher-caption {
  text-align: center;
  color: #dce6ff;
  font-size: 15px;
  line-height: 1.5;
  padding: 0 18px 18px;
}

.teacher-speaking-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #91a4c7;
  font-size: 13px;
  padding: 6px 12px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 999px;
  background: rgba(255,255,255,.05);
}

.teacher-speaking-indicator::before {
  content: "";
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #64748b;
}

.teacher-avatar.speaking .teacher-speaking-indicator::before {
  background: #18c964;
  box-shadow: 0 0 0 5px rgba(24,201,100,.16);
}

.teacher-ear {
  position: absolute;
  top: 108px;
  width: 20px;
  height: 32px;
  background: #efbd9e;
  border-radius: 50%;
  z-index: 2;
}

.teacher-ear.left { left: 53px; }
.teacher-ear.right { right: 53px; }

.teacher-cheek {
  position: absolute;
  top: 130px;
  width: 24px;
  height: 13px;
  background: rgba(255,133,142,.22);
  border-radius: 50%;
  z-index: 4;
}

.teacher-cheek.left { left: 75px; }
.teacher-cheek.right { right: 75px; }

@keyframes teacher-talk-mouth {
  0% { height: 8px; width: 36px; border-radius: 0 0 24px 24px; }
  45% { height: 18px; width: 40px; border-radius: 45%; }
  100% { height: 28px; width: 35px; border-radius: 48%; }
}

@keyframes teacher-head-bob {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(2px); }
}

@keyframes teacher-blink {
  0%, 92%, 100% { transform: scaleY(1); }
  95% { transform: scaleY(.08); }
}
"""

AVATAR_JS = """
function() {
  function updateTeacherSpeakingState() {
    const avatar = document.getElementById("teacher-avatar");
    const indicator = document.getElementById("teacher-indicator-text");
    if (!avatar) return;

    const audios = Array.from(document.querySelectorAll("audio"));
    const speaking = audios.some(audio => {
      return audio && !audio.paused && !audio.ended && audio.currentTime > 0;
    });

    if (speaking) {
      avatar.classList.add("speaking");
      if (indicator) indicator.textContent = "Speaking";
    } else {
      avatar.classList.remove("speaking");
      if (indicator) indicator.textContent = "Waiting";
    }
  }

  setInterval(updateTeacherSpeakingState, 120);

  document.addEventListener("play", updateTeacherSpeakingState, true);
  document.addEventListener("pause", updateTeacherSpeakingState, true);
  document.addEventListener("ended", updateTeacherSpeakingState, true);

  const observer = new MutationObserver(updateTeacherSpeakingState);
  observer.observe(document.body, { childList: true, subtree: true });

  setTimeout(updateTeacherSpeakingState, 500);
}
"""


def cleanup_temp_audio_files() -> None:
    """
    删除之前生成的临时 TTS 音频文件。

    注意：
    不要在返回给 Gradio 后立刻删除刚生成的 TTS 文件，
    否则浏览器可能还没来得及播放。
    所以本函数通常在“下一次生成新音频之前”清理旧音频。
    """
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
    """
    删除 Gradio 麦克风录音生成的输入音频文件。
    这个文件在 Whisper 转写完成后就可以删除。
    """
    if not audio_path:
        return

    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("[Audio Input] Deleted input audio:", audio_path)
    except Exception as e:
        print("[Audio Input] Failed to delete input audio:", audio_path, e)


def transcribe_audio(audio_path: Optional[str]) -> str:
    """
    使用 faster-whisper 识别 Gradio 录音文件。
    转写完成后，立即删除输入音频文件，方便连续录音。
    """
    if not audio_path:
        print("[Whisper] No audio_path received")
        return ""

    print("[Whisper] Transcribing audio:", audio_path)

    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            language="en",
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
        # Whisper 已经完成读取后，马上删除输入录音文件。
        delete_input_audio_file(audio_path)


def call_lm_studio(target_word: str, transcript: str, mode: str) -> str:
    """
    调用 LM Studio 的 OpenAI-compatible API。

    mode:
      - explain: 单词解释
      - conversation: 语音对话
      - correction: 纠正造句
    """
    target_word = target_word.strip() if target_word else ""

    if mode == "explain":
        system_prompt = """
You are a friendly English vocabulary tutor.

Explain the target word to a Chinese learner.

Include:
1. Chinese meaning
2. Part of speech
3. Common usage
4. One short English example sentence
5. Chinese translation of the example
6. One simple question to help the learner practice

Use clear Chinese explanation. Keep it concise.
"""
        user_prompt = f"""
Target word: {target_word}
"""

    elif mode == "conversation":
        system_prompt = """
You are a friendly English conversation partner and vocabulary tutor.

The user is practicing one target English word.

Important:
- The speech recognition transcript may contain errors.
- If the user says "this word", "the word", "new word", or a similar-sounding wrong word, understand it as the target word.
- Continue the conversation naturally.
- Encourage the user to use the target word.
- Keep your English reply short: 1-3 sentences.
- Add one short Chinese note if useful.
"""
        user_prompt = f"""
Target word: {target_word}
User speech transcript:
{transcript}
"""

    elif mode == "correction":
        system_prompt = """
You are a friendly English sentence correction tutor.

The user is practicing one target English word.

Your tasks:
1. Understand the user's sentence from the speech transcript.
2. Correct the sentence naturally.
3. Try to include the target word in the corrected sentence if appropriate.
4. Explain the correction briefly in Chinese.
5. Ask one simple follow-up question in English using the target word.

Keep the response concise.
"""
        user_prompt = f"""
Target word: {target_word}
User speech transcript:
{transcript}
"""

    else:
        system_prompt = """
You are a helpful English tutor.
Reply briefly and clearly.
"""
        user_prompt = transcript

    print("[LLM] Calling LM Studio")
    print("[LLM] Mode:", mode)
    print("[LLM] Target word:", target_word)
    print("[LLM] Transcript:", transcript)

    try:
        response = requests.post(
            LM_STUDIO_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                "temperature": 0.3,
                "max_tokens": 220,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()

        reply = data["choices"][0]["message"]["content"].strip()
        print("[LLM] Reply:", reply)

        return reply

    except Exception as e:
        print("[LLM] Error:", e)
        return f"调用 LM Studio 失败：{e}"


def select_female_english_voice(engine: pyttsx3.Engine) -> None:
    """
    优先选择 Windows 上的英文女声。
    常见英文女声：Zira / Aria / Jenny / Sonia / Hazel / Samantha 等。
    如果找不到女声，则退回英文 voice；再找不到则使用系统默认 voice。
    """
    voices = engine.getProperty("voices")

    female_keywords = [
        "zira",
        "aria",
        "jenny",
        "sonia",
        "hazel",
        "susan",
        "heather",
        "samantha",
        "female",
    ]

    english_keywords = [
        "english",
        "en-us",
        "en-gb",
        "en_",
        "en-",
    ]

    selected_voice_id = None

    # 第一轮：优先找英文女声
    for voice in voices:
        name = (voice.name or "").lower()
        voice_id = (voice.id or "").lower()
        combined = name + " " + voice_id

        is_female = any(keyword in combined for keyword in female_keywords)
        is_english = any(keyword in combined for keyword in english_keywords)

        if is_female and is_english:
            selected_voice_id = voice.id
            print("[TTS] Selected female English voice:", voice.name)
            break

    # 第二轮：如果没有英文女声，至少找英文 voice
    if not selected_voice_id:
        for voice in voices:
            name = (voice.name or "").lower()
            voice_id = (voice.id or "").lower()
            combined = name + " " + voice_id

            is_english = any(keyword in combined for keyword in english_keywords)

            if is_english:
                selected_voice_id = voice.id
                print("[TTS] Selected English voice:", voice.name)
                break

    if selected_voice_id:
        engine.setProperty("voice", selected_voice_id)
    else:
        print("[TTS] No English female voice found. Using default voice.")


def text_to_speech_file(text: str) -> Optional[str]:
    """
    使用 Windows 本地 SAPI 语音把文本保存成 wav 文件。
    返回 wav 文件路径，给 Gradio Audio 播放。

    清理策略：
    - 生成新音频前，先删除上一次生成的临时音频
    - 新生成的音频保留给 Gradio 页面播放
    - 下一次生成音频时再删除它
    - 程序退出时也尝试删除最后剩余的音频
    """
    global TEMP_AUDIO_FILES

    if not text or not text.strip():
        return None

    # 生成新音频前，删除旧的 TTS 文件，避免临时文件越积越多。
    cleanup_temp_audio_files()

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
    ).name

    try:
        engine = pyttsx3.init()

        # 语速可以调。数字越大越快。
        engine.setProperty("rate", 165)

        # 音量 0.0 ~ 1.0
        engine.setProperty("volume", 1.0)

        # 优先选择英文女声。
        select_female_english_voice(engine)

        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()

        TEMP_AUDIO_FILES.append(output_path)

        print("[TTS] Saved audio:", output_path)
        return output_path

    except Exception as e:
        print("[TTS] Error:", e)

        # 如果生成失败，删除失败文件，避免留下损坏 wav。
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
                print("[TTS] Deleted failed audio:", output_path)
        except Exception as delete_error:
            print("[TTS] Failed to delete failed audio:", delete_error)

        return None


def explain_word(target_word: str) -> Tuple[str, Optional[str]]:
    """
    单词解释按钮。
    不需要录音，直接讲解当前单词，并生成朗读音频。
    """
    if not target_word:
        return "请先选择或输入一个单词。", None

    reply = call_lm_studio(target_word, "", "explain")
    audio_reply = text_to_speech_file(reply)

    return reply, audio_reply


def voice_conversation(target_word: str, audio_path: Optional[str]):
    """
    语音对话按钮。
    先用 Whisper 识别语音，再让 LLM 继续对话，并朗读回答。

    返回 4 个值：
    1. Whisper 识别结果
    2. LLM 回答
    3. LLM 回答朗读音频
    4. 清空录音输入组件
    """
    print("[App] voice_conversation audio_path =", audio_path)

    if not audio_path:
        return (
            "没有收到录音文件。请先在麦克风控件里录音，停止录音后再点击按钮。",
            "",
            None,
            None,
        )

    transcript = transcribe_audio(audio_path)

    if not transcript:
        return (
            "Whisper 没有识别到语音。请说长一点、声音大一点，并确认麦克风权限正常。",
            "",
            None,
            None,
        )

    reply = call_lm_studio(target_word, transcript, "conversation")
    audio_reply = text_to_speech_file(reply)

    # 第 4 个返回值 None 会清空 gr.Audio 输入组件，方便下一次连续录音。
    return transcript, reply, audio_reply, None


def correct_sentence(target_word: str, audio_path: Optional[str]):
    """
    纠正我的造句按钮。
    先用 Whisper 识别语音，再让 LLM 纠正句子，并朗读回答。

    返回 4 个值：
    1. Whisper 识别结果
    2. LLM 回答
    3. LLM 回答朗读音频
    4. 清空录音输入组件
    """
    print("[App] correct_sentence audio_path =", audio_path)

    if not audio_path:
        return (
            "没有收到录音文件。请先在麦克风控件里录音，停止录音后再点击按钮。",
            "",
            None,
            None,
        )

    transcript = transcribe_audio(audio_path)

    if not transcript:
        return (
            "Whisper 没有识别到语音。请说长一点、声音大一点，并确认麦克风权限正常。",
            "",
            None,
            None,
        )

    reply = call_lm_studio(target_word, transcript, "correction")
    audio_reply = text_to_speech_file(reply)

    # 第 4 个返回值 None 会清空 gr.Audio 输入组件，方便下一次连续录音。
    return transcript, reply, audio_reply, None


def update_words_from_text(words_text: str):
    """
    从文本框更新单词下拉列表。
    支持换行、空格、逗号、中文逗号、顿号、分号。
    """
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


TEACHER_HTML = """
<div id="teacher-avatar" class="teacher-avatar teacher-panel">
  <div class="teacher-stage">
    <div class="teacher-hair-back"></div>
    <div class="teacher-neck"></div>
    <div class="teacher-body"></div>
    <div class="teacher-shirt"></div>

    <div class="teacher-head">
      <div class="teacher-ear left"></div>
      <div class="teacher-ear right"></div>
      <div class="teacher-face"></div>
      <div class="teacher-bang"></div>
      <div class="teacher-brow left"></div>
      <div class="teacher-brow right"></div>
      <div class="teacher-eye left"></div>
      <div class="teacher-eye right"></div>
      <div class="teacher-nose"></div>
      <div class="teacher-cheek left"></div>
      <div class="teacher-cheek right"></div>
      <div class="teacher-mouth"></div>
    </div>
  </div>
  <div class="teacher-speaking-indicator">
    <span id="teacher-indicator-text">Waiting</span>
  </div>
  <div class="teacher-caption">
    Your English teacher speaks when the audio is playing.
  </div>
</div>
"""


with gr.Blocks(
    title="LLM Voice Tutor",
    css=AVATAR_CSS,
    js=AVATAR_JS,
) as demo:
    gr.Markdown("# LLM Voice Tutor")
    gr.Markdown("Python + Gradio + faster-whisper + LM Studio 本地英语口语练习")

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
                label="单词解释",
                lines=10,
            )

            explain_audio_output = gr.Audio(
                label="单词解释朗读",
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
        inputs=[target_word],
        outputs=[explain_output, explain_audio_output],
    )

    gr.Markdown("## 语音练习")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="录音输入",
    )

    with gr.Row():
        chat_btn = gr.Button("和 LLM 对话")
        correction_btn = gr.Button("纠正我的造句")

    transcript_output = gr.Textbox(
        label="Whisper 识别结果",
        lines=4,
    )

    reply_output = gr.Textbox(
        label="LLM 回答",
        lines=10,
    )

    audio_reply_output = gr.Audio(
        label="LLM 回答朗读",
        type="filepath",
        autoplay=True,
    )

    chat_btn.click(
        fn=voice_conversation,
        inputs=[target_word, audio_input],
        outputs=[transcript_output, reply_output, audio_reply_output, audio_input],
    )

    correction_btn.click(
        fn=correct_sentence,
        inputs=[target_word, audio_input],
        outputs=[transcript_output, reply_output, audio_reply_output, audio_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
