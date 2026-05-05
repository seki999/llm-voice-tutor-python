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


with gr.Blocks(title="LLM Voice Tutor") as demo:
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
