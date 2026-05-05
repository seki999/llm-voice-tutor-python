import os
import requests
import gradio as gr
from faster_whisper import WhisperModel

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen2.5-1.5b-instruct-unsloth-bnb-thinker"

# Windows CPU 推荐先用 base 或 small
# base 更快，small 更准
whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
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


def transcribe_audio(audio_path: str) -> str:
    if not audio_path:
        return ""

    segments, info = whisper_model.transcribe(
        audio_path,
        language="en",
        vad_filter=True,
        beam_size=5,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def call_lm_studio(target_word: str, transcript: str, mode: str) -> str:
    if mode == "conversation":
        system_prompt = f"""
You are a friendly English speaking tutor.

The user is practicing this target word: {target_word}

The user speaks by voice, and the transcript may contain speech-recognition errors.
If the transcript says "this word", "the word", "new word", or a similar-sounding wrong word, understand it as the target word.

Your task:
1. Understand the user's meaning.
2. Continue the conversation naturally.
3. Help the user use the target word.
4. Keep the reply short: 1-3 sentences.
5. Give a short Chinese note if useful.

Return plain text, not JSON.
"""
        user_prompt = f"""
Target word: {target_word}
User transcript: {transcript}
"""

    elif mode == "explain":
        system_prompt = """
You are a friendly English vocabulary tutor.
Explain the target word to a Chinese learner.

Include:
- Chinese meaning
- part of speech
- usage
- one short English example
- Chinese translation of the example

Return concise Chinese explanation with the English example.
"""
        user_prompt = f"Target word: {target_word}"

    else:
        system_prompt = f"""
You are a friendly English sentence correction tutor.

The user is practicing this target word: {target_word}

Correct the user's sentence naturally.
Explain briefly in Chinese.
Then ask one simple follow-up question in English using the target word.
"""
        user_prompt = f"""
Target word: {target_word}
User sentence transcript: {transcript}
"""

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
            "max_tokens": 180,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def explain_word(target_word: str) -> str:
    return call_lm_studio(target_word, "", "explain")


def voice_conversation(target_word: str, audio_path: str):
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return "没有识别到语音。", ""

    reply = call_lm_studio(target_word, transcript, "conversation")
    return transcript, reply


def correct_sentence(target_word: str, audio_path: str):
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return "没有识别到语音。", ""

    reply = call_lm_studio(target_word, transcript, "correction")
    return transcript, reply


with gr.Blocks(title="LLM Voice Tutor") as demo:
    gr.Markdown("# LLM Voice Tutor")
    gr.Markdown("本地英语口语练习：Gradio + faster-whisper + LM Studio")

    with gr.Row():
        target_word = gr.Dropdown(
            choices=DEFAULT_WORDS,
            value=DEFAULT_WORDS[0],
            label="当前练习单词",
            allow_custom_value=True,
        )

    with gr.Row():
        explain_btn = gr.Button("单词解释")
        explain_output = gr.Textbox(label="单词解释", lines=8)

    explain_btn.click(
        fn=explain_word,
        inputs=[target_word],
        outputs=[explain_output],
    )

    gr.Markdown("## 语音对话")
    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="录音输入",
    )

    with gr.Row():
        chat_btn = gr.Button("和 LLM 对话")
        correction_btn = gr.Button("纠正我的造句")

    transcript_output = gr.Textbox(label="Whisper 识别结果", lines=3)
    reply_output = gr.Textbox(label="LLM 回答", lines=8)

    chat_btn.click(
        fn=voice_conversation,
        inputs=[target_word, audio_input],
        outputs=[transcript_output, reply_output],
    )

    correction_btn.click(
        fn=correct_sentence,
        inputs=[target_word, audio_input],
        outputs=[transcript_output, reply_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )