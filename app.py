import atexit
import asyncio
import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import pyttsx3
import edge_tts
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

# OpenAI API Key 读取顺序：
# 1. 优先读取环境变量 OPENAI_API_KEY
# 2. 如果环境变量没有设置，则读取 app.py 同目录下的 openai_api_key.txt
#
# 注意：
# openai_api_key.txt 只放在本地，不要上传到 GitHub。
OPENAI_API_KEY_FILE = Path(__file__).parent / "openai_api_key.txt"


def load_openai_api_key() -> Optional[str]:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        print("[OpenAI] Loaded API key from environment variable.")
        return env_key.strip()

    if OPENAI_API_KEY_FILE.exists():
        key = OPENAI_API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            print("[OpenAI] Loaded API key from local file:", OPENAI_API_KEY_FILE)
            return key

    print("[OpenAI] No API key found. Use OPENAI_API_KEY or openai_api_key.txt.")
    return None


OPENAI_API_KEY = load_openai_api_key()
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

# Windows 上 edge-tts / aiohttp 有时会在 Proactor event loop 关闭连接时打印
# ConnectionResetError。切换到 SelectorEventLoop 通常更稳定。
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        print("[edge-tts] Failed to set WindowsSelectorEventLoopPolicy:", e)

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1/chat/completions")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5-1.5b-instruct-unsloth-bnb-thinker")


# ============================================================
# TTS settings
# ============================================================
# TTS 调用方式可以在页面上选择：
#   1. pyttsx3 本地朗读（默认）
#   2. edge-tts 在线微软 Neural Voice
#   3. OpenAI TTS API
#
# edge-tts:
#   pip install edge-tts
#
# OpenAI TTS:
#   使用同一个 OpenAI API Key
#   可以通过环境变量修改模型和 voice：
#     $env:OPENAI_TTS_MODEL="gpt-4o-mini-tts"
#     $env:OPENAI_TTS_VOICE="nova"
# ============================================================
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")

EDGE_TTS_EN_VOICE = os.getenv("EDGE_TTS_EN_VOICE", "en-US-JennyNeural")
EDGE_TTS_ZH_VOICE = os.getenv("EDGE_TTS_ZH_VOICE", "zh-CN-XiaoxiaoNeural")


# ============================================================
# Whisper settings
# ============================================================
# "small" 比 "base" 慢一点，但中英文识别更稳。
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


def check_teacher_avatar_file() -> None:
    """
    Confirm that teacher.gif and teacher_speaking.gif exist in the same folder as app.py.
    """
    base_dir = Path(__file__).parent

    idle_path = base_dir / "teacher.gif"
    speaking_path = base_dir / "teacher_speaking.gif"

    if idle_path.exists():
        print("[Avatar] Found local idle teacher image:", idle_path)
    else:
        print("[Avatar] WARNING: teacher.gif not found next to app.py.")
        print("[Avatar] Please put teacher.gif in the same folder as app.py.")

    if speaking_path.exists():
        print("[Avatar] Found local speaking teacher image:", speaking_path)
    else:
        print("[Avatar] WARNING: teacher_speaking.gif not found next to app.py.")
        print("[Avatar] Please put teacher_speaking.gif in the same folder as app.py.")


def get_teacher_idle_path() -> str:
    """
    Return local teacher.gif path.
    """
    path = Path(__file__).parent / "teacher.gif"
    return str(path) if path.exists() else ""


def get_teacher_speaking_path() -> str:
    """
    Return local teacher_speaking.gif path.
    """
    path = Path(__file__).parent / "teacher_speaking.gif"
    return str(path) if path.exists() else get_teacher_idle_path()


def set_teacher_idle():
    return get_teacher_idle_path()


def set_teacher_speaking():
    return get_teacher_speaking_path()


# ============================================================
# Avatar UI
# ============================================================
# 使用 Gradio 原生 Image + Audio 事件切换图片，避免自定义 JS 导致页面加载问题。


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
    """
    Gradio 前端有时会在回调结束后继续读取录音文件。
    如果这里立刻删除，浏览器可能出现 Failed to fetch / ERR_CONTENT_LENGTH_MISMATCH。
    所以这里只打印日志；真正的连续录音通过 gr.update(value=None) 清空组件来实现。
    """
    if not audio_path:
        return

    print("[Audio Input] Keep input audio temporarily for Gradio stability:", audio_path)


# ============================================================
# Text helpers
# ============================================================
def clean_markdown_stars(text: str) -> str:
    """
    Remove markdown bold/list asterisks from model output.
    """
    return (text or "").replace("*", "").strip()


def remove_phonetic_lines_for_tts(text: str) -> str:
    """
    For TTS only:
    Skip lines that are pure phonetic symbols, such as:
      [bɔːk]
      [əˈkaʊnt]

    Rule:
    If a line starts with "[" and ends with "]", do not read that line.
    The displayed explanation text is not changed.
    """
    if not text:
        return ""

    kept_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            print("[TTS] Skip phonetic line:", stripped)
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def remove_equal_lines_for_display_and_tts(text: str) -> str:
    """
    For batch word explanation display and TTS:
    Remove lines that contain "=".
    Example lines like "======" or "wreak = cause damage" will be removed.
    """
    if not text:
        return ""

    kept_lines = []

    for line in text.splitlines():
        if "=" in line:
            print("[Text Filter] Skip line containing '=':", line)
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def keep_only_chinese_english_for_tts(text: str) -> str:
    """
    For TTS only:
    Remove symbols and punctuation as much as possible.
    Keep only:
      - Chinese characters
      - English letters
      - numbers
      - whitespace

    This prevents TTS from reading symbols like quotes, brackets, slashes, colons, etc.
    """
    if not text:
        return ""

    chars = []

    for ch in text:
        is_chinese = "\u4e00" <= ch <= "\u9fff"
        is_english_or_number = ch.isascii() and ch.isalnum()
        is_space = ch.isspace()

        if is_chinese or is_english_or_number or is_space:
            chars.append(ch)
        else:
            # Replace punctuation/symbol with a space so words do not stick together.
            chars.append(" ")

    cleaned = "".join(chars)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    return cleaned.strip()


def prepare_explanation_text_for_tts(text: str) -> str:
    """
    Common cleanup before explanation TTS:
    1. Remove lines containing "="
    2. Remove pure phonetic lines like [bɔːk]
    3. Remove symbols/punctuation, keeping only Chinese and English text
    """
    text = remove_equal_lines_for_display_and_tts(text)
    text = remove_phonetic_lines_for_tts(text)
    text = keep_only_chinese_english_for_tts(text)
    return text


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
    新版单词解释提示词要求模型直接返回固定文本格式。
    如果有 raw_text，则直接显示 raw_text。
    注意：这里绝对不要删除音标行。音标行只在 TTS 朗读前过滤。
    """
    if data.get("raw_text"):
        return clean_markdown_stars(data.get("raw_text", ""))

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
    pyttsx3 用于分开朗读时：
    如果是新版 raw_text 格式，中文部分读完整说明，英文部分留空。
    edge-tts / OpenAI TTS API 会使用合并朗读。

    注意：
    - 页面显示用 format_explanation，不删除音标行。
    - 朗读用这里的文本，要删除纯音标行。
    """
    if data.get("raw_text"):
        tts_text = prepare_explanation_text_for_tts(data.get("raw_text", ""))
        return clean_markdown_stars(tts_text), ""

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
            # 不固定 language="en"，让 Whisper 自动识别英语 / 汉语。
            # 这样你可以用中文问问题，也可以用英语练习。
            initial_prompt="The speech may be English or Chinese.",
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
            "请把 key 写入 app.py 同目录下的 openai_api_key.txt，\n"
            "或者在 PowerShell 中这样启动：\n"
            '$env:OPENAI_API_KEY="sk-你的key"\n'
            "python app.py"
        )

    target_word = target_word.strip() if target_word else ""

    if mode == "conversation":
        system_prompt = """
You are a friendly English conversation partner and vocabulary tutor.

The user is a Chinese bilingual learner practicing English.
The user may speak English or Chinese.

Very important:
- Answer ONLY the current speech transcript.
- Do NOT reuse previous explanations, previous examples, or previous answers.
- The current transcript is the user's latest message.
- If the current transcript is Chinese, understand the meaning and help the user express it naturally in English.
- If the current transcript asks a new question, answer that new question directly.
- If the current transcript is an English sentence attempt, respond to that sentence.
- If the transcript says "this word", "the word", "new word", "这个单词", "这个词", "这个英语单词", or a similar-sounding wrong word, understand it as the target word.
- Keep your reply fresh and specific to the current transcript.
- Encourage the user to use the target word naturally.
- Keep your English reply short: 1-3 sentences.
- Add one short Chinese note if useful.
- Do not use markdown asterisks.

Preferred response style:
1. If the user asks in Chinese, answer briefly in Chinese first if needed.
2. Then give a natural English expression or reply.
3. Ask one simple English follow-up question when appropriate.
"""
        user_prompt = f"""
TARGET_WORD:
{target_word}

CURRENT_USER_TRANSCRIPT:
{transcript}

Instruction:
The transcript may be English or Chinese.
Reply to CURRENT_USER_TRANSCRIPT only.
If the user used Chinese, help convert their idea into natural spoken English.
Do not repeat a previous answer.
"""

    elif mode == "correction":
        system_prompt = """
You are a friendly English sentence correction and translation tutor.

The user is a Chinese bilingual learner practicing English.
The user may speak English or Chinese.

Very important:
- Work ONLY on the current speech transcript.
- Do NOT reuse previous corrections, previous examples, or previous answers.
- The current transcript is the user's latest sentence attempt or latest idea.
- If the current transcript is English, correct it naturally.
- If the current transcript is Chinese, translate the user's intended meaning into natural spoken English.
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
The transcript may be English or Chinese.
If it is Chinese, translate the intended meaning into natural spoken English.
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
            "usage_cn": "请把 key 写入 app.py 同目录下的 openai_api_key.txt，或设置 OPENAI_API_KEY 后重新启动。",
            "example_en": "",
            "example_cn": "",
            "practice_question_en": "",
            "practice_question_cn": "",
        }

    system_prompt = """
角色 (Role)
你是一位专业的英语教学专家（TESOL）与 TOEIC 词汇导师。你擅长通过“词根词缀分析”与“极简情景对话”帮助学习者深度掌握单词。

核心指令 (Core Instructions)
1. 多义词拆解: 若一个单词有多个核心词义，必须为每个词义生成独立的模块。
2. 记忆方法: 优先使用词根词缀法。若无明显词根词缀，则使用联想或谐音。
3. 对话约束:
   - 每个模块必须包含一个中文问答语境翻译。
   - 每个模块必须包含一个英语提问句。
   - 每个模块必须包含一个包含目标词的英语回答句。
   - 英语回答句必须 7 词以内。
4. 不要输出 JSON。
5. 不要输出 markdown。
6. 不要使用星号。
7. 请严格遵循以下顺序，严禁输出任何额外解释或开场白。音标行必须保留在输出中。

输出格式要求 (Output Format)
每个词义模块都必须严格按照下面格式输出：

[美式发音国际音标]
[单词], [单词], [单词]
[当前含义对应的词性], [中文核心意思]
记忆方法: [纯中文拆解/联想法]
相关词族: [中文分类，如：动词, 名词]
[对应的英语单词列表]
[下方问答对话的中文翻译内容]
[英语提问句？]
[包含目标词的英语回答句 (7词以内)]

示例 (Example)
输入: account
[əˈkaʊnt]
account, account, account
动词, 解释；说明（原因）
记忆方法: 前缀 ac- (加强) + count (计算)。对账目核算并说明理由，引申为“解释”。
相关词族: 名词, 形容词
account, accountable
你今天为什么迟到？/ 交通堵塞导致了我的延迟。
Why were you late today?
Bad traffic accounts for my delay.

[əˈkaʊnt]
account, account, account
名词, 账户；账目
记忆方法: count 本意是计数。在银行记下的数字，即为“账户”。
相关词族: 名词 (会计)
accountant
我该如何付款？/ 请记在我的账上。
How can I pay for this?
Please charge it to my account.
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

        return {
            "word": target_word,
            "raw_text": output_text,
        }

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

The user is a Chinese bilingual learner practicing English.
The user may speak English or Chinese.

Very important:
- Answer ONLY the current speech transcript.
- Do NOT reuse previous explanations, previous examples, or previous answers.
- The current transcript is the user's latest message.
- If the current transcript is Chinese, understand the meaning and help the user express it naturally in English.
- If the current transcript asks a new question, answer that new question directly.
- If the current transcript is an English sentence attempt, respond to that sentence.
- If the transcript says "this word", "the word", "new word", "这个单词", "这个词", "这个英语单词", or a similar-sounding wrong word, understand it as the target word.
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
The transcript may be English or Chinese.
Reply to CURRENT_USER_TRANSCRIPT only.
If the user used Chinese, help convert their idea into natural spoken English.
Do not repeat a previous answer.
"""

    elif mode == "correction":
        system_prompt = """
You are a friendly English sentence correction and translation tutor.

The user is a Chinese bilingual learner practicing English.
The user may speak English or Chinese.

Very important:
- Work ONLY on the current speech transcript.
- Do NOT reuse previous corrections, previous examples, or previous answers.
- The current transcript is the user's latest sentence attempt or latest idea.
- If the current transcript is English, correct it naturally.
- If the current transcript is Chinese, translate the user's intended meaning into natural spoken English.
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
The transcript may be English or Chinese.
If it is Chinese, translate the intended meaning into natural spoken English.
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
角色 (Role)
你是一位专业的英语教学专家（TESOL）与 TOEIC 词汇导师。你擅长通过“词根词缀分析”与“极简情景对话”帮助学习者深度掌握单词。

核心指令 (Core Instructions)
1. 多义词拆解: 若一个单词有多个核心词义，必须为每个词义生成独立的模块。
2. 记忆方法: 优先使用词根词缀法。若无明显词根词缀，则使用联想或谐音。
3. 对话约束:
   - 每个模块必须包含一个中文问答语境翻译。
   - 每个模块必须包含一个英语提问句。
   - 每个模块必须包含一个包含目标词的英语回答句。
   - 英语回答句必须 7 词以内。
4. 不要输出 JSON。
5. 不要输出 markdown。
6. 不要使用星号。
7. 请严格遵循以下顺序，严禁输出任何额外解释或开场白。

输出格式要求 (Output Format)
每个词义模块都必须严格按照下面格式输出：

[美式发音国际音标]
[单词], [单词], [单词]
[当前含义对应的词性], [中文核心意思]
记忆方法: [纯中文拆解/联想法]
相关词族: [中文分类，如：动词, 名词]
[对应的英语单词列表]
[下方问答对话的中文翻译内容]
[英语提问句？]
[包含目标词的英语回答句 (7词以内)]

示例 (Example)
输入: account
[əˈkaʊnt]
account, account, account
动词, 解释；说明（原因）
记忆方法: 前缀 ac- (加强) + count (计算)。对账目核算并说明理由，引申为“解释”。
相关词族: 名词, 形容词
account, accountable
你今天为什么迟到？/ 交通堵塞导致了我的延迟。
Why were you late today?
Bad traffic accounts for my delay.

[əˈkaʊnt]
account, account, account
名词, 账户；账目
记忆方法: count 本意是计数。在银行记下的数字，即为“账户”。
相关词族: 名词 (会计)
accountant
我该如何付款？/ 请记在我的账上。
How can I pay for this?
Please charge it to my account.
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

        return {
            "word": target_word,
            "raw_text": output_text,
        }

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


def contains_cjk(text: str) -> bool:
    """
    判断文本里是否包含中文字符。
    edge-tts 如果用英文 voice 读中文，会读得很差或不读。
    所以只要包含中文，就优先使用中文 Neural Voice。
    """
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


async def edge_tts_save_async(text: str, output_path: str, voice: str) -> None:
    """
    edge-tts async wrapper.
    """
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)


def edge_tts_file(text: str, language: str = "en", cleanup_before: bool = True) -> Optional[str]:
    """
    使用 edge-tts 生成 mp3。
    注意：edge-tts 需要联网，但生成的是完整 mp3 文件，之后在本地播放。
    """
    global TEMP_AUDIO_FILES

    text = clean_markdown_stars(text)

    if not text:
        return None

    if cleanup_before:
        cleanup_temp_audio_files()

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ).name

    # 关键修正：
    # 即使调用方传入 language="en"，只要 LLM 回复中包含中文，
    # edge-tts 就使用中文 Neural Voice，否则英文 voice 会把中文读得很差或直接不读。
    if language == "zh" or contains_cjk(text):
        voice = EDGE_TTS_ZH_VOICE
        print("[edge-tts] Chinese text detected. Use Chinese voice.")
    else:
        voice = EDGE_TTS_EN_VOICE

    try:
        print("[edge-tts] Voice:", voice)
        asyncio.run(edge_tts_save_async(text, output_path, voice))

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            raise RuntimeError("edge-tts generated audio is empty or too small")

        TEMP_AUDIO_FILES.append(output_path)
        print("[edge-tts] Saved audio:", output_path)
        return output_path

    except Exception as e:
        print("[edge-tts] Error:", e)

        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

        print("[edge-tts] Falling back to pyttsx3.")
        if cleanup_before:
            return text_to_speech_file(text, language=language)
        return create_tts_file(text, language=language)

def openai_tts_file(text: str, language: str = "en", cleanup_before: bool = True) -> Optional[str]:
    """
    使用 OpenAI TTS API 生成 mp3。
    使用同一个 OPENAI_API_KEY / openai_api_key.txt。
    """
    global TEMP_AUDIO_FILES

    text = clean_markdown_stars(text)

    if not text:
        return None

    if client is None:
        print("[OpenAI TTS] No OpenAI API Key. Falling back to pyttsx3.")
        if cleanup_before:
            return text_to_speech_file(text, language=language)
        return create_tts_file(text, language=language)

    if cleanup_before:
        cleanup_temp_audio_files()

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ).name

    try:
        print("[OpenAI TTS] Model:", OPENAI_TTS_MODEL)
        print("[OpenAI TTS] Voice:", OPENAI_TTS_VOICE)

        # 推荐方式：完整生成音频文件后再播放，避免 buffer / 丢开头。
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            raise RuntimeError("OpenAI TTS generated audio is empty or too small")

        TEMP_AUDIO_FILES.append(output_path)
        print("[OpenAI TTS] Saved audio:", output_path)
        return output_path

    except Exception as e:
        print("[OpenAI TTS] Error:", e)

        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

        print("[OpenAI TTS] Falling back to pyttsx3.")
        if cleanup_before:
            return text_to_speech_file(text, language=language)
        return create_tts_file(text, language=language)

def synthesize_tts_file(tts_provider: str, text: str, language: str = "en", cleanup_before: bool = True) -> Optional[str]:
    """
    页面选择的 TTS 调用入口。

    tts_provider:
      - pyttsx3
      - edge-tts
      - OpenAI TTS API
    """
    if tts_provider == "edge-tts":
        return edge_tts_file(text, language=language, cleanup_before=cleanup_before)

    if tts_provider == "OpenAI TTS API":
        return openai_tts_file(text, language=language, cleanup_before=cleanup_before)

    if cleanup_before:
        return text_to_speech_file(text, language=language)

    return create_tts_file(text, language=language)


# ============================================================
# Conversation history helpers
# ============================================================
def append_conversation_history(
    history_text: str,
    mode_label: str,
    target_word: str,
    transcript: str,
    reply: str,
) -> str:
    """
    Append one LLM interaction to page-visible text history.
    This history is only kept in the current Gradio session.
    Every app restart starts with empty history.
    """
    history_text = history_text or ""

    item = (
        f"---\n"
        f"模式：{mode_label}\n"
        f"练习单词：{target_word or ''}\n"
        f"我说的话 / 识别结果：\n{transcript or ''}\n\n"
        f"LLM 回答：\n{reply or ''}\n"
    )

    if history_text.strip():
        return history_text.rstrip() + "\n\n" + item

    return item


def clear_conversation_history():
    """
    Clear the page-visible conversation history.
    """
    return "", ""


def export_conversation_history(history_text: str) -> Optional[str]:
    """
    Export conversation history to a local txt file and return the filepath for Gradio download.
    """
    history_text = history_text or ""

    if not history_text.strip():
        history_text = "当前还没有对话历史。"

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt",
        mode="w",
        encoding="utf-8",
    ).name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(history_text)

    print("[History] Exported conversation history:", output_path)
    return output_path


# ============================================================
# UI CSS
# ============================================================
CUSTOM_CSS = """
/* Make important text areas easier to read */
textarea {
  font-size: 18px !important;
  line-height: 1.65 !important;
}

label, .wrap label {
  font-size: 15px !important;
}

/* Make markdown output text a little larger */
.prose, .markdown {
  font-size: 16px !important;
}

/* Keep audio components compact */
audio {
  width: 100% !important;
}
"""


def parse_words_from_text(words_text: str):
    """
    Parse up to 10 words from the words textbox.
    Supports newline, comma, Chinese comma, Japanese comma, semicolon, and spaces.
    """
    if not words_text:
        return DEFAULT_WORDS[:10]

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
        return DEFAULT_WORDS[:10]

    return words[:10]


def export_text_to_file(text: str, suffix: str = ".txt") -> Optional[str]:
    """
    Export text to a temporary file for Gradio download.
    """
    text = text or ""

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        mode="w",
        encoding="utf-8",
    ).name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("[Export] Saved text file:", output_path)
    return output_path


def explain_all_words(
    llm_provider: str,
    tts_provider: str,
    words_text: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], str]:
    """
    Generate explanations for up to 10 words.
    Also generate one combined TTS audio file and downloadable txt/audio files.
    """
    words = parse_words_from_text(words_text)

    if not words:
        empty_msg = "没有找到可解释的单词。请先输入或更新单词列表。"
        return empty_msg, None, None, None, get_teacher_idle_path()

    print("[Batch Explain] Words:", words)

    sections = []

    for index, word in enumerate(words, start=1):
        word = word.strip()
        if not word:
            continue

        print(f"[Batch Explain] Explaining {index}/{len(words)}:", word)

        data = call_llm_explain_json(llm_provider, word)
        explanation_text = format_explanation(data)

        # 全部单词解释中：过滤所有包含 "=" 的行，显示和朗读都不保留。
        explanation_text = remove_equal_lines_for_display_and_tts(explanation_text)

        section = (
            f"{index}. {word}\n"
            f"{explanation_text}"
        )
        sections.append(section)

    if not sections:
        empty_msg = "没有生成任何单词解释。"
        return empty_msg, None, None, None, get_teacher_idle_path()

    all_text = "\n\n".join(sections).strip()

    # Export text for download.
    text_file = export_text_to_file(all_text, suffix=".txt")

    # For TTS, remove "=" lines, phonetic lines, and symbols.
    tts_text = prepare_explanation_text_for_tts(all_text)

    # Generate one combined audio for all explanations.
    # For edge-tts / OpenAI TTS API, this is the preferred path.
    # For pyttsx3, it still generates one audio, but mixed CN/EN may be less natural.
    audio_file = synthesize_tts_file(
        tts_provider,
        tts_text,
        language="zh" if contains_cjk(tts_text) else "en",
        cleanup_before=True,
    )

    return all_text, audio_file, text_file, audio_file, get_teacher_speaking_path()

# ============================================================
# Gradio event functions
# ============================================================
def make_explanation_combined_tts_text(data: Dict[str, str]) -> str:
    """
    edge-tts / OpenAI TTS API 用：
    中英混合时不再拆成两个音频，而是一次生成完整朗读音频。

    注意：
    - 页面显示用 format_explanation，不删除音标行。
    - 朗读用这里的文本，要删除纯音标行。
    """
    if data.get("raw_text"):
        tts_text = prepare_explanation_text_for_tts(data.get("raw_text", ""))
        return clean_markdown_stars(tts_text)

    lines = [
        f"单词：{data.get('word', '')}",
        f"中文意思：{data.get('meaning_cn', '')}",
        f"词性：{data.get('part_of_speech_cn', '')}",
        f"常见用法：{data.get('usage_cn', '')}",
        f"英文例句：{data.get('example_en', '')}",
        f"中文翻译：{data.get('example_cn', '')}",
        f"练习问题：{data.get('practice_question_en', '')}",
        f"{data.get('practice_question_cn', '')}",
    ]
    return clean_markdown_stars("。".join([line for line in lines if line.strip()]))


def explain_word(llm_provider: str, tts_provider: str, target_word: str) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Word explanation:
    - pyttsx3：中文解释、英文例句分开生成两个音频
    - edge-tts / OpenAI TTS API：中英混合内容合并成一个音频
    """
    if not target_word:
        return "请先选择或输入一个单词。", None, None, get_teacher_idle_path()

    data = call_llm_explain_json(llm_provider, target_word)
    display_text = format_explanation(data)

    cleanup_temp_audio_files()

    if tts_provider == "pyttsx3":
        # pyttsx3 对中英混合自动切换 voice 不好，所以仍然分开朗读。
        zh_tts_text, en_tts_text = make_explanation_tts_texts(data)
        zh_audio = synthesize_tts_file(tts_provider, zh_tts_text, language="zh", cleanup_before=False)
        en_audio = synthesize_tts_file(tts_provider, en_tts_text, language="en", cleanup_before=False)
        # UI 中第一个音频自动播放，第二个音频不自动播放，避免中英文同时播放。
        return display_text, zh_audio, en_audio, get_teacher_speaking_path()

    # edge-tts / OpenAI TTS API：
    # 使用一个完整音频朗读中英混合内容，不再拆成中文和英文两个音频。
    combined_tts_text = make_explanation_combined_tts_text(data)
    lang = "zh" if contains_cjk(combined_tts_text) else "en"
    combined_audio = synthesize_tts_file(
        tts_provider,
        combined_tts_text,
        language=lang,
        cleanup_before=False,
    )

    return display_text, combined_audio, None, get_teacher_speaking_path()


def voice_conversation(llm_provider: str, tts_provider: str, target_word: str, audio_path: Optional[str], history_text: str):
    print("[App] voice_conversation audio_path =", audio_path)

    if not audio_path:
        return (
            "没有收到录音文件。请先在麦克风控件里录音，停止录音后再点击按钮。",
            "",
            None,
            gr.update(value=None),
            get_teacher_idle_path(),
            history_text or "",
            history_text or "",
        )

    transcript = transcribe_audio(audio_path)

    if not transcript:
        return (
            "Whisper 没有识别到语音。请说长一点、声音大一点，并确认麦克风权限正常。",
            "",
            None,
            gr.update(value=None),
            get_teacher_idle_path(),
            history_text or "",
            history_text or "",
        )

    print("[App] Current conversation transcript sent to OpenAI:", transcript)
    reply = call_llm_text(llm_provider, target_word, transcript, "conversation")
    audio_reply = synthesize_tts_file(tts_provider, reply, language="en")
    new_history = append_conversation_history(
        history_text,
        "和 LLM 对话",
        target_word,
        transcript,
        reply,
    )

    return transcript, reply, audio_reply, gr.update(value=None), get_teacher_speaking_path(), new_history, new_history



def voice_conversation_on_stop(
    llm_provider: str,
    tts_provider: str,
    target_word: str,
    audio_path: Optional[str],
    history_text: str,
):
    """
    Triggered automatically when the microphone recording is stopped.
    This merges the recording Stop action and the old "和 LLM 对话" button.
    """
    return voice_conversation(
        llm_provider,
        tts_provider,
        target_word,
        audio_path,
        history_text,
    )


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



with gr.Blocks(title="LLM Voice Tutor") as demo:
    gr.Markdown("# LLM Voice Tutor")
    gr.Markdown("Python + Gradio + faster-whisper + OpenAI API / Local LM Studio + 多种 TTS 本地英语口语练习（支持中文 / 英文语音输入，老师形象使用本地 GIF）")

    llm_provider = gr.Radio(
        choices=["OpenAI API", "Local LM Studio"],
        value="OpenAI API",
        label="LLM 调用方式",
        info="默认使用 OpenAI API；请准备 openai_api_key.txt 或 OPENAI_API_KEY。需要本地模型时可切换到 Local LM Studio。",
    )

    tts_provider = gr.Radio(
        choices=["pyttsx3", "edge-tts", "OpenAI TTS API"],
        value="edge-tts",
        label="TTS 朗读方式",
        info="默认使用 edge-tts；edge-tts 和 OpenAI TTS API 需要联网。需要完全本地离线朗读时可切换到 pyttsx3。",
    )


    gr.HTML(
        '<a href="https://platform.openai.com/usage" target="_blank" '
        'style="font-size:14px; text-decoration: underline;">API Usage</a>'
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

    with gr.Accordion("TTS 连接信息（只读，修改请用环境变量）", open=False):
        gr.Markdown(
            f"""
当前 OpenAI TTS 模型：`{OPENAI_TTS_MODEL}`

当前 OpenAI TTS Voice：`{OPENAI_TTS_VOICE}`

当前 edge-tts 英文 Voice：`{EDGE_TTS_EN_VOICE}`

当前 edge-tts 中文 Voice：`{EDGE_TTS_ZH_VOICE}`

注意：edge-tts 遇到中英混合文本时，会优先使用中文 Voice，以保证中文可以正常朗读。

如需修改，在 PowerShell 启动前设置：

`$env:OPENAI_TTS_MODEL="gpt-4o-mini-tts"`

`$env:OPENAI_TTS_VOICE="nova"`

`$env:EDGE_TTS_EN_VOICE="en-US-JennyNeural"`

`$env:EDGE_TTS_ZH_VOICE="zh-CN-XiaoxiaoNeural"`
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
            explain_all_btn = gr.Button("全部10个单词解释")

        with gr.Column(scale=1):
            teacher_image = gr.Image(
                value=get_teacher_idle_path(),
                label="Emily · Your English Teacher",
                type="filepath",
                height=420,
                interactive=False,
            )

        with gr.Column(scale=2):
            explain_output = gr.Textbox(
                label="单词解释（TESOL/TOEIC格式）",
                lines=10,
            )

            with gr.Row():
                explain_audio_zh_output = gr.Audio(
                    label="单词解释朗读（edge-tts / OpenAI TTS 合并朗读；pyttsx3 中文解释）",
                    type="filepath",
                    autoplay=True,
                )

                explain_audio_en_output = gr.Audio(
                    label="英文例句朗读（仅 pyttsx3 分开朗读时使用；需手动播放）",
                    type="filepath",
                    autoplay=False,
                )

            gr.Markdown("### 全部单词解释")

            explain_all_output = gr.Textbox(
                label="全部10个单词解释（可导出）",
                lines=18,
                interactive=False,
            )

            explain_all_audio_output = gr.Audio(
                label="全部单词解释朗读",
                type="filepath",
                autoplay=False,
            )

            with gr.Row():
                explain_all_text_file = gr.File(
                    label="下载全部单词解释 TXT",
                    visible=True,
                )
                explain_all_audio_file = gr.File(
                    label="下载全部单词解释音频",
                    visible=True,
                )

    update_words_btn.click(
        fn=update_words_from_text,
        inputs=[words_text],
        outputs=[target_word],
    )

    explain_btn.click(
        fn=explain_word,
        inputs=[llm_provider, tts_provider, target_word],
        outputs=[explain_output, explain_audio_zh_output, explain_audio_en_output, teacher_image],
    )

    explain_all_btn.click(
        fn=explain_all_words,
        inputs=[llm_provider, tts_provider, words_text],
        outputs=[
            explain_all_output,
            explain_all_audio_output,
            explain_all_text_file,
            explain_all_audio_file,
            teacher_image,
        ],
    )


    gr.Markdown("## 语音练习")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="录音输入（可说英语 / 中文）",
    )

    gr.Markdown("录音后点击录音控件里的停止按钮，会自动发送给 LLM 对话。")

    transcript_output = gr.Textbox(
        label="Whisper 识别结果（自动识别中/英）",
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

    gr.Markdown("## 对话历史记录")

    conversation_history_state = gr.State("")

    conversation_history_output = gr.Textbox(
        label="对话历史记录（本次启动期间保留，可手动清空，可导出）",
        value="",
        lines=14,
        interactive=False,
    )

    with gr.Row():
        clear_history_btn = gr.Button("清空历史记录")
        export_history_btn = gr.Button("导出历史记录")

    history_file_output = gr.File(
        label="下载导出的历史记录",
        visible=True,
    )

    clear_history_btn.click(
        fn=clear_conversation_history,
        inputs=None,
        outputs=[conversation_history_state, conversation_history_output],
    )

    export_history_btn.click(
        fn=export_conversation_history,
        inputs=[conversation_history_state],
        outputs=[history_file_output],
    )


    # Gradio 原生 Audio 事件：播放时显示 teacher_speaking.gif，暂停/停止时显示 teacher.gif。
    for _audio in [explain_audio_zh_output, explain_audio_en_output, explain_all_audio_output, audio_reply_output]:
        try:
            _audio.play(fn=set_teacher_speaking, inputs=None, outputs=teacher_image)
            _audio.pause(fn=set_teacher_idle, inputs=None, outputs=teacher_image)
            _audio.stop(fn=set_teacher_idle, inputs=None, outputs=teacher_image)
        except Exception as e:
            print("[Avatar] Audio event binding skipped:", e)

    # 录音停止时自动执行“和 LLM 对话”
    # Gradio 版本差异较大：优先使用 stop_recording；如果没有，则尝试 change 事件。
    _conversation_outputs = [
        transcript_output,
        reply_output,
        audio_reply_output,
        audio_input,
        teacher_image,
        conversation_history_state,
        conversation_history_output,
    ]

    try:
        audio_input.stop_recording(
            fn=voice_conversation_on_stop,
            inputs=[llm_provider, tts_provider, target_word, audio_input, conversation_history_state],
            outputs=_conversation_outputs,
        )
    except Exception as e:
        print("[UI] audio_input.stop_recording binding failed, fallback to change:", e)
        try:
            audio_input.change(
                fn=voice_conversation_on_stop,
                inputs=[llm_provider, tts_provider, target_word, audio_input, conversation_history_state],
                outputs=_conversation_outputs,
            )
        except Exception as change_error:
            print("[UI] audio_input.change binding also failed:", change_error)


if __name__ == "__main__":
    check_teacher_avatar_file()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[str(Path(__file__).parent)],
        css=CUSTOM_CSS,
    )
