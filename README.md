# LLM Voice Tutor

本项目是一个本地英语口语练习工具，使用：

- Gradio 作为 Web UI
- faster-whisper 做本地语音识别
- OpenAI API 或本地 LM Studio 作为 LLM
- pyttsx3 做本地语音朗读
- 浏览器页面中的 HTML/CSS/JS 做英语老师头像和嘴型动画

支持：

- 英语 / 中文 / 日语语音输入
- Whisper 自动识别语音语言
- 单词解释
- 中文解释朗读
- 英文例句朗读
- 和 LLM 对话
- 纠正我的造句
- OpenAI API / Local LM Studio 两种 LLM 调用方式切换
- 录音输入文件自动删除
- TTS 临时音频文件自动清理

---

## 1. 项目文件

建议目录结构：

```text
llm-voice-tutor-python/
├─ app.py
├─ README.md
└─ .venv/
```

如果你下载到的新环境里文件名是：

```text
app_llm_provider_select.py
```

可以改名为：

```text
app.py
```

---

## 2. 安装 Python

建议使用：

```text
Python 3.10 / 3.11 / 3.12
```

Windows 可以从这里安装：

```text
https://www.python.org/downloads/
```

安装时建议勾选：

```text
Add python.exe to PATH
```

确认 Python 是否安装成功：

```powershell
python --version
```

---

## 3. 创建虚拟环境

在项目目录打开 PowerShell：

```powershell
cd C:\Users\sekine\Documents\llm-voice-tutor-python
python -m venv .venv
.\.venv\Scripts\activate
```

激活成功后，命令行前面会显示：

```text
(.venv)
```

---

## 4. 安装依赖

执行：

```powershell
pip install gradio faster-whisper pyttsx3 openai requests
```

也可以创建 `requirements.txt`：

```txt
gradio
faster-whisper
pyttsx3
openai
requests
```

然后执行：

```powershell
pip install -r requirements.txt
```

---

## 5. 关于 faster-whisper 模型下载

第一次运行时，程序会自动从 Hugging Face 下载 Whisper 模型。

当前代码默认使用：

```python
WhisperModel("small", device="cpu", compute_type="int8")
```

`small` 的中英文识别比较稳，但第一次下载会慢一些。

如果下载太慢，或者电脑运行太慢，可以把代码里的：

```python
WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)
```

改成：

```python
WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
)
```

对比：

```text
base  = 更快，中文识别一般
small = 稍慢，中英日识别更稳
```

---

## 6. Hugging Face Warning 说明

第一次运行时可能看到：

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

这不是错误。意思是没有登录 Hugging Face，下载速度或次数可能有限制。

通常可以忽略。

也可能看到：

```text
cache-system uses symlinks by default but your machine does not support them
```

这也不是错误，只是 Windows 没启用符号链接，缓存可能多占一点空间。

如果想隐藏这个 warning：

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
python app.py
```

---

# 7. LLM 调用方式

页面顶部有一个选项：

```text
LLM 调用方式
○ OpenAI API
○ Local LM Studio
```

默认是：

```text
OpenAI API
```

---

# 8. 方式一：使用外部 LLM，OpenAI API

## 8.1 准备 OpenAI API Key

进入 OpenAI Platform 创建 API Key：

```text
https://platform.openai.com/api-keys
```

建议：

```text
不要把 API Key 写进代码
不要提交到 GitHub
不要写进前端 JS
只放在环境变量里
```

---

## 8.2 PowerShell 临时设置 API Key

推荐每次启动时设置临时环境变量：

```powershell
$env:OPENAI_API_KEY="sk-你的key"
python app.py
```

这个方式的特点：

```text
不写入磁盘
关闭当前 PowerShell 窗口后失效
比较安全
```

---

## 8.3 设置 OpenAI 模型

默认模型是：

```text
gpt-5.4-mini
```

如果想指定模型：

```powershell
$env:OPENAI_MODEL="gpt-5.4-mini"
$env:OPENAI_API_KEY="sk-你的key"
python app.py
```

---

## 8.4 OpenAI 费用安全建议

推荐设置：

```text
Auto recharge：关闭
账户里只放少量 credits，例如 $5
设置预算提醒
```

这样即使程序异常循环调用，也不会无限扣费。

---

# 9. 方式二：使用本地 LLM，LM Studio

## 9.1 安装并启动 LM Studio

下载 LM Studio：

```text
https://lmstudio.ai/
```

启动后：

1. 下载一个本地模型
2. 打开 Local Server
3. 启动 OpenAI-compatible API Server

默认 API 地址通常是：

```text
http://localhost:1234/v1/chat/completions
```

---

## 9.2 设置本地模型名

当前代码默认本地模型名是：

```text
qwen2.5-1.5b-instruct-unsloth-bnb-thinker
```

如果你的 LM Studio 里模型名不同，需要设置环境变量。

示例：

```powershell
$env:LOCAL_LLM_MODEL="你的本地模型名"
python app.py
```

如果 API 地址不同：

```powershell
$env:LOCAL_LLM_URL="http://localhost:1234/v1/chat/completions"
$env:LOCAL_LLM_MODEL="你的本地模型名"
python app.py
```

---

## 9.3 同时准备 OpenAI 和本地 LLM

可以同时设置：

```powershell
$env:OPENAI_API_KEY="sk-你的key"
$env:OPENAI_MODEL="gpt-5.4-mini"
$env:LOCAL_LLM_URL="http://localhost:1234/v1/chat/completions"
$env:LOCAL_LLM_MODEL="你的本地模型名"
python app.py
```

然后在页面里用单选按钮切换：

```text
OpenAI API
Local LM Studio
```

---

# 10. 启动项目

在项目目录里：

```powershell
.\.venv\Scripts\activate
python app.py
```

启动成功后会显示类似：

```text
Running on local URL: http://127.0.0.1:7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

---

# 11. 手机访问方式

如果想让同一个 Wi-Fi 下的手机访问，需要把代码最后的：

```python
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
)
```

改成：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
)
```

然后在 Windows 上查 IP：

```powershell
ipconfig
```

找到类似：

```text
IPv4 Address . . . . . . . . . . : 192.168.1.23
```

手机浏览器打开：

```text
http://192.168.1.23:7860
```

如果打不开，通常是 Windows 防火墙挡住了 7860 端口。

---

# 12. 常见问题

## 12.1 Whisper 没有识别到语音

请确认：

```text
1. 浏览器允许麦克风权限
2. 录音后点击停止
3. 再点击「和 LLM 对话」或「纠正我的造句」
4. 说话时间不要太短，建议 3 秒以上
```

可以查看 VS Code Terminal：

```text
[Whisper] Final text:
```

如果这里为空，说明 Whisper 没识别出来。

---

## 12.2 新一轮说话后，LLM 还是旧回答

请查看 Terminal：

```text
[Whisper] Final text:
[App] Current conversation transcript sent to OpenAI:
```

如果显示的是旧内容，说明录音输入没有刷新。

当前代码已经用：

```python
gr.update(value=None)
```

在处理完成后清空录音输入。

---

## 12.3 中文输入能识别吗？

可以。当前代码不再固定：

```python
language="en"
```

而是让 Whisper 自动识别：

```text
英语 / 汉语 / 日语
```

Terminal 里会显示：

```text
[Whisper] Detected language: zh
[Whisper] Detected language: en
[Whisper] Detected language: ja
```

---

## 12.4 中文朗读没有声音或读得很奇怪

这通常是 Windows 没安装中文语音。

可以在 Windows 设置里安装：

```text
设置 → 时间和语言 → 语音 → 添加语音
```

添加中文语音后重启程序。

---

## 12.5 英文朗读不是女声

代码会优先寻找英文女声，例如：

```text
Zira
Aria
Jenny
Sonia
Hazel
Samantha
```

如果 Windows 没有这些 voice，会退回系统默认 voice。

可以在 Windows 设置里添加英文语音。

---

# 13. 推荐启动命令

## 13.1 OpenAI API 模式

```powershell
cd C:\Users\sekine\Documents\llm-voice-tutor-python
.\.venv\Scripts\activate
$env:OPENAI_API_KEY="sk-你的key"
$env:OPENAI_MODEL="gpt-5.4-mini"
python app.py
```

## 13.2 本地 LM Studio 模式

```powershell
cd C:\Users\sekine\Documents\llm-voice-tutor-python
.\.venv\Scripts\activate
$env:LOCAL_LLM_URL="http://localhost:1234/v1/chat/completions"
$env:LOCAL_LLM_MODEL="你的本地模型名"
python app.py
```

然后页面里选择：

```text
Local LM Studio
```

## 13.3 同时支持 OpenAI 和本地 LM Studio

```powershell
cd C:\Users\sekine\Documents\llm-voice-tutor-python
.\.venv\Scripts\activate

$env:OPENAI_API_KEY="sk-你的key"
$env:OPENAI_MODEL="gpt-5.4-mini"

$env:LOCAL_LLM_URL="http://localhost:1234/v1/chat/completions"
$env:LOCAL_LLM_MODEL="你的本地模型名"

python app.py
```

---

# 14. 安全注意事项

API Key 安全原则：

```text
不要写进 app.py
不要写进 HTML / JS
不要上传 GitHub
不要截图发给别人
推荐只用 PowerShell 临时环境变量
```

OpenAI 费用安全：

```text
关闭 Auto recharge
只放少量 credits，例如 $5
设置预算提醒
定期查看 Usage
```

---

# 15. 当前项目能力总结

当前版本支持：

```text
本地 Whisper 语音识别
OpenAI API / Local LM Studio 可切换
中文 / 英文 / 日文语音输入
中文解释朗读
英文例句朗读
英文对话朗读
录音文件自动删除
TTS 临时文件自动清理
老师头像嘴型动画
```
