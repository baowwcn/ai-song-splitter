# 歌曲分割器 · ai-song-splitter

按**歌声断点**把一首歌切成 **8–15 秒**(可调)的片段,本地 Flask Web 应用,界面全中文。
支持人声分离(BSRoformer / demucs)、**歌词驱动分段**(LRC 或纯文本)、在线试听、微调切点、单段/整包下载。
音频全程在本机处理,不上传任何服务器。

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-green) ![Models](https://img.shields.io/badge/models-BSRoformer%20%2F%20Qwen3-ASR%20%2F%20FunASR-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [模型下载与放置(重点)](#模型下载与放置重点)
- [离线 / 便携部署](#离线--便携部署)
- [项目结构](#项目结构)
- [使用说明](#使用说明)
- [歌词分段算法规则](#歌词分段算法规则)
- [HTTP 接口](#http-接口)
- [环境变量](#环境变量)
- [常见问题与坑](#常见问题与坑)
- [许可](#许可)

---

## 功能特性

| 能力 | 说明 |
|---|---|
| 4 种检测模式 | 相对能量断点、BSRoformer 人声分离、demucs 人声分离、**歌词驱动分段**(默认) |
| 歌词切点 | 支持 LRC(带时间戳)与**纯文本歌词**(自动识别人声 → 逐字对齐 → 推出每句起点) |
| 3 套识别/对齐引擎 | Qwen3-ASR(默认,中英通用)、FunASR(中文最准)、whisper large-v3 / medium |
| 2 套人声分离 | BSRoformer(`pymss`,默认,失败自动回退 demucs)、demucs 4.x 兜底 |
| 智能吸附 | 切点向左吸到 250ms 内能量最安静的 50ms 窗口,避免削掉首字音 |
| 手动微调 | 页面拖动切点后 `/re-cut` 重切,边界吸附回最近的有效断点 |
| 歌词导出 | 歌词模式切完自动生成 `歌词_时间戳.txt`(LRC 风格,UTF-8-sig,Excel 直接打开) |
| 打包下载 | 单段下载 / 整首 zip(片段 + 歌词) |
| 自包含 | 代码里没有任何硬编码盘符,模型全部落在项目内 `runtime\models\`,整目录拷走即用 |

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11(代码里 `ffmpeg.exe`、模型目录都按 Windows 写;其他平台需自行适配) |
| 架构 | **amd64 (x86_64)**。ARM64 机器(如骁龙 X)会走 Windows 模拟层,能用但慢;装依赖务必选 amd64 |
| Python | 3.11 ~ 3.13(开发时用 3.13.12) |
| ffmpeg | 必需,pydub 读 mp3/flac 靠它 |
| 显卡 | 可选。有 NVIDIA 卡(≥8GB 显存)强烈建议,分离快 10 倍以上;无卡可跑 CPU,一首 4 分钟的歌分离要几分钟 |
| 显存 | BSRoformer 峰值约 **10 GB**;Qwen3 双模型(1.7B+0.6B,bf16)约 **6 GB** |
| 磁盘 | 只装代码 < 1 MB;按需下模型 275 MB ~ 13 GB(见下表);切好的片段每首几十 MB |
| 网络 | 首次需要联网下模型;之后可完全离线 |

---

## 快速开始

> 仓库里**不含** Python 运行时和模型(体积近 20 GB),下面 5 步自己装。

### 1. 克隆

```powershell
git clone https://github.com/baowwcn/ai-song-splitter.git
cd ai-song-splitter
```

或直接下载 ZIP 解压。

### 2. 装 Python 依赖(虚拟环境)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell 若被禁执行:Set-ExecutionPolicy -Scope Process Bypass

pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # 有 NVIDIA 显卡:cu126 版 torch
pip install -r requirements.txt
```

- **CPU / 无 NVIDIA 显卡**:跳过 torch 那行,`pip install -r requirements.txt` 会装 CPU 版(torch 2.14)。
- **不要装 torchaudio**:torch 2.14 没有对应的 cu126 wheel,会直接装不上;FunASR 用 `kaldi-native-fbank` 代替。
- **不要设置 `OMP_NUM_THREADS`**:torch 2.14 在 Windows 上会死锁。
- `requirements.txt` 里除 torch 外都是 PyPI 正常包:`flask pydub audioop-lts numpy demucs pymss funasr kaldi-native-fbank openai-whisper opencc-python-reimplemented`。

### 3. 装 ffmpeg

二选一:

**A. 放进项目内(推荐,自动识别,不用改 PATH)**

把 ffmpeg 解压成 `tools\ffmpeg\<版本>\bin\ffmpeg.exe`,例如:

```
tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe
```

`app.py` 启动时会自动在 `tools/ffmpeg/*/bin` 里找并注入 PATH(只认项目内,不翻上级目录,保证整目录拷走就能用)。
下载:https://www.gyan.dev/ffmpeg/builds/ 或 https://github.com/BtbN/FFmpeg-Builds

**B. 装到系统里并加入 PATH**(例如 `winget install Gyan.FFmpeg`),项目在 PATH 里能找到就行。

### 4. 下模型

代码不自带任何权重,按[下一节](#模型下载与放置重点)的表下载并放到指定目录。
最省事的组合:**BSRoformer(自动下)+ FunASR(自动下)+ ffmpeg**,就能跑「相对能量 / 人声分离 / 歌词分段」全部功能;
想用界面默认的 Qwen3 引擎,再手动下 5.7 GB 的两个 Qwen3 模型。

### 5. 启动

```powershell
.venv\Scripts\python.exe app.py
```

或直接双击 `start.bat`(有 `runtime\python\python.exe` 就用便携运行时,否则用 PATH 里的 python,会自动打开浏览器)。
浏览器访问 <http://127.0.0.1:8000>。

- 端口被占用:`Get-Process python | Stop-Process` 清掉旧服务,或改 `app.py` 末尾 `app.run(...)` 的 `port`。
- 任务状态存在内存里,**重启服务即丢**;已切好的片段还在 `uploads\<job_id>\` 里。

---

## 模型下载与放置(重点)

所有模型的**查找路径都写死在项目内**,不依赖系统环境变量(`app.py` 顶部把 `HF_HOME` / `TORCH_HOME` 指向 `runtime\models\`)。

| 用途 | 模型 | 何时需要 | 体积 | 获取方式 | 放置目录 |
|---|---|---|---|---|---|
| 人声分离(默认引擎) | `bs_roformer_voc_hyperacev2` | 选「BSRoformer 人声分离」或歌词模式 | 275 MB | **自动下载**(pymss 首次运行从 ModelScope 拉) | `runtime\models\pymss\vocal\vocal_extraction\*.ckpt` + `*.yaml` |
| 人声分离(备用) | `HTDemucs` | BSRoformer 失败自动回退 | 80 MB | **自动下载**(demucs 4.x 走 HF hub) | `runtime\models\huggingface\hub\models--adefossez--HTDemucs\` |
| 歌词识别+对齐(界面默认) | `Qwen3-ASR-1.7B`<br>`Qwen3-ForcedAligner-0.6B` | 歌词模式 + 引擎选 Qwen3 | 3.9 GB + 1.8 GB | **手动下载**(代码用 `local_files_only=True`,不会自动下) | `runtime\models\qwen\Qwen3-ASR\`<br>`runtime\models\qwen\Qwen3-ForcedAligner\` |
| 歌词识别+对齐(中文) | `FunAudioLLM/Fun-ASR-Nano-2512`<br>`funasr/paraformer-zh`<br>`funasr/fsmn-vad` | 歌词模式 + 引擎选 FunASR | 合计约 2.8 GB | **自动下载**(funasr `hub="hf"`,首次运行触发) | `runtime\models\huggingface\hub\` |
| 识别/对齐(英文兜底) | `large-v3.pt` / `medium.pt` | 引擎选 whisper | 2.9 GB / 1.46 GB | 首次使用自动下载,或用 `tools\download_whisper.py` 预下 | `runtime\models\whisper\<model>.pt` |

### Qwen3(界面默认引擎,必须手动下)

仓库/文件来自 `Qwen/Qwen3-ASR-1.7B` 与 `Qwen/Qwen3-ForcedAligner-0.6B`(Apache-2.0)。

```powershell
pip install -U "huggingface_hub[cli]"

# 走 Hugging Face
huggingface-cli download Qwen/Qwen3-ASR-1.7B            --local-dir runtime\models\qwen\Qwen3-ASR
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B  --local-dir runtime\models\qwen\Qwen3-ForcedAligner

# 国内走 ModelScope(更稳)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B           --local_dir runtime\models\qwen\Qwen3-ASR
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir runtime\models\qwen\Qwen3-ForcedAligner
```

> 新版 CLI 命令为 `hf download <repo> --local-dir ...`;两个名字都有,哪个能用用哪个。
> 放好后必须能看到 `runtime\models\qwen\Qwen3-ASR\model.safetensors`。要放别处就设环境变量
> `QWEN3_ASR_DIR` / `QWEN3_ALIGNER_DIR`。
> ⚠️ 别用 `config.wrappers.json.bak` 覆盖 `config.json`——那是 transformers 4.57 的旧包装格式,会加载失败。

### FunASR(中文,自动下载;想预下就手动拉)

```powershell
$env:HF_HOME = "$PWD\runtime\models\huggingface"     # 与 app.py 保持一致,否则会下到别处
huggingface-cli download funasr/paraformer-zh funasr/fsmn-vad FunAudioLLM/Fun-ASR-Nano-2512
```

不预下也行:歌词模式 + FunASR 引擎第一次跑会自动下载这三个模型(约 2.8 GB)。
注意 FunASR **只支持中文**;界面「歌曲语言」选英文时会自动改走 whisper large-v3。

### BSRoformer 人声分离(自动下载;想预下/换源就手动)

```powershell
# 预下载到项目内模型目录(默认 ModelScope 源)
.venv\Scripts\python.exe -m pymss download bs_roformer_voc_hyperacev2 --model-dir runtime\models\pymss

# 换 Hugging Face / hf-mirror 源
.venv\Scripts\python.exe -m pymss download bs_roformer_voc_hyperacev2 --model-dir runtime\models\pymss --source huggingface
.venv\Scripts\python.exe -m pymss download bs_roformer_voc_hyperacev2 --model-dir runtime\models\pymss --source hf-mirror
```

- 完整目录:`runtime\models\pymss\vocal\vocal_extraction\bs_roformer_voc_hyperacev2.ckpt` + 同名 `.yaml`。
- 想换别的 BSRoformer 模型,设 `BSROFORMER_MODEL=<模型名>`(该模型要在 pymss 的模型库里)。
- 页面右上角「模型未就绪」= `runtime\models\pymss` 下没有 `*.ckpt`。

### whisper(可选兜底)

```powershell
.venv\Scripts\python.exe tools\download_whisper.py large-v3 medium
# 走镜像(断点续传,断线重跑同一命令即可)
.venv\Scripts\python.exe tools\download_whisper.py large-v3 --base-url https://你的镜像/whisper
```

也可以设环境变量 `WHISPER_DOWNLOAD_BASE` 长期走镜像。
⚠️ 不要用 HF 上的 `openai/whisper-medium`:那是 transformers(safetensors)格式,`openai-whisper` 库读不了。
不预下时,首次选该引擎会自己从 OpenAI 官方 CDN 下载(曾限速 ~50 KiB/s,建议预下)。

### 全部预下完成后自检

页面右上角的模型状态点会变绿;也可以直接调接口看:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model-status
```

---

## 离线 / 便携部署

代码无任何外部目录依赖,目标机器装好后:

1. 把 `runtime\models\` 整个目录拷到目标机(约 13 GB);
2. `tools\ffmpeg\<版本>\bin\` 一起拷(或目标机自己装 ffmpeg);
3. 目标机没有 Python 时,把便携 Python 一起拷进来:

```
runtime\python\python.exe        ← 嵌入式 Python 3.13.12 amd64(含全部依赖)
runtime\python\Lib\site-packages\
runtime\python\python313._pth
```

4. 双击 `start.bat` 即可(它会优先用 `runtime\python\python.exe`)。

详见 [MIGRATE.md](MIGRATE.md)。

---

## 项目结构

```
ai-song-splitter/
├── app.py                 Flask 服务与全部路由(上传/分割/轮询/试听/重切/下载)
├── splitter.py            切割核心:静音断点检测、切点吸附、贪心/歌词优先分段、人声分离
├── lyrics.py              歌词切点:LRC 解析、纯文本对齐、三套识别/对齐引擎
├── templates/index.html   单页界面(全中文)
├── static/                静态资源(头像等)
├── tools/
│   ├── download_whisper.py  whisper 权重预下载(断点续传)
│   └── ffmpeg/              ffmpeg(自行放置,<版本>/bin 结构)
├── runtime/               ← 不入库:便携 Python + 全部模型
│   ├── python/
│   └── models/
│       ├── pymss/         BSRoformer 权重
│       ├── qwen/          Qwen3-ASR + Qwen3-ForcedAligner
│       ├── huggingface/   FunASR / Paraformer / fsmn-vad / HTDemucs 缓存
│       ├── whisper/       large-v3.pt / medium.pt
│       └── torch/         demucs 旧 checkpoint 目录(可空)
├── uploads/               ← 不入库:上传的原文件与切好的片段
├── requirements.txt
├── start.bat              一键启动
├── MIGRATE.md             便携版迁移说明
└── AGENTS.md              项目内部规则/踩坑记录(给 AI 助手看)
```

上传后的文件按时间戳命名:`uploads\<job_id>.<ext>` 是原文件,`uploads\<job_id>\segment_001.wav …` 是切好的片段,
`job_id` = 上传时刻 `%Y%m%d_%H%M%S`,按文件名即可排出先后。

---

## 使用说明

1. **上传**歌曲(mp3 / wav / flac / m4a 等 ffmpeg 能解的格式,单文件 ≤200 MB)。
2. **选检测模式**:
   - `按歌词分段`(默认,效果最好):填歌词。填 LRC 就直接按时间戳切;只填纯文本会先识别人声再逐字对齐。
   - `BSRoformer 人声分离`:先分离人声,再在纯人声轨上找静音断点,最通用。
   - `demucs 人声分离`:同上,备用引擎。
   - `相对能量断点`:不分离,直接分析原曲,最快,伴奏重的歌容易切错。
3. **高级参数**(都有默认值,一般不用动):

   | 参数 | 默认 | 含义 |
   |---|---|---|
   | 每段最短 / 最长 | 8s / 15s | 目标片段时长 |
   | 相对阈值 | 10 dB | 相对能量模式的断点灵敏度,越小切得越碎 |
   | 静音阈值 | -40 dB | 绝对静音阈值(人声轨用) |
   | 最短静音 | 500 ms | 短于此长度的停顿不算断点 |
   | 间奏阈值 | 3.0 s | 静音超过此长度判为间奏,间奏独立成段 |
   | 歌曲语言 | 中文 | 中文 / 英文,影响 whisper 与 Qwen3,英文时 FunASR 自动换成 whisper |

4. **切完可以**在线试听每段、拖动切点后点「重新切分」(`/re-cut`)、单段下载或整包 zip 下载。
5. 歌词模式还会输出 `歌词_时间戳.txt`(LRC 风格,`[mm:ss.xx] 歌词`,间奏段空行),可在页面上单独下载,也会打进 zip。

---

## 歌词分段算法规则

(改动分割逻辑前请先看这里,这些是明确约定,不是实现细节)

1. 先按**标点断句**(中英文标点全覆盖),每处分句都是一个候选切点;纯文本对齐后取每句起点,LRC 行内按字符比例分配时间。
2. 短于 8s 的段合并到 8–15s;**间奏**(静音 ≥ 间奏阈值,默认 3s)是完整独立段,**不与歌唱段合并**;间奏过长时拆成约 7–15s 的段。
   收尾修正(末段过短时左移上一刀凑长)优先落在歌词标点切点上,允许上一段略短于 8s,没有标点才按时间收。
3. 合并例外(后处理,按最终相邻段判断):间奏 < 7s **且**相邻歌唱段 < 8s 时才允许合并。
4. 切点严格递增,相邻段之间强制留 0.05s 以上,多刀不得落在同一时刻。

---

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页界面 |
| POST | `/upload` | 上传歌曲,返回 `job_id` |
| POST | `/split` | 提交分割任务(参数同上表),后台线程执行 |
| GET | `/status/<job_id>` | 轮询进度与分段结果 |
| POST | `/re-cut` | 按手动调整后的边界重新切 |
| POST | `/extract-lyrics` | 只识别歌词,不切分 |
| GET | `/model-status` | 检查人声分离模型是否就绪 |
| GET | `/audio/<job_id>/<idx>` | 在线试听某一段 |
| GET | `/download/<job_id>/<idx>` | 下载某一段 wav |
| GET | `/download/<job_id>/all` | 打包 zip(片段 + 歌词) |
| GET | `/download/<job_id>/lyrics` | 下载 `歌词_时间戳.txt` |
| POST | `/open-dir` | 打开输出目录 |

---

## 环境变量

模型查找全部相对项目目录,下面这些只是覆盖手段,一般不用设:

| 变量 | 作用 |
|---|---|
| `BSROFORMER_MODEL` | 换 BSRoformer 模型名(默认 `bs_roformer_voc_hyperacev2`) |
| `QWEN3_ASR_DIR` / `QWEN3_ALIGNER_DIR` | Qwen3 两个模型的自定义目录 |
| `WHISPER_DOWNLOAD_BASE` | whisper 下载镜像前缀 |

`HF_HOME` / `TORCH_HOME` 由 `app.py` 强制设为 `runtime\models\huggingface`、`runtime\models\torch`,手动预下模型时要跟它保持一致。

---

## 常见问题与坑

| 现象 | 原因 / 解决 |
|---|---|
| `WinError 2`(pydub 找不到 ffmpeg) | ffmpeg 没放对。只认 `tools\ffmpeg\<版本>\bin\ffmpeg.exe`,或装系统版进 PATH |
| 卡死在 demucs / 进程无响应 | 设置了 `OMP_NUM_THREADS`,torch 2.14 在 Windows 会死锁,删掉该变量 |
| `pip` 装 torchaudio 失败 | torch 2.14 没有 cu126 的 torchaudio wheel,别装 |
| 页面提示「模型未就绪」 | `runtime\models\pymss` 下没有 `*.ckpt`,先跑一次分离让它自动下载 |
| 提示「未找到 Qwen3-ASR 模型目录」 | 目录结构不对,应有 `Qwen3-ASR\model.safetensors`;或设 `QWEN3_ASR_DIR` |
| 歌词切点全挤到歌尾 | 分离输出不是 16bit int16 WAV。`_separate_bsroformer` 里 `audio_params={"wav_bit_depth": "PCM_16"}` 必须保留 |
| 切点削掉歌曲第一个字 | 已内置 `snap_cuts_to_gaps`(向左吸到 250ms 内最安静的 50ms),若仍削首字可调大「最短静音」 |
| 改了 HTML 但页面没变 | 需 `Ctrl+F5`;Jinja 会缓存模板,`app.py` 已开自动重载,仍不行就重启服务 |
| PowerShell `curl -d` 报 400 | 引号被 bash 风格吞掉,用 `Invoke-RestMethod -ContentType "application/json"` |
| 端口 8000 被占用 | `Get-Process python \| Stop-Process` |
| 杀毒软件误报 | 便携/嵌入式 Python 首次运行常见误报,加信任即可 |
| 2080 Ti(sm75)跑分离有告警 | 无 cuDNN MHA 加速会回退慢路径,结果正常,只是慢 |
| 任务重启后没了 | 状态在内存 `JOBS` 里,重启即丢;切好的片段还在 `uploads\<job_id>\` |

---

## 相关链接

- pymss(BSRoformer 封装,模型库 `baicai1145/pymss`):https://github.com/baicai1145/pymss
- Qwen3-ASR:https://github.com/QwenLM/Qwen3-ASR
- FunASR:https://github.com/modelscope/FunASR
- openai/whisper:https://github.com/openai/whisper
- demucs:https://github.com/adefossez/demucs

---

## 许可

本项目代码以 [MIT](LICENSE) 许可发布(Copyright (c) 2026 baowwcn),可自由商用、修改、再分发。

依赖与模型各自遵循原始许可,本仓库不分发任何权重或第三方二进制:

| 组件 | 许可 |
|---|---|
| PyTorch / Flask / pydub | BSD-3-Clause |
| demucs / openai-whisper / FunASR | MIT |
| pymss | MIT |
| Qwen3-ASR / Qwen3-ForcedAligner | Apache-2.0 |
| BSRoformer(`bs_roformer_voc_hyperacev2`) | 以模型库 `baicai1145/pymss` 页面标注为准 |
| ffmpeg | 依所下构建而定(essentials build 为 LGPL/GPL),请自行遵守其许可 |
