# 迁移指南(便携版)

整个项目**完全自包含、无任何外部目录依赖**:拷贝到任何 Windows 电脑(无需安装
Python / ffmpeg / 无需联网下模型)即可运行。

## 迁移步骤

### 1. 复制整个目录

```
song-splitter\  ← 整个目录复制到新电脑(约 19.7 GB)
```

> 可选瘦身(不影响功能):
> - `uploads\` 运行时生成,可清空(本机约 1.3 GB)
> - `runtime\models\whisper\medium.pt` 1.46 GB(UI 选 medium 时才需要)
> - `runtime\models\torch\hub\checkpoints\*.th` 80 MB,demucs 3.x 旧格式遗留,当前版本不再读取
> - `runtime\models\whisper\large-v3.pt` 2.9 GB,英文歌与 qwen3 不可用时的兜底,建议留

### 2. 双击 start.bat

浏览器自动打开 http://127.0.0.1:8000,直接使用。

## 项目自包含内容

| 组件 | 位置 | 大小 | 说明 |
|---|---|---|---|
| Python 运行时 | `runtime\python\` | 4.9 GB | 嵌入式 Python 3.13.12(amd64),含全部依赖(torch 2.14.0+cu126) |
| ffmpeg | `tools\ffmpeg\ffmpeg-9.0.2-essentials_build\` | 314 MB | 只用 `bin\` 即可 |
| BSRoformer 人声分离 | `runtime\models\pymss\vocal\vocal_extraction\` | 275 MB | `bs_roformer_voc_hyperacev2`(默认分离引擎) |
| Qwen3-ASR + ForcedAligner | `runtime\models\qwen\` | 5.7 GB | 界面默认识别引擎(最准,中英都行) |
| FunASR(Nano + Paraformer) | `runtime\models\huggingface\hub\` | 2.8 GB | 中文识别/对齐引擎 |
| demucs(备用分离) | `runtime\models\huggingface\hub\models--adefossez--HTDemucs\` | 80 MB | BSRoformer 失败时自动回退 |
| whisper | `runtime\models\whisper\` | 4.4 GB | `large-v3.pt` 2.9 GB + `medium.pt` 1.46 GB |
| 应用代码 | `app.py` / `splitter.py` / `lyrics.py` / `templates\` / `static\` | — | 界面全中文单页 |

模型查找全部相对项目目录(`app.py` 顶部设 `HF_HOME` / `TORCH_HOME`,
`pymss` 用 `model_dir` 指向 `runtime\models\pymss`,Qwen3 用 `lyrics._QWEN_DIR`),
**不含任何硬编码盘符**。唯一可覆盖项是环境变量(一般不用设):
`BSROFORMER_MODEL`、`QWEN3_ASR_DIR`、`QWEN3_ALIGNER_DIR`、`WHISPER_DOWNLOAD_BASE`。

## 注意事项

- **架构**:运行时为 amd64(x86_64),通过 Windows on ARM 模拟层也可运行(本机即 ARM64 系统)
- **端口**:默认 8000,被占用时先 `Get-Process python | Stop-Process` 清掉旧服务,
  或改 `app.py` 末尾 `app.run(...)` 的 port
- **不要设置 `OMP_NUM_THREADS`**:torch 2.14 Windows 已知死锁问题
- **不要装 cu126 源的 torchaudio**:2.14 无对应 wheel,会装不上(FunASR 用 `kaldi-native-fbank`)
- **显存**:BSRoformer 分离峰值约 10 GB;Qwen3 双模型(1.7B + 0.6B,bf16)约 6 GB
- **首次启动稍慢**:嵌入式 Python 首次加载依赖较慢,属正常

## 常见问题

- **WinError 2**:确认 `tools\ffmpeg\<版本>\bin\ffmpeg.exe` 存在(结构不能变;
  `app.py` 只在本目录内查找,不翻上级目录)
- **提示「未找到 Qwen3-ASR 模型目录」**:确认 `runtime\models\qwen\Qwen3-ASR\model.safetensors`
  存在;要换位置就设 `QWEN3_ASR_DIR` / `QWEN3_ALIGNER_DIR`
- **界面「模型未就绪」**:`/model-status` 检查的是 `runtime\models\pymss` 下有无 `*.ckpt`
- **杀毒软件拦截**:嵌入式 Python 首次运行可能被误报,添加信任即可
