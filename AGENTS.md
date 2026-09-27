# AGENTS.md

按歌声断点切割歌曲的本地 Flask 应用,每段默认 8–15s(可调),支持 rel/BSRoformer(人声分离)/歌词(LRC 或纯文本)三种模式。UI 全中文,单页 `templates/index.html`。**无测试、无 lint、无 git**——验证方式就是启动服务后手工跑一遍。

## 运行

```powershell
# 不要用系统 Python!依赖只装在便携运行时里
E:\ai\song-splitter\runtime\python\python.exe E:\ai\song-splitter\app.py
# 或双击 start.bat(自动注入 ffmpeg PATH + 开浏览器),浏览器访问 http://127.0.0.1:8000
```

- 安装/升级依赖必须用 `runtime\python\python.exe -m pip install …`(GPU torch 走 `--index-url https://download.pytorch.org/whl/cu126`,见 requirements.txt 注释)。
- whisper 模型预下载/续传:`tools\download_whisper.py <模型>`(官方 azure CDN,断线重跑续传);走国内镜像加 `--base-url https://镜像/whisper` 或在环境变量设 `WHISPER_DOWNLOAD_BASE`,代码里对应 `lyrics._WHISPER_URLS` 与 `_fetch_whisper_model`。注意 HF/hf-mirror 上的 `openai/whisper-medium` 是 safetensors(transformers)格式,`openai-whisper` 库不能直接用,别配这种镜像。
- 端口冲突时先 `Get-Process python | Stop-Process`——旧服务没停干净会导致新服务起不来。
- 迁移/便携说明见 `MIGRATE.md`。

## 架构与入口

- `app.py`:Flask 服务与全部路由(`/upload` `/split` `/status/<job_id>` `/audio` `/download` `/download/<job_id>/lyrics` `/re-cut` `/extract-lyrics` `/model-status`)。任务状态存内存 `JOBS` 字典,**重启即丢**。分割走 daemon 后台线程 + 前端轮询,勿改同步。**job_id = 上传时刻 `%Y%m%d_%H%M%S`**(`new_job_id()`,同一秒重复上传追加 `-2`),原文件和切好的片段都在 `uploads\<job_id>.*` 与 `uploads\<job_id>\`,按文件名即可排出先后;2026-09-26 已把旧的 32 位随机 hex 目录/文件全部按 mtime 重命名成时间戳。
- `splitter.py`:切割核心。`detect_silence_rel`(相对能量,抗伴奏)与 `detect_silence_abs`(绝对阈值,用于人声轨)、`cut_points_from_silences`、`snap_cuts_to_gaps`(把切点向左吸到 250ms 内能量最安静的 50ms 窗口削首字音)、`segment_with_bounds`(贪心)、`segment_lyrics_priority`(歌词模式)、`separate_vocals`(**BSRoformer(pymss)优先,失败回退 `_separate_demucs`**;`engine` 参数:`auto` 默认=BSRoformer 优先+demucs 兜底、`bsroformer` 仅 BSRoformer、`demucs` 仅旧 demucs)、`torch_device`、`split_song` 主入口(mode 支持 `rel`/`demucs`/`bsroformer`/`lyrics`)。**歌词模式(`mode=lyrics`)分割成功后会在 `out_dir` 写出 `歌词_时间戳.txt`**(UTF-8-sig,LRC 风格 `[mm:ss.xx] 该段歌词`,每段起始时间,含 `[length:xx.xx]` 头行;间奏段空文本),`app.py` 读它给你 `/download/<job_id>/lyrics` 下载与打包 zip,纯文本/LRC 两种输入都会生成。
- `lyrics.py`:歌词切点。`parse_lrc/lrc_cut_points`(LRC 直接按戳切)、`align_lines_to_audio`(纯文本 → 分离人声 → 所选引擎逐字时间戳 → DP 文字对齐)、`extract_lyrics`(识别提取歌词)。**三引擎可选**(model 参数):`qwen3` = UI 默认 = Qwen3-ASR-1.7B 提取 + Qwen3-ForcedAligner-0.6B 逐字对齐(transformers 5.17 原生,中英都行,最准);`funasr` = Fun-ASR-Nano 提取 + Paraformer-zh 原生逐字时间戳对齐(中文准);`large-v3`/`medium` = OpenAI whisper(段内按字符占比线性插值 + 0.2/0.4 双温度按语音覆盖选优,文本须 opencc 繁体→简体归一)。`language` 参数(`zh`/`en`)对 whisper 与 qwen3 生效;UI「歌曲语言」选英文时,**funasr 自动改走 whisper large-v3**(Nano/Paraformer 仅中文)。引擎都缓存于进程内模块级字典,按 (engine, device) 复用。

## 歌词模式算法规则(用户明确指定,勿改)

1. 先按**标点断句**(`_PUNCT`,中英文标点全覆盖),每处分句都是切点;纯文本对齐输出每句起点,LRC 行内按字符比例分配时间。
2. <8s 短段合并到 8–15s;间奏(静音 ≥ interlude_min_s,默认 3s)是**完整独立段,不与歌唱段合并**;间奏 > max 平均拆成 ~(7,15]s 段。收尾修正(末段 < min 时左移上一刀凑长):**优先落歌词标点切点,允许上一段略短于 min**,无标点才按时间收。
3. 合并例外(后处理,按最终相邻段判断):间奏 <7s 且相邻歌唱段 <8s 才允许合并。
4. 切点须严格递增,段间强制 prev+0.05s 以上,多刀不得叠同一时刻。

## 环境陷阱(踩过的坑)

- **切勿设置 `OMP_NUM_THREADS`**:torch 2.14 在 Windows 上会令 demucs 死锁。
- embeddable Python 的 `._pth` 限制 sys.path,`app.py` 顶部已手动 `sys.path.insert(0, BASE)`——新增顶层模块要放项目根目录。
- ffmpeg 在 `tools\ffmpeg\<版本>\bin`,由 `app.py` 启动时自动查找注入 PATH(pydub 找不到会报 WinError 2)。结构不能变。**只在本项目目录内查找**(`_find_ffmpeg_bin` 只 glob `BASE/tools/ffmpeg/*/bin`,早期版本还会翻上级目录,已去掉,保证整目录拷走即用、不会误用别的 ffmpeg)。
- 运行时为 **amd64**(本机是 ARM64 系统走模拟层);装 Python/依赖一律 amd64,ARM64 装不上 win_amd64 wheel。
- demucs 4.x 模型走 HuggingFace hub,`HF_HOME`/`TORCH_HOME` 已指到项目内 `runtime\models\huggingface\` / `runtime\models\torch\`(离线可用)。`runtime\models\torch\hub\checkpoints\*.th` 是旧格式遗留,已不被使用。
- **BSRoformer(pymss)**:默认模型 `bs_roformer_voc_hyperacev2`(288MB,vocals/instrumental 双轨),模型缓存在项目内 `runtime\models\pymss\`(结构 `vocal\vocal_extraction\*.ckpt|.yaml`),模型名可用环境变量 `BSROFORMER_MODEL` 覆盖(须在 pymss 目录内)。pymss 默认下载源 ModelScope,`model_dir` 指向项目内目录实现离线可用;`pymss` 依赖 `av`/`librosa`/`pymss-core`,不依赖 torchaudio。**`_separate_bsroformer` 显式传 `audio_params={"wav_bit_depth": "PCM_16"}`,强制输出 16bit int16 WAV——pymss 默认 FLOAT(fmt3/bits32)会让 `lyrics._qwen3_audio16k`、`snap_cuts_to_gaps` 等按 int16 裸读 raw_data 解码成噪声,ASR/对齐直接塌缩(切点全挤到歌尾),这是踩过的坑,别去掉**。GPU 实测:44s 歌分离约 24-38s,峰值显存 ~10GB(2080 Ti 上 sm75 无 cuDNN MHA 加速会回退慢路径并告警,属正常)。UI 检测模式有「BSRoformer 人声分离(默认)」与「demucs 人声分离(备用)」两项:前者 mode=`bsroformer`(固定 engine=`auto`),后者 mode=`demucs`(固定 engine=`demucs`);歌词模式在「高级参数 → 分离引擎」可选 `auto`(默认,BSRoformer 优先+demucs 兜底)或 `demucs`,纯文本对齐与 LRC 间奏检测都复用该 engine。
- `separate_vocals` 找人声轨:BSRoformer 输出 `<basename>_vocals.wav`(用 `endswith("_vocals")` 兜底匹配),demucs 输出 `vocals.wav`(须精确 `os.path.splitext(f)[0] == "vocals"`,不能用 `"vocals" in f`,会误把 `no_vocals.wav` 当人声)。
- `/model-status` 检查的就是 pymss 目录下有无 `*.ckpt`(不再是旧 torch hub 的 `.th`)。
- `runtime\models\whisper\large-v3.pt` 由用户从 `D:\download\large-v3.pt` 备份恢复;medium 无本地备份,若在 UI 选 medium 且本地无文件会触发在线下载(openai CDN 曾限速 ~50KiB/s)。whisper 对齐**不要用 word_timestamps**(Windows 无 Triton 走慢速 DTW 回退),现为段内按字符占比线性插值 + 0.2/0.4 双温度按语音覆盖选优,识别文本须经 opencc 繁体→简体归一。whisper 对无参考歌词场景易幻觉循环复读,中文场景默认用 funasr。
- FunASR 引擎依赖:`funasr` + `kaldi-native-fbank`(fbank 后端;**切勿装 cu126 源的 torchaudio**,2.14 无对应 wheel 会装不上)。Paraformer-zh / fsmn-vad / Fun-ASR-Nano-2512 模型经 hub="hf" 缓存在 `HF_HOME`(`runtime\models\huggingface\hub\models--funasr--paraformer-zh`、`models--funasr--fsmn-vad`、`models--FunAudioLLM--Fun-ASR-Nano-2512`)。
- **Paraformer 逐字时间戳的参数名是 `pred_timestamp`(不是 `predict_timestamp`)**——funasr 1.4.16 的模型源码按 `pred_timestamp` 读取,传错名字会静默无时间戳。输出 `text`(空格分隔字)与 `timestamp`([[start_ms,end_ms],…])一一对应,逐字毫秒级,直接替换 whisper 的段内插值。
- Fun-ASR-Nano 需 `trust_remote_code=True`(模型仓库自带 model.py)且有 VAD 配合(`vad_model="fsmn-vad"` + `max_single_segment_time=30000`)避免长歌超 context;输出文本自带标点断句,提取歌词后直接按句末标点分行。
- FunASR 对 GPU 直呼 `cuda` 即可,但对 emulation 环境的 `device` 统一归一为 `"cuda:0"`(见 `lyrics._dev_str`)。
- **Qwen3 引擎(UI 默认)**:模型已随项目携带在 `runtime\models\qwen\Qwen3-ASR`(1.7B,3.9GB)与 `runtime\models\qwen\Qwen3-ForcedAligner`(0.6B,1.8GB),**不再依赖任何外部目录**;原先指向 `D:\ai\YuE2-T8-Local-...\models`,已于 2026-09-26 把权重与缺失文件全部拷入项目并改 `lyrics._QWEN_DIR`。仍可用 `QWEN3_ASR_DIR` / `QWEN3_ALIGNER_DIR` 覆盖。`config.json` 用 transformers 5.x 原生格式(`audio_config` + `Qwen3ASRFeatureExtractor`),旧 4.57 `thinker_config` 包装版留档在 `Qwen3-ASR\config.wrappers.json.bak`,别拿它覆盖。`local_files_only=True` 加载,离线可用;双模型 bf16 约占 6GB 显存。ForcedAligner 单次 ≤300s,`_qwen3_word_times` 按时间等比切块(文字按词数占比分配)应对长歌。
- PowerShell 里 `curl -d "{...}"` 会剥引号导致 Flask 400,要用 `Invoke-RestMethod -ContentType "application/json"`。
- 改页面后浏览器可能残留旧版(`Cache-Control: no-store` 已加),确认时 Ctrl+F5。**更隐蔽的坑:Jinja 会把 `templates\index.html` 缓存在服务进程里,改完 HTML 不重启服务,浏览器拿到的永远是旧版页面**——`no-store` 只挡浏览器缓存,挡不住服务端模板缓存。已在 `app.py` 开 `TEMPLATES_AUTO_RELOAD` + `jinja_env.auto_reload` 并给 HTML 响应加随机 ETag,改完页面直接刷新即可生效。验证页面改动是否真的上线,别只看浏览器,直接 `Invoke-WebRequest http://127.0.0.1:8000/` 抓内容比对。

## 分发/便携(2026-09-26 已收口)

- 项目**自包含、无硬编码盘符**:整目录拷走即用,离线可跑。模型查找全走 `HF_HOME`/`TORCH_HOME`(app.py 顶部)、`_BSROFORMER_MODEL_DIR`(splitter.py)、`_QWEN_DIR`(lyrics.py)、`WHISPER_DIR`(lyrics.py)。
- 可覆盖的环境变量仅:`BSROFORMER_MODEL`、`QWEN3_ASR_DIR`、`QWEN3_ALIGNER_DIR`、`WHISPER_DOWNLOAD_BASE`,一般不用设。
- 打包体积约 19.7 GB,大头:qwen 5.7G / site-packages 4.9G / whisper 4.4G / HF hub 2.8G / ffmpeg 0.31G / pymss 0.28G;`uploads\` 1.3G 是用户数据,分发时自行排除。
- 分发前自查:全文搜 `[A-Za-z]:\\` 应只在 `MIGRATE.md`/`AGENTS.md` 的说明文字里出现,代码里不能有。
- **教学视频素材(不属于程序本体,分发时可删)**:`教学视频脚本.md`(口播稿)、`口播音频\`(62 份 mp3 + `字幕.srt` + `索引.md`,共约 17 分钟)、`tools\gen_voiceover.py`(把口播稿合成音频的脚本)。`gen_voiceover.py` 依赖 `edge-tts`,装在项目外(`TTS_LIB` 环境变量指向,默认 `C:\Users\bww\AppData\Local\Temp\opencode\tts`),不随包分发;只有要重新生成音频时才需要,且要联网。

## 用户偏好

- 中文交流;研究结论记成独立 md 文档,不入代码注释。
- 需求/规则演进较频繁,改动分割逻辑前先向用户确认当前的分段规则(见上方算法规则)。