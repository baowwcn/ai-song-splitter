#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""歌词分段支持:
1. LRC 歌词(带 [mm:ss.xx] 时间戳)→ 直接取时间戳作切点
2. 纯文本歌词 → 按所选引擎识别人声得逐字时间戳,动态规划把歌词文字与
   识别文字对齐,推出每句歌词的起始时间作切点

识别引擎(model 参数):
- "funasr"(默认):Fun-ASR-Nano-2512 提取歌词 + Paraformer-zh 逐字时间戳对齐
  (达摩院中文引擎,精度约为 whisper 的 2-3 倍)
- "qwen3":Qwen3-ASR-1.7B 提取歌词 + Qwen3-ForcedAligner-0.6B 逐字时间戳对齐
  (transformers 原生;中英文都支持,歌词识别效果好)
- "large-v3" / "medium":OpenAI whisper 引擎(段内按字符占比线性插值)
"""

import os
import re
import math

import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(_BASE, "runtime", "models", "whisper")

_FUNASR_ENGINES = {}
_WHISPER_MODELS = {}
_QWEN3_ENGINES = {}

# Qwen3 模型目录(随项目走,便携;可用 QWEN3_ASR_DIR / QWEN3_ALIGNER_DIR 环境变量覆盖)
_QWEN_DIR = os.path.join(_BASE, "runtime", "models", "qwen")
QWEN3_ASR_DIR = os.environ.get(
    "QWEN3_ASR_DIR", os.path.join(_QWEN_DIR, "Qwen3-ASR"))
QWEN3_ALIGNER_DIR = os.environ.get(
    "QWEN3_ALIGNER_DIR", os.path.join(_QWEN_DIR, "Qwen3-ForcedAligner"))


def _dev_str(device):
    """funasr 接受 cpu 或 cuda:0;统一归一。"""
    return "cpu" if str(device) == "cpu" else "cuda:0"


# ---------------------------------------------------------------- FunASR

def _paraformer(device):
    """Paraformer-zh + fsmn-vad 长音频 VAD,按 (模型, 设备) 进程内复用。"""
    key = ("paraformer", str(device))
    if key not in _FUNASR_ENGINES:
        from funasr import AutoModel
        _FUNASR_ENGINES[key] = AutoModel(
            model="paraformer-zh", vad_model="fsmn-vad",
            hub="hf", device=_dev_str(device), disable_update=True)
    return _FUNASR_ENGINES[key]


def _nano(device):
    """Fun-ASR-Nano-2512 + fsmn-vad,LLM-ASR 旗舰,支持歌词识别。"""
    key = ("nano", str(device))
    if key not in _FUNASR_ENGINES:
        from funasr import AutoModel
        _FUNASR_ENGINES[key] = AutoModel(
            model="FunAudioLLM/Fun-ASR-Nano-2512", trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            hub="hf", device=_dev_str(device), disable_update=True)
    return _FUNASR_ENGINES[key]


def _paraformer_char_times(vocals_wav, device, progress=None):
    """Paraformer 逐字时间戳 → (asr_chars, 每字符时间秒)。
    返回的 asr_chars 与 paraformer 识别结果一一对应(中文字,英文具体到字母)。"""
    eng = _paraformer(device)
    res = eng.generate(input=vocals_wav, pred_timestamp=True)[0]
    raw = res.get("text", "") or ""
    ts = res.get("timestamp") or []
    tokens = raw.split()
    if not tokens or len(ts) != len(tokens):
        raise RuntimeError("Paraformer 未识别出有效语音或时间戳与文本不对应")
    chars, times = [], []
    for tok, (s, e) in zip(tokens, ts):
        t = ((float(s) + float(e)) / 2.0) / 1000.0
        for ch in _norm(tok):
            if not ch:
                continue
            chars.append(ch)
            times.append(t)
    if not chars:
        raise RuntimeError("Paraformer 未识别出有效语音,无法对齐歌词")
    return "".join(chars), times


# ---------------------------------------------------------------- Qwen3-ASR + ForcedAligner

# ForcedAligner 单次可处理 ≤300s 语音;留 10s 余量切片
_FA_MAX_SECONDS = 290.0


def _qwen3_engine(device):
    """Qwen3-ASR-1.7B(提取)+ Qwen3-ForcedAligner-0.6B(逐字时间戳),
    按设备进程内复用。两模型 transformers 5.17 原生加载。"""
    key = str(device)
    if key not in _QWEN3_ENGINES:
        if not os.path.isdir(QWEN3_ASR_DIR):
            raise RuntimeError(
                "未找到 Qwen3-ASR 模型目录:%s\n"
                "可在环境变量设 QWEN3_ASR_DIR 指向模型位置" % QWEN3_ASR_DIR)
        if not os.path.isdir(QWEN3_ALIGNER_DIR):
            raise RuntimeError(
                "未找到 Qwen3-ForcedAligner 模型目录:%s\n"
                "可在环境变量设 QWEN3_ALIGNER_DIR 指向模型位置" % QWEN3_ALIGNER_DIR)
        import torch
        from transformers import (AutoModelForSpeechSeq2Seq,
                                  AutoModelForTokenClassification,
                                  AutoProcessor)
        dev = _dev_str(device)
        dtype = torch.bfloat16 if dev != "cpu" else torch.float32
        asr_proc = AutoProcessor.from_pretrained(
            QWEN3_ASR_DIR, local_files_only=True)
        asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            QWEN3_ASR_DIR, dtype=dtype).to(dev)
        al_proc = AutoProcessor.from_pretrained(
            QWEN3_ALIGNER_DIR, local_files_only=True)
        al = AutoModelForTokenClassification.from_pretrained(
            QWEN3_ALIGNER_DIR, dtype=dtype).to(dev)
        _QWEN3_ENGINES[key] = (asr_proc, asr, al_proc, al)
    return _QWEN3_ENGINES[key]


def _qwen3_audio16k(wav):
    """vocals wav → 16kHz 单声道 float32 数组。
    BSRoformer 等来源可能输出 float32/非 16bit PCM,先强制重编码为 16bit,避免
    把 float 字节当 int16 误解码导致 ASR 塌缩。"""
    from pydub import AudioSegment
    a = AudioSegment.from_wav(wav)
    if a.sample_width != 2:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            a.export(tmp, format="wav", parameters=["-acodec", "pcm_s16le"])
            a = AudioSegment.from_wav(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if a.frame_rate != 16000 or a.channels != 1:
        a = a.set_frame_rate(16000).set_channels(1)
    return (np.frombuffer(a.raw_data, dtype=np.int16).astype(np.float32)
            / 32768.0), 16000


def _qwen3_to_device(inputs, dev, dtype):
    import torch
    out = {}
    for k, v in inputs.items():
        v = v.to(dev)
        if v.is_floating_point():
            v = v.to(dtype)
        out[k] = v
    return out


def _qwen3_transcribe(wav, device, language="zh", progress=None):
    """Qwen3-ASR 转写人声轨 → 纯文本(带标点)。
    language: ISO 码("zh"/"en")或全名("Chinese"/"English")。"""
    asr_proc, asr, _, _ = _qwen3_engine(device)
    import torch
    audio, sr = _qwen3_audio16k(wav)
    inputs = asr_proc.apply_transcription_request(
        audio=audio, language=(language or None),
        audio_kwargs={"sampling_rate": sr})
    inputs = _qwen3_to_device(inputs, _dev_str(device), asr.dtype)
    with torch.inference_mode():
        out = asr.generate(**inputs, max_new_tokens=1500, do_sample=False)
    return asr_proc.decode(out[0], return_format="transcription_only")


def _qwen3_align(audio, sr, transcript, device, offset_s=0.0):
    """对 (16kHz float32 音频, 转写文本) 跑一次 ForcedAlign,返回
    [{text,start_time,end_time}],时间已加 offset_s(切块偏移)。"""
    _, _, al_proc, al = _qwen3_engine(device)
    import torch
    inputs, word_lists = al_proc.prepare_forced_aligner_inputs(
        audio=audio, transcript=transcript, language="Chinese",
        audio_kwargs={"sampling_rate": sr})
    inputs = _qwen3_to_device(inputs, _dev_str(device), al.dtype)
    with torch.inference_mode():
        outputs = al(**inputs)
    rows = al_proc.decode_forced_alignment(
        logits=outputs.logits, input_ids=inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=al.config.timestamp_token_id)[0]
    dur = len(audio) / sr
    for r in rows:
        r["start_time"] = max(0.0, min(offset_s + r["start_time"], dur))
        r["end_time"] = max(0.0, min(offset_s + r["end_time"], dur))
    return rows


def _qwen3_word_times(wav, transcript, device):
    """整首歌对齐(超 290s 自动切块):返回逐词 [{text,start_time,end_time}]。"""
    audio, sr = _qwen3_audio16k(wav)
    if len(audio) / sr <= _FA_MAX_SECONDS:
        return _qwen3_align(audio, sr, transcript, device)
    # 长歌按时间等比切块,每块文本词数按占比分配
    words = transcript.split()
    n_chunks = int(math.ceil((len(audio) / sr) / _FA_MAX_SECONDS))
    n_samp = len(audio)
    rows = []
    for ci in range(n_chunks):
        s0 = int(n_samp * ci / n_chunks)
        s1 = int(n_samp * (ci + 1) / n_chunks)
        w0 = int(len(words) * ci / n_chunks)
        w1 = int(len(words) * (ci + 1) / n_chunks)
        seg_words = words[w0:w1]
        if not seg_words:
            continue
        rows.extend(_qwen3_align(audio[s0:s1], sr, " ".join(seg_words).strip(),
                                 device, offset_s=s0 / sr))
    return rows


def _qwen3_char_times(wav, device, language="zh", progress=None):
    """Qwen3-ASR 转写 + ForcedAligner 逐字对齐 → (asr_chars, 每字符时间秒)。
    中文字逐字符;英文按词的中点给词内每个字符同一时间。"""
    transcript = _qwen3_transcribe(wav, device, language=language)
    words = _qwen3_word_times(wav, transcript, device)
    chars, times = [], []
    for w in words:
        t = (float(w["start_time"]) + float(w["end_time"])) / 2.0
        for ch in _norm(w["text"]):
            if not ch:
                continue
            chars.append(ch)
            times.append(t)
    if not chars:
        raise RuntimeError("Qwen3 未识别出有效语音,无法对齐歌词")
    return "".join(chars), times


# ---------------------------------------------------------------- OpenAI whisper

# openai-whisper 官方 .pt(azure CDN,国内可能慢)。可用环境变量
# WHISPER_DOWNLOAD_BASE 指定镜像前缀(下拉模型名 .pt),例如:
#   WHISPER_DOWNLOAD_BASE=https://your-mirror/whisper
# 不设则走官方直链(下载支持断点续传,断线重跑即可)。
_WHISPER_URLS = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

# 各模型完整大小(字节),用于识别"下了一半就中断"的半截文件
_WHISPER_EXPECTED_SIZES = {
    "tiny": 75619168, "base": 147850135, "small": 486585216,
    "medium": 1528008539, "large": 3087371615, "large-v3": 3087371615,
}


def _fetch_whisper_model(model, base_url=None):
    """确保 runtime\\models\\whisper\\<model>.pt 完整存在;缺失/半截则下载续传。
    base_url 若给出,则 URL = base_url/<model>.pt(镜像布局),否则官方直链。
    下载按 Content-Length/Content-Range 校验,中途断流不判完成,自动续传。"""
    path = os.path.join(WHISPER_DIR, model + ".pt")
    tmp = path + ".part"

    def complete(path):
        actual = os.path.getsize(path)
        expected = _WHISPER_EXPECTED_SIZES.get(model)
        if expected is not None:
            return actual >= expected
        return actual > 1000000

    # 已有完整文件:跳过。半截文件则挪作 .part 续传
    if os.path.exists(path):
        if complete(path):
            return
        os.replace(path, tmp)
    # .part 已完整(上次 replace 前崩溃):直接收尾
    if os.path.exists(tmp) and complete(tmp):
        os.replace(tmp, path)
        return

    if base_url is None:
        base_url = os.environ.get("WHISPER_DOWNLOAD_BASE", "").strip()
    if base_url:
        url = base_url.rstrip("/") + "/" + model + ".pt"
    else:
        url = _WHISPER_URLS.get(model)
    if not url:
        return  # 未知模型,交给 whisper 库自行处理(可能慢或联网失败)
    os.makedirs(WHISPER_DIR, exist_ok=True)

    import urllib.request
    last_err = None
    for attempt in range(200):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-" % have})
            with urllib.request.urlopen(req, timeout=120) as r:
                content_range = r.headers.get("Content-Range", "") or ""
                if "/" in content_range:
                    total = int(content_range.rsplit("/", 1)[-1])
                else:
                    total = int(r.headers.get("Content-Length") or 0) + (have if r.status == 206 else 0)
                mode = "ab" if have > 0 and r.status == 206 else "wb"
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            got = os.path.getsize(tmp)
            if total and got < total * 0.99:
                raise IOError("incomplete %d/%d" % (got, total))  # 主动断言失败,续传
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            print("...下载中断(%s, 已 %.1f MB),继续续传..." %
                  (e, (os.path.getsize(tmp) if os.path.exists(tmp) else 0) / 1e6))
    raise RuntimeError("whisper 模型 %s 下载失败(%s);建议设 WHISPER_DOWNLOAD_BASE 镜像,或稍后重跑 tools\\download_whisper.py 续传" % (model, last_err))


def _whisper_engine(model, device):
    key = (model, str(device))
    if key not in _WHISPER_MODELS:
        import whisper
        _fetch_whisper_model(model)
        _WHISPER_MODELS[key] = whisper.load_model(
            model, device=device, download_root=WHISPER_DIR)
    return _WHISPER_MODELS[key]


def _whisper_transcribe(model, wav, device, language="zh"):
    """whisper 转写人声轨,0.2/0.4 两温度按语音覆盖时长取优。
    language: "zh"/"en" 等 whisper 语言码;None 时自动检测。"""
    from pydub import AudioSegment
    eng = _whisper_engine(model, device)
    total = len(AudioSegment.from_wav(wav)) / 1000.0
    best, best_score = None, float("-inf")
    for temp in (0.2, 0.4):
        res = eng.transcribe(wav, language=language, temperature=temp,
                             fp16=(device != "cpu"), verbose=False)
        segs = res.get("segments") or []
        text = res.get("text") or ""
        if not segs:
            continue
        speech = sum(max(0.0, min(s["end"], total) - s["start"])
                     for s in segs)
        score = speech + min(len(text) * 0.001, 1.0)
        if score > best_score:
            best, best_score = res, score
    return best if best is not None else eng.transcribe(
        wav, language=language, temperature=0.2, fp16=(device != "cpu"), verbose=False)


def _whisper_char_times(model, wav, device, language="zh", progress=None):
    """whisper 段内按字符占比线性插值 → (asr_chars, 每字符时间秒)。
    繁体→简体归一化,丢标点/空白。"""
    result = _whisper_transcribe(model, wav, device, language=language)
    chars, times = [], []
    for seg in result.get("segments") or []:
        s_ms = float(seg["start"]) * 1000.0
        e_ms = float(seg["end"]) * 1000.0
        kept = [ch for ch in _norm(seg["text"]) if ch]
        n = len(kept)
        if n == 0 or e_ms <= s_ms:
            continue
        for k, ch in enumerate(kept):
            chars.append(ch)
            times.append((s_ms + (e_ms - s_ms) * (k + 0.5) / n) / 1000.0)
    if not chars:
        raise RuntimeError("whisper 未识别出有效语音,无法对齐歌词")
    return "".join(chars), times


# ---------------------------------------------------------------- LRC

_LRC_TIME = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")


def is_timed_lyrics(text):
    """是否含 LRC 时间戳格式。"""
    return bool(_LRC_TIME.search(text or ""))


def parse_lrc(text):
    """解析 LRC/时间戳歌词 → 升序 [(time_ms, line)]。元数据标签([ar:]等)忽略。"""
    items = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        times = []
        rest = line
        m = _LRC_TIME.match(rest)
        while m:
            mm, ss, ff = int(m.group(1)), int(m.group(2)), m.group(3) or "0"
            ms = (mm * 60 + ss) * 1000 + int(ff.ljust(3, "0")[:3])
            times.append(ms)
            rest = rest[m.end():]
            m = _LRC_TIME.match(rest)
        text = rest.strip("  ")
        if times and not text:
            continue  # 纯标签行(元数据)忽略
        for t in sorted(times):
            items.append((t, text))
    items.sort(key=lambda x: x[0])
    return items


def lrc_cut_points(text):
    """LRC 歌词 → 候选切点(ms):
    每条第时间戳行再按句末标点分句,行内区间按字符数比例分配时间,让
    "。" "！" 等处的分段也断开。过滤紧贴首尾的切点。"""
    items = parse_lrc(text)
    if not items:
        return []
    times = [t for t, _ in items]
    cuts = set()
    for i, (t, raw) in enumerate(items):
        cls = _split_clauses(raw)
        if not cls:
            continue
        next_t = times[i + 1] if i + 1 < len(items) else None
        if next_t is None or next_t <= t:
            continue  # 末行无区间可分配
        span = next_t - t
        total_chars = sum(len(n) for n, _ in cls)
        for _, off in cls:
            if off <= 0:
                continue  # 行首本身是切点,由下一行去减
            ct = t + int(span * off / total_chars)
            if ct < 250 or ct > next_t - 250:
                continue
            cuts.add(ct)
    # 行首时间戳本身也算切点
    for t in times:
        if 250 < t:
            cuts.add(t)
    return sorted(cuts)


# ---------------------------------------------------------------- 文字归一化

_OPENCC_T2S = None


def _t2s(s):
    """繁体→简体(whisper 常输出繁体,简体歌词须归一才能对上)。"""
    global _OPENCC_T2S
    try:
        if _OPENCC_T2S is None:
            from opencc import OpenCC
            _OPENCC_T2S = OpenCC("t2s")
        return _OPENCC_T2S.convert(s)
    except Exception:
        return s


def _norm(s):
    """去标点/空白/大小写,繁体转简体,保留中文与字母数字。"""
    return re.sub(r"[\W_]+", "", _t2s(s)).lower()


# 歌词标点:歌词里出现的标点都按此处断开(。，、！？…等中英文常见标点全覆盖)
_PUNCT = re.compile(
    r"[。．·•，,、！!？?…;；：:～~《》〈〉「」『』()（）\[\]【】<>——–\-]+")


def _split_clauses(line):
    """把一行歌词按标点切成若干分句,返回 [(归一化分句, 行内字符偏移)]。
    偏移基于归一化后的字符位置(标点不进偏移量)。"""
    out = []
    pos = 0
    for part in _PUNCT.split(line):
        if not part:
            continue
        n = _norm(part)
        if not n:
            continue
        out.append((n, pos))
        pos += len(n)
    return out


# ---------------------------------------------------------------- 动态规划对齐

def _align_times(user_chars, asr_chars, asr_char_times):
    """全局对齐 user_chars 与 asr_chars,返回每个 user 字符的时间(秒)。
    缺失处用相邻已匹配字符时间线性插值。"""
    M, N = len(user_chars), len(asr_chars)
    if M == 0 or N == 0:
        return []

    INF = int(1e9)
    dp = np.full((M + 1, N + 1), INF, dtype=np.int32)
    dp[0, 0] = 0
    dp[0, 1:] = np.arange(1, N + 1, dtype=np.int32)
    dp[1:, 0] = np.arange(1, M + 1, dtype=np.int32)

    A = np.array([ord(c) for c in asr_chars], dtype=np.int32)
    for i in range(1, M + 1):
        ui = ord(user_chars[i - 1])
        eqc = (A != ui).astype(np.int32)  # mismatch 代价
        prev = dp[i - 1]
        c = np.minimum(prev[1:] + 1, prev[:-1] + eqc)
        j = np.arange(1, N + 1, dtype=np.int32)
        dp[i, 1:] = j + np.minimum.accumulate(c - j)

    # 回溯得到每个 user 字符对应的 asr 字符下标(或 -1 表示未匹配到)
    mapping = [-1] * M
    i, j = M, N
    while i > 0 and j > 0:
        cost = 0 if user_chars[i - 1] == asr_chars[j - 1] else 1
        if dp[i, j] == dp[i - 1, j - 1] + cost:
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i, j] == dp[i - 1, j] + 1:
            i -= 1  # user 字符跳到(前部未对齐),保持 -1
        else:
            j -= 1
    # 由 mapping + asr_char_times 生成时间,缺失处插值
    times = []
    for idx in mapping:
        if idx >= 0:
            times.append(asr_char_times[idx])
        else:
            times.append(None)
    filled = times
    last = None
    for k in range(len(filled)):
        if filled[k] is None:
            nxt = next((filled[q] for q in range(k + 1, len(filled)) if filled[q] is not None), None)
            if last is not None and nxt is not None:
                filled[k] = last + (nxt - last) * 0.5
            else:
                filled[k] = nxt if nxt is not None else (last if last is not None else 0.0)
        else:
            last = filled[k]
    return filled


# ---------------------------------------------------------------- 主入口

def extract_lyrics(vocals_wav, model="funasr", device="cuda", language="zh",
                   progress=None):
    """无人声歌词(无参考文本):识别引擎识别人声,返回逐行文本
    (自带标点断句,供用户核对修正后再切)。
    model: "funasr"(Fun-ASR-Nano,仅中文)/ "qwen3"(Qwen3-ASR,中英可)
           / "large-v3" / "medium"(whisper)。
    language: 识别语言码("zh"/"en"),funasr 固定中文。"""
    def report(msg, pct=None):
        if progress:
            progress(msg, pct)

    if str(model) == "funasr":
        report("加载 Fun-ASR-Nano 歌词识别模型(首次下载约 4.6GB)...", 50)
        eng = _nano(device)
        report("Fun-ASR-Nano 识别歌词(中文,GPU)...", 60)
        res = eng.generate(input=[vocals_wav], cache={}, batch_size=1,
                           language="中文", itn=True)
        text = (res[0].get("text") or "").strip()
    elif str(model) == "qwen3":
        report("Qwen3-ASR 识别歌词(中英,GPU)...", 60)
        text = (_qwen3_transcribe(vocals_wav, device, language=language)
                or "").strip()
    else:
        report("加载 whisper 模型(%s)..." % model, 50)
        text = (_whisper_transcribe(model, vocals_wav, device, language=language)
                .get("text") or "").strip()

    if not text:
        raise RuntimeError("识别引擎未输出有效歌词")
    # 按句末标点断行方便用户核对
    text = re.sub(r"(?<=[。！？.!?])", "\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines or [text]


def align_lines_to_audio(vocals_wav, lines, model="funasr", device="cuda",
                         language="zh", progress=None):
    """纯文本歌词对齐:识别人声得逐字时间戳 → DP 对齐用户歌词 → 每句起始 ms。
    model: "funasr"(Paraformer 逐字毫秒时间戳,仅中文)
           / "qwen3"(Qwen3-ASR + ForcedAligner 逐字对齐,中英可)
           / "large-v3" / "medium"(whisper)。
    language: 识别语言码("zh"/"en")。"""
    def report(msg, pct=None):
        if progress:
            progress(msg, pct)

    user_lines = [_norm(l) for l in lines]
    user_lines = [l for l in user_lines if l]
    if not user_lines:
        raise RuntimeError("歌词内容为空")
    user_chars = "".join(user_lines)

    if str(model) == "funasr":
        report("FunASR Paraformer 识别人声(中文,逐字时间戳)...")
        asr_chars, asr_char_times = _paraformer_char_times(vocals_wav, device)
    elif str(model) == "qwen3":
        report("Qwen3-ASR 识别人声 + ForcedAligner 逐字对齐(中英,GPU)...")
        asr_chars, asr_char_times = _qwen3_char_times(
            vocals_wav, device, language=language)
    else:
        report("whisper 识别人声(中文,%s,段内插值)..." % model)
        asr_chars, asr_char_times = _whisper_char_times(
            model, vocals_wav, device, language=language)

    line_starts = []
    acc = 0
    for i, l in enumerate(user_lines):
        line_starts.append((i, acc))
        acc += len(l)

    report("对齐歌词文字与音频时间...")
    char_times = _align_times(user_chars, asr_chars, asr_char_times)

    # 每行按句末标点再分句,每个分句起头都是一刀
    cuts = []
    prev = 0.0
    for i, start in line_starts:
        base = start
        for ncl, off in _split_clauses(lines[i]):
            t = char_times[base + off]
            t = max(float(t), prev + 0.05)  # 强制严格递增,防多刀叠在一起
            prev = t
            cuts.append(int(round(t * 1000)))
        base += len(user_lines[i])
    return [c for c in cuts if c > 0]


# ---------------------------------------------------------------- 汇总

def lyric_cut_points(text, vocals_wav, lines, device="cuda", model="funasr",
                     language="zh", progress=None):
    """按歌词取切点:
    - 带时间戳(LRC)→ 直接用戳
    - 纯文本 → 语音对齐
    """
    if is_timed_lyrics(text):
        if progress:
            progress("解析歌词时间戳...", None)
        return lrc_cut_points(text)
    return align_lines_to_audio(vocals_wav, lines, model=model, device=device,
                                language=language, progress=progress)


def lyrics_for_segments(text, lines, cuts, bounds):
    """把歌词按最终段边界切配到每段,返回 len(bounds)-1 个歌词文本。
    纯文本:cuts 取 align_lines_to_audio 的返回值(每分句一刀),按 _split_clauses
    相同的过滤复原原文分句;LRC:按 parse_lrc 时间戳行归属各段。间奏等无歌词
    段返回空串。"""
    def _raw_clauses(raw):
        """按 _PUNCT 断句,但把标点保留并挂到分句尾部(_PUNCT.split 会剥掉标点)。"""
        out = []
        start = 0
        for m in _PUNCT.finditer(raw):
            part = raw[start:m.start()].strip()
            if part and _norm(part):
                out.append(part + m.group())
            start = m.end()
        tail = raw[start:].strip()
        if tail and _norm(tail):
            out.append(tail)
        return out

    events = []  # (time_ms, text)
    if is_timed_lyrics(text):
        for t, raw in parse_lrc(text):
            if raw.strip():
                events.append((t, raw.strip()))
    else:
        cs = list(cuts)
        idx = 0
        for ln in lines:
            for raw in _raw_clauses(ln):
                if idx >= len(cs):
                    break
                events.append((cs[idx], raw))
                idx += 1

    out = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        out.append("".join(t for tm, t in events if s <= tm < e))
    return out


def write_timed_lyrics(text, lines, cuts, bounds, path):
    """生成 LRC 风格「带时间戳歌词」txt:每段起始时间 + 该段歌词。"""
    seg_texts = lyrics_for_segments(text, lines, cuts, bounds)
    total_ms = bounds[-1] if bounds else 0
    lines_out = ["[length:%.2f]" % (total_ms / 1000.0)]
    for s, txt in zip(bounds, seg_texts):
        mm = s // 60000
        ss = (s % 60000) // 1000
        cc = (s % 1000) // 10
        lines_out.append("[%02d:%02d.%02d] %s" % (mm, ss, cc, txt))
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines_out) + "\n")
    return path