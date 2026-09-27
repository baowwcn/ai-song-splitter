#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""歌曲分割核心:在歌声断点(间隙)处切割,每段控制在 [min_len, max_len] 秒。"""

import math
import os
import statistics
import subprocess
import sys
import tempfile

from pydub import AudioSegment

# demucs 4.x 模型走 huggingface hub 缓存,指向项目内便于离线/便携
_BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault(
    "HF_HOME", os.path.join(_BASE, "runtime", "models", "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# BSRoformer(pymss)模型目录,走项目内便于离线/便携
_BSROFORMER_MODEL_DIR = os.path.join(_BASE, "runtime", "models", "pymss")


# ---------------------------------------------------------------- 加载

def load_audio(path):
    return AudioSegment.from_file(path)


# ---------------------------------------------------------------- 静音检测

def detect_silence_abs(audio, silence_thresh_db, min_silence_ms):
    """绝对阈值模式:适合纯人声轨(间隙处是真静音)。"""
    from pydub.silence import detect_silence
    return detect_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        seek_step=10,
    )


def detect_silence_rel(audio, frame_ms=50, window_ms=500,
                       rel_thresh_db=10, min_silence_ms=500):
    """相对能量模式:低于【全局中位数 - rel_thresh_db】的连续段视为间隙。
    对"伴奏还在"的原曲有一定抗性。"""
    n = len(audio)
    frame_len = frame_ms
    frames = []
    for start in range(0, n, frame_len):
        frames.append(audio[start:start + frame_len].rms)

    win = max(1, window_ms // frame_len)
    smoothed = []
    for i in range(len(frames)):
        lo = max(0, i - win // 2)
        hi = min(len(frames), i + win // 2 + 1)
        seg = frames[lo:hi]
        smoothed.append(sum(seg) / len(seg))

    def to_db(rms):
        return 20 * math.log10(rms / 32767.0) if rms > 0 else -120.0

    positive = [f for f in smoothed if f > 0]
    med_db = to_db(statistics.median(positive)) if positive else -60.0
    thresh = med_db - rel_thresh_db

    silences = []
    in_sil = False
    start_ms = 0
    for i, rms in enumerate(smoothed):
        if to_db(rms) < thresh:
            if not in_sil:
                in_sil = True
                start_ms = i * frame_len
        else:
            if in_sil:
                in_sil = False
                if i * frame_len - start_ms >= min_silence_ms:
                    silences.append((start_ms, i * frame_len))
    if in_sil and len(smoothed) * frame_len - start_ms >= min_silence_ms:
        silences.append((start_ms, len(smoothed) * frame_len))
    return silences


# ---------------------------------------------------------------- 切点

def cut_points_from_silences(silences, total_ms, keep_silence_ms=250):
    """每个间隙段的中点作为候选切点;过滤紧贴首尾的。"""
    cuts = []
    for s, e in silences:
        mid = (s + e) // 2
        if mid < keep_silence_ms or mid > total_ms - keep_silence_ms:
            continue
        cuts.append(mid)
    return cuts


def snap_cuts_to_gaps(vocals, cuts, win_ms=250):
    """把歌词切点向左吸到最近的能量谷底(两分句之间的停顿):
    切点若偏晚,上一段的尾巴会带上下一句第一个字的音,吸到谷底可削掉它。"""
    import numpy as np
    total = len(vocals)
    sr = vocals.frame_rate
    cuts = sorted(set(int(c) for c in cuts if 0 < c < total))
    if not cuts:
        return cuts
    samples = np.frombuffer(vocals.get_array_of_samples(), dtype=np.int16)
    n = max(10, int(50 * sr / 1000))          # 50ms 能量窗
    hop = max(5, int(10 * sr / 1000))         # 10ms 步长
    out = []
    prev = 0
    for c in cuts:
        lo = max(0, c - win_ms)
        hi = min(total, c + 60)               # 只往前找,不许越到句内更远处
        bi = int(lo * sr / 1000)
        ei = int(hi * sr / 1000)
        seg = samples[bi:ei]
        if len(seg) > n:
            buf = np.empty(len(seg) + n, dtype=np.int16)
            buf[:len(seg)] = seg
            wins = []
            for st in range(0, max(1, len(seg) - n), hop):
                w = buf[st:st + n]
                wins.append((float(np.sqrt(np.mean(np.square(w.astype(np.float64))))), st))
            st = min(wins, key=lambda x: x[0])[1]
            bc = lo + int(st * 1000 / sr)
        else:
            bc = c - int(0.15 * 1000)
        bc = max(bc, prev + 80)               # 维持严格递增、防与下一刀打架
        out.append(bc)
        prev = bc
    return out


# ---------------------------------------------------------------- 分段(8-15s 约束)

def segment_with_bounds(audio, cuts, min_len_ms, max_len_ms):
    """贪心分段:优先在断点处切,同时尽量满足 [min_len, max_len]。
    - 范围内有断点:取段长尽量长的断点
    - 断点太密(< min):跳过,取第一个 >= min 的断点
    - 断点太疏(首个就 > max):硬切在 max 处
    - 最后一段不足 min:并入前一段;并入会超 max 时收尾把上一切点左移凑长,
      优先落断点(允许上一段略短于 min),无断点才按时间收
    """
    total = len(audio)
    bounds = [0]
    cur = 0
    while cur < total:
        remaining = total - cur
        if remaining <= max_len_ms:
            if remaining >= min_len_ms:
                bounds.append(total)
            elif len(bounds) >= 2 and total - bounds[-2] <= max_len_ms:
                bounds[-1] = total  # 并入前一段,且合并后不超 max
            else:
                bounds.append(total)  # 合并会超 max,单独留一小段
            break

        candidates = [c for c in cuts if cur < c < total]
        valid = [c for c in candidates
                 if cur + min_len_ms <= c <= cur + max_len_ms]
        if valid:
            nxt = valid[-1]  # 段长尽量长,跳过过密断点
        else:
            over_min = [c for c in candidates if c >= cur + min_len_ms]
            if over_min and over_min[0] <= cur + max_len_ms:
                nxt = over_min[0]
            else:
                nxt = cur + max_len_ms  # 太疏或无边,硬切在 max
        bounds.append(nxt)
        cur = nxt

    # 收尾:末段不足 min 且并入会超 max 时,把上一切点左移,让末段补到 min。
    # 优先落既有的歌词标点/间奏切点——即使上一段会略短于 min 也坚持标点
    # (用户规则:标点优先,允许略短);全部切点都不合适才按时间收。
    if (len(bounds) >= 3 and bounds[-1] == total
            and total - bounds[-2] < min_len_ms):
        tail = total - bounds[-2]
        prev_len = bounds[-2] - bounds[-3]
        need = min_len_ms - tail
        if prev_len - need >= min_len_ms:
            target = bounds[-2] - need
            earlier = [c for c in cuts if bounds[-3] < c < bounds[-2]
                       and c <= target]
            bound = None
            if earlier:
                # 选一个能让上一段仍 ≥ min 的最大标点切点;
                # 若都不够,退回最近的标点切点(允许上一段略短)
                for c in reversed(earlier):
                    if c - bounds[-3] >= min_len_ms:
                        bound = c
                        break
                if bound is None:
                    bound = earlier[-1]
            else:
                bound = target
            bounds[-2] = bound

    segments = []
    for i in range(len(bounds) - 1):
        segments.append(audio[bounds[i]:bounds[i + 1]])
    return segments, bounds


def segment_lyrics_priority(audio, lyric_cuts, inter_sils, min_len_ms,
                            max_len_ms):
    """歌词模式最终分段:
    1. 间奏(静音 ≥ interlude_min_s 秒)= 完整独立段:不并入歌唱段,
       歌唱段的短段也不并入间奏;例外:间奏 < 7s 且相邻歌唱段 < 8s 时可合并
    2. 间奏 > max 时,平均切成若干 (>7s, ≤max] 的段
    3. 其余歌唱区域在 [min, max] 内贪心(短段合并,优先用歌词分句切点)
    返回 (segments, bounds)。"""
    total = len(audio)

    # 间奏区间:静音段贴边的并到首尾,重叠的合并
    ivs = []
    for s, e in inter_sils:
        a, b = s, e
        if a < 300:
            a = 0
        if b > total - 300:
            b = total
        if 0 < b and a < total and a < b:
            ivs.append((a, b))
    ivs = [iv for iv in sorted(set(ivs))]
    inter = []
    for a, b in ivs:
        if a == 0 and b == total:
            continue  # 整曲无实质歌唱,忽略
        if inter and a <= inter[-1][1]:
            inter[-1] = (inter[-1][0], max(inter[-1][1], b))
        else:
            inter.append((a, b))

    ly = sorted(set(c for c in lyric_cuts if 0 < c < total))

    pieces = []  # (start, end, is_interlude)
    cur = 0
    for a, b in inter:
        if a > cur:
            pieces.append((cur, a, False))
        pieces.append((a, b, True))
        cur = b
    if cur < total:
        pieces.append((cur, total, False))
    pieces = [p for p in pieces if p[1] - p[0] >= 1]

    result = []  # (s, e, is_interlude)
    for a, b, is_int in pieces:
        if is_int:
            if b - a <= max_len_ms:
                result.append((a, b, True))
                continue
            n = -(-(b - a) // max_len_ms)  # ceil(len/max)
            step = (b - a) / float(n)
            for k in range(n):
                s = a + int(round(step * k))
                e = a + int(round(step * (k + 1)))
                if s < e:
                    result.append((s, e, True))
        else:
            cuts_in = [c - a for c in ly if a < c < b]
            sub, sub_bounds = segment_with_bounds(
                audio[a:b], cuts_in, min_len_ms, max_len_ms)
            result.extend((a + s, a + e, False)
                          for s, e in zip(sub_bounds, sub_bounds[1:]))

    # 合并例外(后处理,按最终相邻段):间奏 < 7s 且相邻歌唱段 < 8s → 合并
    merged = []
    for s, e, tag in result:
        if tag:
            if (e - s) < 7000 and merged and not merged[-1][2]:
                ms, me, _ = merged[-1]
                if me == s and (me - ms) < 8000:
                    merged[-1] = (ms, e, False)
                    continue
            merged.append((s, e, True))
        else:
            if merged and merged[-1][2] and (s - merged[-1][0]) < 7000 \
                    and (e - s) < 8000 and s == merged[-1][1]:
                merged[-1] = (merged[-1][0], e, False)
            else:
                merged.append((s, e, False))
    result = merged

    # 去重复/重合边界,保证严格递增
    out = []
    for s, e, _ in result:
        if not out:
            out.append([s, e])
            continue
        if s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)  # 时间重叠则并轨
        else:
            out.append([s, e])

    bounds = [r[0] for r in out] + [out[-1][1]]
    segments = [audio[r[0]:r[1]] for r in out]
    return segments, bounds


# ---------------------------------------------------------------- 人声分离(BSRoformer 优先,demucs 兜底)

def torch_device():
    """探测 torch 可用设备:cuda 优先,其次 cpu。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _separate_bsroformer(input_path, tmp, device):
    """pymss/BSRoformer 分离人声(vocals/instrumental 双轨),返回 vocals wav 路径。
    模型名可用环境变量 BSROFORMER_MODEL 覆盖(需在 pymss 模型目录内)。"""
    from pymss import MSSeparator
    model = os.environ.get("BSROFORMER_MODEL", "bs_roformer_voc_hyperacev2")
    sep = MSSeparator.from_model_name(
        model,
        model_dir=_BSROFORMER_MODEL_DIR,
        download=True,
        device=device,
        output_format="wav",
        audio_params={"wav_bit_depth": "PCM_16"},
        store_dirs={"vocals": tmp},
    )
    try:
        sep.process_folder(input_path)
    finally:
        try:
            sep.close()
        except Exception:
            pass
    for root, _, files in os.walk(tmp):
        for f in files:
            stem = os.path.splitext(f)[0]
            if stem.endswith("_vocals"):
                return os.path.join(root, f)
    raise RuntimeError("BSRoformer 未产出 vocals 文件")


def _separate_demucs(input_path, work_dir, device):
    """旧 demucs 兜底路径(BSRoformer 不可用时)。"""
    os.makedirs(work_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="demucs_", dir=work_dir)
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-d", device, "-o", tmp, input_path]
    subprocess.run(cmd, check=True)
    for root, _, files in os.walk(tmp):
        for f in files:
            stem = os.path.splitext(f)[0]
            if stem == "vocals":
                return os.path.join(root, f)
    raise RuntimeError("demucs 未产出 vocals 文件")


def separate_vocals(input_path, work_dir, device="cuda", engine="auto"):
    """提取人声轨并返回 vocals wav 路径。
    engine: 'bsroformer' 只用 BSRoformer;'demucs' 只用旧 demucs;
    'auto'(默认)= BSRoformer 优先,失败回退 demucs。"""
    if engine == "demucs":
        return _separate_demucs(input_path, work_dir, device)
    os.makedirs(work_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="bsroformer_", dir=work_dir)
    try:
        return _separate_bsroformer(input_path, tmp, device)
    except Exception as exc:
        if engine == "bsroformer":
            raise
        import torch
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.stderr.write(f"[BSRoformer 分离失败,回退 demucs] {exc}\n")
        return _separate_demucs(input_path, work_dir, device)


# ---------------------------------------------------------------- 主入口

def split_song(input_path, out_dir, min_len_s=8, max_len_s=15,
               mode="rel", rel_thresh_db=10, silence_thresh_db=-40,
               min_silence_ms=500, keep_silence_ms=250,
               lyrics=None, whisper_model="funasr", song_lang="zh",
               interlude_min_s=3.0,
               engine="auto",
               progress=None, meta_out=None):
    """分割歌曲。progress(msg, pct) 可选回调。返回段信息列表。
    meta_out(dict) 可选:歌词模式下回传 lines/align_cuts,供手动重切后
    按新边界重写带时间戳歌词。"""
    def report(msg, pct):
        if progress:
            progress(msg, pct)

    report("加载音频...", 5)
    audio = load_audio(input_path)
    total_ms = len(audio)

    # 1. 计算候选切点
    lyrics_mode = bool(lyrics and lyrics.strip())
    if lyrics_mode:
        from lyrics import lyric_cut_points, is_timed_lyrics
        lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
        device = torch_device()
        # 英文歌一律走 whisper(Nano/Paraformer 仅中文);中文且选了 funasr 才用 FunASR
        eff_model = whisper_model
        if str(song_lang) != "zh" and whisper_model == "funasr":
            eff_model = "large-v3"
        if is_timed_lyrics(lyrics):
            # 带时间戳(LRC),直接用戳,无需人声
            cuts = lyric_cut_points(lyrics, None, lines,
                                    device=device, progress=report)
        else:
            # 纯文本 → 分离人声 + 所选引擎对齐
            if device == "cuda":
                report((("demucs" if engine == "demucs" else "BSRoformer")
                        + " 提取人声(GPU 加速,请稍候)..."), 15)
            else:
                report((("demucs" if engine == "demucs" else "BSRoformer")
                        + " 提取人声(CPU 处理,较慢,请稍候)..."), 15)
            vocals_path = separate_vocals(
                input_path, out_dir, device=device, engine=engine)
            cuts = lyric_cut_points(lyrics, vocals_path, lines,
                                    device=device, model=eff_model,
                                    language=str(song_lang), progress=report)
        if not cuts:
            raise RuntimeError("按歌词未得到任何切点,请检查歌词或更换时间戳")
        align_cuts = list(cuts)  # snap 前的对齐/时间戳切点,供带时间戳歌词配文
        # 间奏/前奏识别:无人声超过 interlude_min_s 秒的间隙单独分段。
        # 一律用人声轨做绝对阈值检测(原曲伴奏不断音会测不到间奏)。
        # LRC 模式也补一次人声分离(约几十秒,GPU)。
        inter_ms = max(500, int(interlude_min_s * 1000))
        vocals_audio = None
        try:
            if not is_timed_lyrics(lyrics) and device:
                vocals_audio = load_audio(vocals_path)
            elif is_timed_lyrics(lyrics):
                vs = separate_vocals(input_path, out_dir, device=device,
                                     engine=engine)
                vocals_audio = load_audio(vs)
        except Exception:
            vocals_audio = None
        if vocals_audio is not None:
            cuts = snap_cuts_to_gaps(vocals_audio, cuts)  # 削掉上一段尾带上下一句首字音
        if vocals_audio is not None:
            inter_sil = detect_silence_abs(
                vocals_audio, silence_thresh_db, inter_ms)
        else:
            inter_sil = detect_silence_rel(
                audio, rel_thresh_db=rel_thresh_db, min_silence_ms=inter_ms)
        if inter_sil:
            n = len(inter_sil)
            report(f"检测到 {n} 处间奏/前奏"
                   f"(无人声超 {interlude_min_s:g}s),作为完整独立段", 55)
    else:
        if mode in ("demucs", "bsroformer"):
            # 人声分离模式:demucs=仅旧 demucs;bsroformer=BSRoformer 优先+失败回退
            sep_engine = "auto" if mode == "bsroformer" else "demucs"
            device = torch_device()
            if device == "cuda":
                report((("demucs" if sep_engine == "demucs" else "BSRoformer")
                        + " 分离人声(GPU 加速,请稍候)..."), 15)
            else:
                report((("demucs" if sep_engine == "demucs" else "BSRoformer")
                        + " 分离人声(CPU 处理,较慢,请稍候)..."), 15)
            vocals_path = separate_vocals(input_path, out_dir, device=device,
                                          engine=sep_engine)
            vocals = load_audio(vocals_path)
            silences = detect_silence_abs(vocals, silence_thresh_db, min_silence_ms)
        else:
            report("检测歌声间隙...", 20)
            silences = detect_silence_rel(audio, rel_thresh_db=rel_thresh_db,
                                          min_silence_ms=min_silence_ms)

        if not silences:
            raise RuntimeError("未检测到任何歌声间隙,请尝试 BSRoformer 模式或调低阈值")
        cuts = cut_points_from_silences(silences, total_ms, keep_silence_ms)

    report(f"找到 {len(cuts)} 个切点,开始分段...", 60)

    # 2. 分段:歌词模式按 [标点分句 → 短段合并(8-15) → 间奏独立] 优先布局;
    #  其他模式贪心保证 8-15s
    min_ms = int(min_len_s * 1000)
    max_ms = int(max_len_s * 1000)
    if lyrics_mode:
        segments, bounds = segment_lyrics_priority(
            audio, cuts, inter_sil, min_len_ms=min_ms, max_len_ms=max_ms)
    else:
        segments, bounds = segment_with_bounds(
            audio, cuts, min_len_ms=min_ms, max_len_ms=max_ms)

    if lyrics_mode:
        report("按歌词切分完成", 65)

    if lyrics_mode:
        if meta_out is not None:
            meta_out["lines"] = list(lines)
            meta_out["align_cuts"] = list(align_cuts)
        # 生成带时间戳歌词 txt(每段起始时间 + 该段歌词),供下载
        try:
            from lyrics import write_timed_lyrics
            lrc_path = os.path.join(out_dir, "歌词_时间戳.txt")
            write_timed_lyrics(lyrics, lines, align_cuts, bounds, lrc_path)
        except Exception as e:
            sys.stderr.write(f"[带时间戳歌词生成失败] {e}\n")

    # 4. 导出
    os.makedirs(out_dir, exist_ok=True)
    result = []
    for i, seg in enumerate(segments, 1):
        name = f"segment_{i:03d}.wav"
        path = os.path.join(out_dir, name)
        seg.export(path, format="wav")
        result.append({
            "index": i,
            "file": name,
            "path": path,
            "start_ms": bounds[i - 1],
            "end_ms": bounds[i],
            "duration_s": round(len(seg) / 1000.0, 2),
        })
        report(f"导出第 {i}/{len(segments)} 段...",
               60 + int(40 * i / len(segments)))

    report("完成", 100)
    return result