#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""歌曲分割 Web 应用:按歌声断点切割,每段 8-15 秒。"""

import glob
import io
import json
import os
import sys
import threading
import time
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
# embeddable Python 的 ._pth 限制 sys.path,需手动加入项目目录
if BASE not in sys.path:
    sys.path.insert(0, BASE)


def _find_ffmpeg_bin():
    """自动查找 ffmpeg:项目内 tools/ffmpeg/<版本>/bin,找不到返回 None。
    只认项目内,保证整个目录拷走即可运行(不依赖上级目录)。"""
    for hit in glob.glob(os.path.join(BASE, "tools", "ffmpeg", "*", "bin")):
        if os.path.isfile(os.path.join(hit, "ffmpeg.exe")):
            return hit
    return None


# 确保 ffmpeg 可用(pydub 读 mp3/flac 等、torchaudio 均依赖),不依赖启动脚本
FFMPEG_BIN = _find_ffmpeg_bin()
if FFMPEG_BIN and FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# demucs 模型随项目走(便携):runtime\models\torch\hub\checkpoints\<hash>.th
os.environ["TORCH_HOME"] = os.path.join(BASE, "runtime", "models", "torch")
# demucs 4.x 通过 huggingface hub 下载模型,同样收进项目内,离线可用
os.environ["HF_HOME"] = os.path.join(BASE, "runtime", "models", "huggingface")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from flask import Flask, jsonify, render_template, request, send_file

import lyrics
from splitter import (load_audio, separate_vocals, split_song,
                      torch_device)

UPLOAD_DIR = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB
# 模板改动立刻生效:Jinja 默认把模板缓存在进程里,不加这两行的话
# 改完 index.html 不重启服务,浏览器拿到的仍是旧版页面(no-store 只挡浏览器缓存)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


@app.after_request
def no_cache(resp):
    """禁用页面缓存,避免更新后浏览器仍显示旧版。"""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    if resp.mimetype == "text/html":
        resp.headers["ETag"] = "no-store-%d" % time.time_ns()
    return resp

JOBS = {}  # job_id -> {status, progress, message, segments, error, name}
JOBS_LOCK = threading.Lock()


def fmt(ms):
    s = ms / 1000.0
    return f"{int(s // 60):02d}:{s % 60:06.3f}"


def new_job_id():
    """任务 ID = 上传时刻(20260926_093237),便于在 uploads 里按时间找结果。"""
    base = time.strftime("%Y%m%d_%H%M%S")
    job_id, n = base, 2
    while os.path.exists(os.path.join(UPLOAD_DIR, job_id)) or job_id in JOBS:
        job_id = "%s-%d" % (base, n)
        n += 1
    return job_id


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "未选择文件"}), 400
    job_id = new_job_id()
    ext = os.path.splitext(f.filename)[1].lower() or ".wav"
    path = os.path.join(UPLOAD_DIR, job_id + ext)
    f.save(path)
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "uploaded", "progress": 0, "message": "已上传,等待分割",
            "segments": [], "error": None, "name": f.filename,
            "path": path, "out_dir": os.path.join(UPLOAD_DIR, job_id),
            "lyrics_status": "idle", "lyrics_error": None,
            "extracted_lyrics": "",
            "lyrics_txt": None, "lyrics_align_cuts": None,
            "lyrics_lines": None,
        }
    return jsonify({"ok": True, "job_id": job_id, "name": f.filename})


@app.route("/split", methods=["POST"])
def split():
    data = request.get_json() or {}
    job_id = data.get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "任务不存在"}), 404
        if job["status"] == "running":
            return jsonify({"ok": False, "error": "任务正在运行"}), 400
        if job["lyrics_status"] == "extracting":
            return jsonify({"ok": False, "error": "歌词提取中,请稍候"}), 400
        job.update({
            "status": "running", "progress": 0, "message": "开始",
            "segments": [], "error": None,
            "min_len_s": float(data.get("min_len_s", 8)),
            "max_len_s": float(data.get("max_len_s", 15)),
            "mode": data.get("mode", "rel"),
            "rel_thresh_db": float(data.get("rel_thresh_db", 10)),
            "silence_thresh_db": float(data.get("silence_thresh_db", -40)),
            "min_silence_ms": int(data.get("min_silence_ms", 500)),
            "interlude_min_s": float(data.get("interlude_min_s", 3.0)),
            "lyrics": str(data.get("lyrics", "") or "").strip(),
            "whisper_model": str(data.get("whisper_model", "funasr")
                                  or "funasr"),
            "song_lang": str(data.get("song_lang", "zh") or "zh"),
            "engine": str(data.get("engine", "auto") or "auto"),
        })
    t = threading.Thread(target=_run_split, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"ok": True})


def _run_split(job_id):
    with JOBS_LOCK:
        job = JOBS[job_id]
        params = {k: job[k] for k in
                  ("min_len_s", "max_len_s", "mode", "rel_thresh_db",
                   "silence_thresh_db", "min_silence_ms", "interlude_min_s",
                   "lyrics", "whisper_model", "song_lang", "engine")}
        path, out_dir = job["path"], job["out_dir"]

    def progress(msg, pct):
        with JOBS_LOCK:
            JOBS[job_id]["message"] = msg
            if pct is not None:
                JOBS[job_id]["progress"] = pct

    try:
        meta = {}
        segments = split_song(path, out_dir, progress=progress,
                              meta_out=meta, **params)
        lyric_txt = os.path.join(out_dir, "歌词_时间戳.txt")
        with JOBS_LOCK:
            JOBS[job_id].update({
                "status": "done", "progress": 100,
                "message": "完成", "segments": segments,
                "reset_boundaries": [0] + [s["end_ms"] for s in segments],
                "lyrics_txt": (os.path.basename(lyric_txt)
                               if os.path.isfile(lyric_txt) else None),
                "lyrics_align_cuts": meta.get("align_cuts"),
                "lyrics_lines": meta.get("lines"),
            })
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id].update({
                "status": "error", "message": "失败", "error": str(e),
            })


@app.route("/re-cut", methods=["POST"])
def re_cut():
    """手动微调后按新边界重新切割。boundaries = 升序毫秒数组(首项必须 0,
    末项为歌曲总长),相邻两项即为一段。用于:合并相邻段(删边界)或
    左右微调共享边界。"""
    data = request.get_json() or {}
    job_id = data.get("job_id", "")
    raw = data.get("boundaries")
    if not isinstance(raw, list) or len(raw) < 2:
        return jsonify({"ok": False, "error": "边界数据无效"}), 400
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "任务不存在"}), 404
        if job["status"] == "running":
            return jsonify({"ok": False, "error": "分割进行中,请等待"}), 400
        if job["lyrics_status"] == "extracting":
            return jsonify({"ok": False, "error": "歌词提取中,请等待"}), 400
        path, out_dir = job["path"], job["out_dir"]
        old = list(job.get("segments") or [])
        lyrics_text = job.get("lyrics") or ""
        lyric_name = job.get("lyrics_txt")
        align_cuts = job.get("lyrics_align_cuts")
        lyric_lines = job.get("lyrics_lines")
    try:
        bnds = sorted({int(round(float(b))) for b in raw})
        if bnds[0] < 0 or len(bnds) < 2:
            return jsonify({"ok": False, "error": "边界需以 0 起且至少两段"}), 400
        audio = load_audio(path)
        total = len(audio)
        if bnds[-1] > total:
            return jsonify({"ok": False, "error": f"边界超过歌曲总长 {total}ms"}), 400
        if bnds[-1] < total - 300:
            return jsonify({"ok": False, "error": "末边界应为歌曲结尾,请保留最后一段时长"}), 400
        bnds[-1] = total

        for p in old:
            if os.path.isfile(p.get("path", "")):
                try:
                    os.remove(p["path"])
                except OSError:
                    pass

        result = []
        for i in range(len(bnds) - 1):
            if bnds[i + 1] - bnds[i] < 1:
                continue
            seg = audio[bnds[i]:bnds[i + 1]]
            name = f"segment_{len(result) + 1:03d}.wav"
            seg.export(os.path.join(out_dir, name), format="wav")
            result.append({
                "index": len(result) + 1,
                "file": name,
                "path": os.path.join(out_dir, name),
                "start_ms": bnds[i],
                "end_ms": bnds[i + 1],
                "duration_s": round(len(seg) / 1000.0, 2),
            })

        if lyrics_text and lyric_name:
            try:
                lines = lyric_lines or [l.strip() for l in
                                        lyrics_text.splitlines() if l.strip()]
                if lyrics.is_timed_lyrics(lyrics_text) or align_cuts:
                    lyrics.write_timed_lyrics(
                        lyrics_text, lines, align_cuts or [], bnds,
                        os.path.join(out_dir, lyric_name))
            except Exception as e:
                sys.stderr.write(f"[带时间戳歌词更新失败] {e}\n")

        with JOBS_LOCK:
            JOBS[job_id].update({"segments": result})
        return jsonify({
            "ok": True,
            "segments": [{
                "index": s["index"], "file": s["file"],
                "start": fmt(s["start_ms"]), "end": fmt(s["end_ms"]),
                "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                "duration_s": s["duration_s"],
            } for s in result],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/extract-lyrics", methods=["POST"])
def extract_lyrics():
    """无人声歌词时,从歌曲提取歌词(BSRoformer 提人声 + 所选引擎识别)。
    识别完成后用户可在页面上修改,再按歌词分割。"""
    data = request.get_json() or {}
    job_id = data.get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "任务不存在"}), 404
        if job["lyrics_status"] == "extracting":
            return jsonify({"ok": False, "error": "歌词提取中,请稍候"}), 400
        if job["status"] == "running":
            return jsonify({"ok": False, "error": "分割进行中,请等待"}), 400
        job["lyrics_status"] = "extracting"
        job["lyrics_error"] = None
        job["extracted_lyrics"] = ""
        job["whisper_model"] = str(data.get("whisper_model", "funasr")
                                   or "funasr")
        job["song_lang"] = str(data.get("song_lang", "zh") or "zh")
    t = threading.Thread(target=_run_extract, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"ok": True})


def _run_extract(job_id):
    with JOBS_LOCK:
        job = JOBS[job_id]
        path, out_dir = job["path"], job["out_dir"]
        model = job.get("whisper_model", "funasr") or "funasr"
        song_lang = str(job.get("song_lang", "zh") or "zh")
    if song_lang != "zh" and model == "funasr":
        model = "large-v3"  # 英文歌统一走 whisper(Nano/Paraformer 仅中文)

    def progress(msg, pct):
        with JOBS_LOCK:
            JOBS[job_id]["message"] = msg
            if pct is not None:
                JOBS[job_id]["progress"] = pct

    try:
        device = torch_device()
        if device == "cuda":
            progress("BSRoformer 分离人声(GPU 加速,请稍候)...", 15)
        else:
            progress("BSRoformer 分离人声(CPU 处理,较慢,请稍候)...", 15)
        vocals_path = separate_vocals(path, out_dir, device=device)
        progress("识别歌词(GPU)...", 45)
        lines = lyrics.extract_lyrics(vocals_path, model=model, device=device,
                                      language=song_lang,
                                      progress=progress)
        with JOBS_LOCK:
            JOBS[job_id].update({
                "lyrics_status": "done",
                "lyrics_error": None,
                "extracted_lyrics": "\n".join(lines),
                "progress": 100,
                "message": "歌词提取完成,请核对修改标点",
            })
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id].update({
                "lyrics_status": "error", "lyrics_error": str(e),
                "message": "歌词提取失败",
            })


@app.route("/model-status")
def model_status():
    """检查 BSRoformer 模型是否已就绪 + 计算设备(GPU/CPU)。"""
    cache = os.path.join(BASE, "runtime", "models", "pymss")
    ready = False
    if os.path.isdir(cache):
        for root, _, files in os.walk(cache):
            if any(f.endswith(".ckpt") for f in files):
                ready = True
                break
    info = {
        "ready": ready,
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass
    return jsonify(info)


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "notfound"})
        return jsonify({
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "error": job["error"],
            "name": job["name"],
            "lyrics_status": job.get("lyrics_status", "idle"),
            "lyrics_error": job.get("lyrics_error"),
            "extracted_lyrics": job.get("extracted_lyrics", ""),
            "lyrics_txt": job.get("lyrics_txt"),
            "reset_boundaries": job.get("reset_boundaries"),
            "segments": [{
                "index": s["index"], "file": s["file"],
                "start": fmt(s["start_ms"]), "end": fmt(s["end_ms"]),
                "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                "duration_s": s["duration_s"],
            } for s in job["segments"]],
        })


@app.route("/audio/<job_id>/<int:idx>")
def audio(job_id, idx):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or idx < 1 or idx > len(job["segments"]):
            return "not found", 404
        seg = job["segments"][idx - 1]
    return send_file(seg["path"], mimetype="audio/wav")


@app.route("/download/<job_id>/<int:idx>")
def download(job_id, idx):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or idx < 1 or idx > len(job["segments"]):
            return "not found", 404
        seg = job["segments"][idx - 1]
    return send_file(seg["path"], as_attachment=True,
                     download_name=seg["file"])


@app.route("/download/<job_id>/all")
def download_all(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or not job["segments"]:
            return "not found", 404
        segments = list(job["segments"])
        name = job["name"]
        lyric_txt = job.get("lyrics_txt")
        lyric_path = (os.path.join(job["out_dir"], lyric_txt)
                      if lyric_txt else None)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for seg in segments:
            z.write(seg["path"], arcname=seg["file"])
        if lyric_path and os.path.isfile(lyric_path):
            z.write(lyric_path, arcname=os.path.basename(lyric_path))
    buf.seek(0)
    base = os.path.splitext(name)[0]
    return send_file(buf, as_attachment=True,
                     download_name=f"{base}_分割结果.zip",
                     mimetype="application/zip")


@app.route("/download/<job_id>/lyrics")
def download_lyrics(job_id):
    """下载带时间戳歌词 txt(仅歌词模式分割后生成)。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return "not found", 404
        lyric_txt = job.get("lyrics_txt")
        if not lyric_txt:
            return "not found", 404
        path = os.path.join(job["out_dir"], lyric_txt)
    if not os.path.isfile(path):
        return "not found", 404
    return send_file(path, as_attachment=True,
                     download_name="歌词_时间戳.txt",
                     mimetype="text/plain")


@app.route("/open-dir", methods=["POST"])
def open_dir():
    """在资源管理器里打开该任务的输出目录(uploads\\<job_id>)。"""
    data = request.get_json(silent=True) or {}
    with JOBS_LOCK:
        job = JOBS.get(data.get("job_id", ""))
        out_dir = job["out_dir"] if job else None
    if not out_dir:
        return jsonify({"ok": False, "error": "任务不存在"}), 404
    try:
        os.makedirs(out_dir, exist_ok=True)
        os.startfile(out_dir)          # noqa: S606  仅 Windows,交给系统打开
    except AttributeError:
        return jsonify({"ok": False, "error": "当前系统不支持自动打开目录"}), 501
    except OSError as exc:
        return jsonify({"ok": False, "error": "打开失败:%s" % exc}), 500
    return jsonify({"ok": True, "dir": out_dir})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)