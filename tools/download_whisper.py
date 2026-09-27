#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预下载 whisper 模型到 runtime/models/whisper/(断点续传,断线重跑即可)。

用法(runtime/python/python.exe):
  tools/download_whisper.py medium
  tools/download_whisper.py medium large-v3            # 多个模型
  tools/download_whisper.py medium --base-url https://镜像/whisper   # 走镜像
不带 --base-url 时走官方 azure CDN;断线后重跑同一命令会续传。
"""
import argparse
import sys
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

from lyrics import _fetch_whisper_model, _WHISPER_URLS  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="模型名: tiny/base/small/medium/large-v1/large-v2/large-v3")
    ap.add_argument("--base-url", default=None, help="镜像前缀,URL 拼成 <base-url>/<model>.pt")
    args = ap.parse_args()
    for m in args.models:
        if m not in _WHISPER_URLS:
            print("未知模型: %s(可选 %s)" % (m, ", ".join(_WHISPER_URLS)))
            sys.exit(1)
        print("下载 %s ..." % m)
        _fetch_whisper_model(m, base_url=(None if args.base_url is None
                                          else args.base_url))
        path = os.path.join(_BASE, "runtime", "models", "whisper", m + ".pt")
        print("ok: %s (%.2f MB)" % (path, os.path.getsize(path) / 1e6))