#!/usr/bin/env python3
"""Pod smoke-test for the reasoning endpoint image."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure repo root (where handler.py lives) is importable even if CWD != /app.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _configure_cache_dirs() -> dict[str, str]:
    """Point HF/vLLM caches at the mounted network volume (if available)."""
    volume_root = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
    if not os.path.isdir(volume_root):
        return {}

    cache_root = os.path.join(volume_root, "cache")
    hf_home = os.getenv("HF_HOME") or os.path.join(cache_root, "hf")
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or os.path.join(hf_home, "hub")
    vllm_cache = os.getenv("VLLM_CACHE_ROOT") or os.path.join(cache_root, "vllm")

    os.makedirs(hub_cache, exist_ok=True)
    os.makedirs(vllm_cache, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("VLLM_CACHE_ROOT", vllm_cache)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    return {
        "volume_root": volume_root,
        "hub_cache": hub_cache,
        "vllm_cache": vllm_cache,
    }


def _default_result_path(model_name: str) -> str:
    explicit = os.getenv("SMOKE_RESULT_PATH", "").strip()
    if explicit:
        return explicit

    volume_root = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
    if not os.path.isdir(volume_root):
        return ""

    base = os.path.join(volume_root, "smoke-results")
    os.makedirs(base, exist_ok=True)

    safe_model = model_name.replace("/", "__")
    key = os.getenv("SMOKE_KEY", "").strip()
    ts = os.getenv("SMOKE_TS", "").strip() or str(int(time.time()))
    fname = f"{key + '_' if key else ''}{safe_model}_{ts}.json"
    return os.path.join(base, fname)


def _write_result(path: str, data: dict[str, Any]) -> None:
    if not path:
        return

    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp, path)


def _maybe_prefetch(model_name: str) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        print("[pod_smoke] huggingface_hub not available; skipping prefetch", flush=True)
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"[pod_smoke] prefetch start model={model_name}", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=model_name,
        token=token,
        resume_download=True,
    )
    print(f"[pod_smoke] prefetch done in {time.time()-t0:.1f}s", flush=True)


def main() -> int:
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-32B-AWQ")
    caches = _configure_cache_dirs()
    result_path = _default_result_path(model_name)

    started = time.time()
    result: dict[str, Any] = {
        "status": "unknown",
        "model_name": model_name,
        "started_at_unix": started,
        "caches": caches,
        "result_path": result_path,
    }

    try:
        _maybe_prefetch(model_name)

        import handler as h

        print("[pod_smoke] initializing model...", flush=True)
        t0 = time.time()
        h.initialize_model()
        print(f"[pod_smoke] model initialized in {time.time()-t0:.1f}s", flush=True)

        payload = {
            "input": {
                "prompt": "In one sentence: what is 2+2?",
                "max_new_tokens": 16,
                "temperature": 0.1,
            }
        }
        print("[pod_smoke] running inference...", flush=True)
        out = h.handler(payload)
        print("[pod_smoke] output:", out, flush=True)

        ok = isinstance(out, dict) and out.get("status") == "success"
        result["status"] = "success" if ok else "error"
        result["handler_output"] = out
        return 0 if ok else 2
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=30)
        print("[pod_smoke] ERROR:", str(exc), flush=True)
        return 2
    finally:
        result["finished_at_unix"] = time.time()
        result["duration_s"] = round(result["finished_at_unix"] - started, 3)
        _write_result(result_path, result)
        if result_path:
            print(f"[pod_smoke] wrote result to {result_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
