import os
import subprocess
import threading
import time
from typing import Any, Dict

import requests
import runpod


def start_tgi_server() -> None:
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-32B-Instruct")
    cmd = [
        "text-generation-launcher",
        "--model-id", model_id,
        "--quantize", os.getenv("QUANTIZE", "awq"),
        "--max-input-length", os.getenv("MAX_INPUT_LENGTH", "4096"),
        "--max-total-tokens", os.getenv("MAX_TOTAL_TOKENS", "8192"),
        "--port", "80",
        "--hostname", "0.0.0.0",
    ]
    subprocess.Popen(cmd)


threading.Thread(target=start_tgi_server, daemon=True).start()
time.sleep(60)

TGI_URL = "http://localhost:80/generate"


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = job.get("input", {})
        payload = {
            "inputs": data.get("prompt", ""),
            "parameters": {
                "max_new_tokens": data.get("max_new_tokens", 512),
                "temperature": data.get("temperature", 0.5),
                "top_p": data.get("top_p", 0.95),
                "do_sample": True,
            },
        }
        response = requests.post(TGI_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict):
            generated = result.get("generated_text", "")
        else:
            generated = result[0].get("generated_text", "")
        return {"output": generated, "status": "success"}
    except Exception as exc:
        return {"error": str(exc), "status": "error"}


runpod.serverless.start({"handler": handler})
