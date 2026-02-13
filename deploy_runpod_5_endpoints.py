#!/usr/bin/env python3
"""Deploy and smoke-test 5 RunPod endpoints for Elysium."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import runpod
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / "Elysium" / ".env")


@dataclass(frozen=True)
class EndpointSpec:
    key: str
    repo: str
    image: str
    template_name: str
    endpoint_name: str
    gpu_id: str
    workers_min: int
    workers_max: int
    idle_timeout: int
    env: dict[str, str]
    test_payload: dict[str, Any]


SPECS: list[EndpointSpec] = [
    EndpointSpec(
        key="first",
        repo="Aminatorex/First-Endpoint",
        image="docker.io/aminatorex/first-endpoint:latest",
        template_name="elysium-first-endpoint-v1",
        endpoint_name="elysium-first-endpoint",
        gpu_id="AMPERE_48",  # A40 48GB
        workers_min=0,
        workers_max=2,
        idle_timeout=60,
        env={
            "MODEL_NAME": "Qwen/Qwen3-14B-Instruct",
            "MAX_MODEL_LEN": "8192",
            "QUANTIZATION": "awq",
            "TENSOR_PARALLEL_SIZE": "1",
            "GPU_MEMORY_UTILIZATION": "0.95",
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
        test_payload={
            "input": {
                "prompt": "Write a short friendly Instagram comment about a sunset photo.",
                "max_tokens": 64,
                "temperature": 0.7,
            }
        },
    ),
    EndpointSpec(
        key="second",
        repo="Aminatorex/Second-Endpoint",
        image="docker.io/aminatorex/second-endpoint:latest",
        template_name="elysium-second-endpoint-v1",
        endpoint_name="elysium-second-endpoint",
        gpu_id="AMPERE_48",  # A40 48GB (safe GPU code for SDK/API compatibility)
        workers_min=0,
        workers_max=2,
        idle_timeout=90,
        env={
            "MODEL_NAME": "Qwen/Qwen3-VL-8B-Instruct",
            "MAX_MODEL_LEN": "4096",
            "TRUST_REMOTE_CODE": "True",
            "GPU_MEMORY_UTILIZATION": "0.9",
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
        test_payload={
            "input": {
                "prompt": "Describe what the image shows in one sentence.",
                # tiny 1x1 transparent png base64
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9oN7LxkAAAAASUVORK5CYII=",
                "max_tokens": 32,
            }
        },
    ),
    EndpointSpec(
        key="third",
        repo="Aminatorex/Third-Endpoint",
        image="docker.io/aminatorex/third-endpoint:latest",
        template_name="elysium-third-endpoint-v1",
        endpoint_name="elysium-third-endpoint",
        gpu_id="AMPERE_48",  # A40 48GB
        workers_min=0,
        workers_max=1,
        idle_timeout=120,
        env={
            "MODEL_ID": "Qwen/Qwen3-32B-Instruct",
            "MAX_INPUT_LENGTH": "4096",
            "MAX_TOTAL_TOKENS": "8192",
            "QUANTIZE": "awq",
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
        test_payload={
            "input": {
                "prompt": "Analyze briefly whether this posting pattern seems risky.",
                "max_new_tokens": 64,
                "temperature": 0.5,
            }
        },
    ),
    EndpointSpec(
        key="fourth",
        repo="Aminatorex/Fourth-Endpoint",
        image="docker.io/aminatorex/fourth-endpoint:latest",
        template_name="elysium-fourth-endpoint-v1",
        endpoint_name="elysium-fourth-endpoint",
        gpu_id="AMPERE_48",  # A40 48GB (safe GPU code for SDK/API compatibility)
        workers_min=0,
        workers_max=2,
        idle_timeout=60,
        env={
            "MODEL_PATH": "modularai/Qwen3-4B-Instruct-GGUF",
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
        test_payload={
            "input": {
                "prompt": "Write a very short fallback comment.",
                "max_tokens": 32,
                "temperature": 0.7,
            }
        },
    ),
    EndpointSpec(
        key="fifth",
        repo="Aminatorex/Fifth-Endpoint",
        image="docker.io/aminatorex/fifth-endpoint:latest",
        template_name="elysium-fifth-endpoint-v1",
        endpoint_name="elysium-fifth-endpoint",
        gpu_id="AMPERE_80",  # A100 80GB
        workers_min=0,
        workers_max=1,
        idle_timeout=180,
        env={
            "MODEL_NAME": "browser-use/bu-30b-a3b-preview",
            "MAX_MODEL_LEN": "65536",
            "GPU_MEMORY_UTILIZATION": "0.90",
            "TRUST_REMOTE_CODE": "True",
            "DTYPE": "float16",
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        },
        test_payload={
            "input": {
                "prompt": "Find clickable controls on the page.",
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9oN7LxkAAAAASUVORK5CYII=",
                "task_type": "dom_analysis",
                "max_tokens": 64,
                "temperature": 0.6,
            }
        },
    ),
]


def _require_api_key() -> str:
    api_key = os.getenv("RUNPOD_API_KEY", "")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is required")
    runpod.api_key = api_key
    return api_key


def _existing_endpoint_by_name(name: str) -> dict[str, Any] | None:
    data = runpod.get_endpoints() or {}
    endpoints = data.get("myself", {}).get("endpoints", [])
    for ep in endpoints:
        if ep.get("name") == name:
            return ep
    return None


def _create_template(spec: EndpointSpec) -> dict[str, Any]:
    return runpod.create_template(
        name=spec.template_name,
        image_name=spec.image,
        is_serverless=True,
        container_disk_in_gb=50 if spec.key == "fifth" else 30,
        env=spec.env,
    )


def _create_endpoint(spec: EndpointSpec, template_id: str) -> dict[str, Any]:
    return runpod.create_endpoint(
        name=spec.endpoint_name,
        template_id=template_id,
        gpu_ids=spec.gpu_id,
        workers_min=spec.workers_min,
        workers_max=spec.workers_max,
        idle_timeout=spec.idle_timeout,
        scaler_type="QUEUE_DELAY",
        scaler_value=4,
    )


def _runsync(endpoint_id: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    ep = runpod.Endpoint(endpoint_id)
    return ep.run_sync(payload, timeout=timeout)


def deploy_all(skip_tests: bool) -> dict[str, Any]:
    _require_api_key()

    report: dict[str, Any] = {"created": [], "existing": [], "tests": []}

    for spec in SPECS:
        existing = _existing_endpoint_by_name(spec.endpoint_name)
        if existing:
            endpoint_id = existing["id"]
            report["existing"].append({"key": spec.key, "endpoint_id": endpoint_id, "name": spec.endpoint_name})
        else:
            tmpl = _create_template(spec)
            ep = _create_endpoint(spec, tmpl["id"])
            endpoint_id = ep["id"]
            report["created"].append(
                {
                    "key": spec.key,
                    "endpoint_id": endpoint_id,
                    "name": spec.endpoint_name,
                    "template_id": tmpl["id"],
                }
            )

        os.environ[f"RUNPOD_{spec.key.upper()}_ENDPOINT_ID"] = endpoint_id

        if not skip_tests:
            # Give endpoint short warm-up window.
            time.sleep(5)
            try:
                result = _runsync(endpoint_id, spec.test_payload, timeout=240)
                report["tests"].append(
                    {
                        "key": spec.key,
                        "endpoint_id": endpoint_id,
                        "ok": True,
                        "result_preview": str(result)[:300],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                report["tests"].append(
                    {
                        "key": spec.key,
                        "endpoint_id": endpoint_id,
                        "ok": False,
                        "error": str(exc),
                    }
                )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy 5 RunPod endpoints and smoke test")
    parser.add_argument("--skip-tests", action="store_true", help="Skip runsync smoke tests")
    parser.add_argument("--out", default=str(ROOT / "runpod_5_endpoints_report.json"), help="Output report path")
    args = parser.parse_args()

    report = deploy_all(skip_tests=args.skip_tests)

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
