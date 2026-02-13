# Third Endpoint (CrewAI, TGI)

RunPod serverless endpoint for `Qwen/Qwen3-32B-Instruct` on HuggingFace TGI.

## Build

```bash
./scripts/build.sh
```

## Push

```bash
./scripts/push.sh
```

## RunPod settings

- GPU: NVIDIA A40
- Workers: min `0`, max `1`
- Idle timeout: `120s`
- Env:
  - `MODEL_ID=Qwen/Qwen3-32B-Instruct`
  - `MAX_INPUT_LENGTH=4096`
  - `MAX_TOTAL_TOKENS=8192`
  - `QUANTIZE=awq`
  - `HF_TOKEN=<token>`
