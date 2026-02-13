#!/usr/bin/env bash
set -euo pipefail

ENDPOINT_ID="${1:-}"
API_KEY="${RUNPOD_API_KEY:-}"

if [[ -z "$ENDPOINT_ID" || -z "$API_KEY" ]]; then
  echo "Usage: RUNPOD_API_KEY=... $0 <endpoint_id>"
  exit 1
fi

curl -sS -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input":{"prompt":"health check","max_tokens":32}}' | jq .
