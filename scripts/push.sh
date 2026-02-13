#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="ghcr.io/aminatorex/third-endpoint"
VERSION="v0.16.0"

docker push "${IMAGE_NAME}:${VERSION}"
docker push "${IMAGE_NAME}:latest"
echo "Push complete: ${IMAGE_NAME}:${VERSION}"
