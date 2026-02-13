#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="docker.io/aminatorex/third-endpoint"
VERSION="v0.16.0"

docker build -t "${IMAGE_NAME}:${VERSION}" .
docker tag "${IMAGE_NAME}:${VERSION}" "${IMAGE_NAME}:latest"
echo "Build complete: ${IMAGE_NAME}:${VERSION}"
