#!/usr/bin/env bash
# Build & tag local docker image for quick testing

set -euo pipefail

IMAGE_NAME=${1:-face-recognition:local}
DOCKERFILE=${2:-Dockerfile}

echo "Building $IMAGE_NAME using $DOCKERFILE"

docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

echo "Built $IMAGE_NAME"

# Optionally print image size
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep "${IMAGE_NAME%%:*}" || true
