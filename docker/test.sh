#!/usr/bin/env bash
set -euo pipefail

REPO="/home/david-james/Desktop/04-MDPI/MDPI"
DOCKERFILE="$REPO/docker/Dockerfile"
IMAGE_TAG="mdpi-pipeline:latest"

HOST_INPUT="/home/david-james/Desktop/04-MDPI/MDPI/profiles/Project_Example/20230424/night/E07_01"
CONTAINER_INPUT="/data/input"
MODEL_IN_CONTAINER="/app/model"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

echo "[BUILD] Building $IMAGE_TAG from $DOCKERFILE ..."
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" "$REPO"

echo "[RUN] Running container with default entrypoint and provided args ..."
# Pipe a 'Y' to accept the default configuration prompt in run_pipeline.py
printf "Y\n" | docker run --rm -i \
  --user "$HOST_UID":"$HOST_GID" \
  -e MPLCONFIGDIR=/tmp -e HOME=/tmp \
  -v "$HOST_INPUT":"$CONTAINER_INPUT":rw \
  "$IMAGE_TAG" \
  -i "$CONTAINER_INPUT" -m "$MODEL_IN_CONTAINER"
