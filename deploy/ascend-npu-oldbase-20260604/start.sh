#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data

if ! docker image inspect funasr-leader-asr:npu-ascend-oldbase-20260604 >/dev/null 2>&1; then
  ./load_image.sh
fi

docker compose up -d
docker logs -f asr-leader-npu
