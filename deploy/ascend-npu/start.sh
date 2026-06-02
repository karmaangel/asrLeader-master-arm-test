#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! docker image inspect funasr-leader-asr:npu-ascend-20260601 >/dev/null 2>&1; then
  ./load_image.sh
fi

mkdir -p data
docker compose up -d
docker compose logs -f --tail=100
