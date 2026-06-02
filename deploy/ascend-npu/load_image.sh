#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f image/funasr-leader-asr-npu-ascend-20260601.tar ]; then
  echo "missing image/funasr-leader-asr-npu-ascend-20260601.tar" >&2
  exit 1
fi

docker load -i image/funasr-leader-asr-npu-ascend-20260601.tar
