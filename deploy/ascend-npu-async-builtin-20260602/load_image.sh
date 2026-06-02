#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
docker load -i image/funasr-leader-asr-npu-async-builtin-20260602.tar
