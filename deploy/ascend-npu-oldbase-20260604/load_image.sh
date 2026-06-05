#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
image_tar="$(find image -maxdepth 1 -type f -name '*.tar' | head -1)"

if [ -z "${image_tar:-}" ]; then
  echo "missing image/*.tar" >&2
  exit 1
fi

if command -v nerdctl >/dev/null 2>&1; then
  nerdctl -n k8s.io load -i "$image_tar"
elif command -v docker >/dev/null 2>&1; then
  docker load -i "$image_tar"
else
  echo "missing nerdctl or docker" >&2
  exit 1
fi
