#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-meeting-asr:npu-new-oldbase}"
IMAGE_TAG="${IMAGE_TAG:-funasr-leader-asr:npu-prod-oldbase-20260610}"
NERDCTL_NAMESPACE="${NERDCTL_NAMESPACE:-k8s.io}"
BUILD_CONTAINER="${BUILD_CONTAINER:-asr-prod-oldbase-build}"

run_nerdctl() {
  nerdctl -n "${NERDCTL_NAMESPACE}" "$@"
}

cleanup() {
  run_nerdctl rm -f "${BUILD_CONTAINER}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

cd "${ROOT}"

if ! command -v nerdctl >/dev/null 2>&1; then
  echo "nerdctl is required on the production Ascend node" >&2
  exit 1
fi

if ! run_nerdctl image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "missing base image in namespace ${NERDCTL_NAMESPACE}: ${BASE_IMAGE}" >&2
  echo "Run this on the worker that already has the production image." >&2
  exit 1
fi

cleanup

echo "Base image : ${BASE_IMAGE}"
echo "Target tag : ${IMAGE_TAG}"
echo "Namespace  : ${NERDCTL_NAMESPACE}"

run_nerdctl run -d \
  --net none \
  --name "${BUILD_CONTAINER}" \
  --entrypoint /bin/sh \
  "${BASE_IMAGE}" \
  -c "sleep 86400" >/dev/null

run_nerdctl exec "${BUILD_CONTAINER}" mkdir -p /app/app /app/models /data
run_nerdctl cp "${ROOT}/app/." "${BUILD_CONTAINER}:/app/app"
run_nerdctl cp "${ROOT}/docker/entrypoint.npu.sh" "${BUILD_CONTAINER}:/usr/local/bin/asr-npu-entrypoint.sh"
run_nerdctl exec "${BUILD_CONTAINER}" chmod 755 /usr/local/bin/asr-npu-entrypoint.sh

if [ -d "${ROOT}/models/Qwen2.5-1.5B-Instruct" ]; then
  echo "Copying bundled Qwen model"
  run_nerdctl cp "${ROOT}/models/Qwen2.5-1.5B-Instruct" "${BUILD_CONTAINER}:/app/models/Qwen2.5-1.5B-Instruct"
else
  echo "No bundled Qwen model found; startup will not preload Qwen."
fi

run_nerdctl exec "${BUILD_CONTAINER}" python -m compileall -q /app/app

run_nerdctl commit \
  --change 'ENTRYPOINT ["/usr/local/bin/asr-npu-entrypoint.sh"]' \
  --change 'CMD ["python","/app/app/server.py"]' \
  "${BUILD_CONTAINER}" \
  "${IMAGE_TAG}"

echo
echo "Built image:"
run_nerdctl image inspect "${IMAGE_TAG}" | grep -A 3 -E '"Entrypoint"|"Cmd"' || true

echo
echo "Next on master:"
echo "  kubectl -n test set image deployment/asr-leader-npu asr-leader-npu=${IMAGE_TAG}"
echo "  kubectl -n test rollout status deployment/asr-leader-npu --timeout=10m"
