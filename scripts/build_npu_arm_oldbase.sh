#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE_TAG="${IMAGE_TAG:-funasr-leader-asr:npu-ascend-oldbase-20260604}"
BASE_IMAGE="${BASE_IMAGE:-ascendai/pytorch:2.1.0-ubuntu22.04}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.1.0.post10}"
PACKAGE_DIR="${PACKAGE_DIR:-deploy/ascend-npu-oldbase-20260604}"

echo "Building NPU ARM image"
echo "  image: $IMAGE_TAG"
echo "  base : $BASE_IMAGE"
echo "  torch-npu: $TORCH_NPU_VERSION"

docker buildx build \
  --platform linux/arm64 \
  --provenance=false \
  --sbom=false \
  --build-arg "NPU_BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "TORCH_NPU_VERSION=$TORCH_NPU_VERSION" \
  -f docker/Dockerfile.npu-builtin \
  -t "$IMAGE_TAG" \
  --load \
  .

docker image inspect "$IMAGE_TAG" >/dev/null

image_safe_name="$(printf '%s' "$IMAGE_TAG" | sed 's#[:/]#-#g')"
out_dir="$ROOT/$PACKAGE_DIR"
image_dir="$out_dir/image"
mkdir -p "$image_dir"

image_tar="$image_dir/$image_safe_name.tar"
docker save -o "$image_tar" "$IMAGE_TAG"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$out_dir" && sha256sum "image/$image_safe_name.tar" > SHA256SUMS.txt)
fi

package_name="$(basename "$out_dir").tar"
rm -f "$ROOT/$package_name"
tar -cf "$ROOT/$package_name" -C "$(dirname "$out_dir")" "$(basename "$out_dir")"

echo
echo "Done"
echo "  image tar : $image_tar"
echo "  package   : $ROOT/$package_name"
echo "  deploy dir: $out_dir"
