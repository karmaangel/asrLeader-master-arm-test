# ASR Leader NPU oldbase package

Image:

```bash
funasr-leader-asr:npu-ascend-oldbase-20260604
```

Build on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_npu_arm_oldbase.ps1
```

The default base image is `ascendai/pytorch:2.1.0-ubuntu22.04`, built as `linux/arm64`.
It pins `torch==2.1.0`, `torchaudio==2.1.0`, and `torch-npu==2.1.0.post10`.
It does not depend on `meeting-asr:npu-new-oldbase` being present locally.

Build on Linux:

```bash
bash scripts/build_npu_arm_oldbase.sh
```

Run with Docker Compose on an Ascend node:

```bash
tar -xf ascend-npu-oldbase-20260604.tar
cd ascend-npu-oldbase-20260604
./start.sh
```

K8s offline update:

Upload `ascend-npu-oldbase-20260604.tar` to the Ascend node that will run the pod
or to every candidate Ascend node, then load the image:

```bash
tar -xf ascend-npu-oldbase-20260604.tar
cd ascend-npu-oldbase-20260604
sha256sum -c SHA256SUMS.txt
./load_image.sh
nerdctl -n k8s.io images | grep 'funasr-leader-asr'
```

Then update the existing deployment:

```bash
kubectl -n test set image deployment/asr-leader-npu asr-leader-npu=funasr-leader-asr:npu-ascend-oldbase-20260604
kubectl -n test rollout status deployment/asr-leader-npu
```

NPU smoke test after rollout:

```bash
pod=$(kubectl -n test get pod -l app=asr-leader-npu -o jsonpath='{.items[0].metadata.name}')
kubectl -n test exec "$pod" -- python - <<'PY'
import torch, torch_npu
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
print("available", torch.npu.is_available(), "count", torch.npu.device_count())
torch.npu.set_device(0)
print(torch.ones(1).npu())
PY
```
