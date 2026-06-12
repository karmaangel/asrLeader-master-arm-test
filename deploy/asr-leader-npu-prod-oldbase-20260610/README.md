# ASR Leader NPU production oldbase package

This package is a production oldbase overlay. It contains the new app code and
the local Qwen2.5-1.5B-Instruct model, but it intentionally does not contain
the working production base image.

The production base image `meeting-asr:npu-new-oldbase` exists only on the
Ascend production worker. Build the final image on that worker so Python,
FunASR, torch, torch-npu and CANN stay identical to the working production
container.

Target image:

```bash
funasr-leader-asr:npu-prod-oldbase-20260610
```

Known working production versions:

- Python 3.11.6
- funasr 1.3.1
- modelscope 1.36.0
- torch 2.1.0
- torch-npu 2.1.0.post10
- fastapi 0.136.0
- pydantic 2.9.2
- transformers 4.44.0

Bundled model:

- `/app/models/Qwen2.5-1.5B-Instruct`

Build on the production Ascend worker that has `meeting-asr:npu-new-oldbase`:

```bash
tar -xf asr-leader-npu-prod-oldbase-20260610.tar
cd asr-leader-npu-prod-oldbase-20260610
chmod 755 scripts/build_npu_prod_oldbase_on_prod.sh
IMAGE_TAG=funasr-leader-asr:npu-prod-oldbase-20260610 bash scripts/build_npu_prod_oldbase_on_prod.sh
```

Then update only the test/new deployment from master:

```bash
kubectl -n test set image deployment/asr-leader-npu asr-leader-npu=funasr-leader-asr:npu-prod-oldbase-20260610
kubectl -n test rollout status deployment/asr-leader-npu --timeout=10m
kubectl -n test logs deployment/asr-leader-npu --tail=200 --timestamps
```

Do not modify these running production services:

- `dream-acr`
- `dream-acr-new`

Important NPU environment values for the deployment:

```text
ASR_DEVICE=npu:0
ASR_RESOLVE_LOCAL_MODELS=false
POSTPROCESS_PRELOAD=true
ASR_NORMALIZE_ASCEND_DEVICES=false
MODELSCOPE_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

`ASR_RESOLVE_LOCAL_MODELS=false` is required for FunASR 1.3.1. The previous
failed image resolved `/app/models/paraformer-zh` as a local path and FunASR
1.3.1 raised `/app/models/paraformer-zh is not registered`.

`POSTPROCESS_PRELOAD=true` is safe in this package because Qwen is bundled. If
the model directory is removed, the entrypoint automatically switches preload
off before Python starts.
