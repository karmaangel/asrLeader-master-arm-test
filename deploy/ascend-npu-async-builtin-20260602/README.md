# Ascend NPU offline package

This package is for an offline Ascend ARM64 server.

Run on the server:

```bash
tar -xf asr-leader-npu-async-builtin-20260602.tar
cd ascend-npu-async-builtin-20260602
chmod +x *.sh
./start.sh
```

Health check:

```bash
./health_check.sh
```

The image includes ASR models and Qwen2.5-1.5B. It does not download models at startup.

Long audio correction uses the async correction strategy:

- `correct_text=false` skips Qwen correction.
- default uses correction.
- long text returns `correction_task_id`.
- query `/corrections/{task_id}` for async correction result.
