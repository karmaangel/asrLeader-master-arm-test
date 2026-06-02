# Ascend NPU deployment

1. Copy this directory to the Ascend server.
2. Run `chmod +x *.sh`.
3. Run `./start.sh`.
4. Check `http://SERVER_IP:8000/docs` or `http://SERVER_IP:8002/docs`.

The package expects an Ascend host with Docker, Docker Compose, NPU driver devices under `/dev`, and Ascend driver/toolkit installed under `/usr/local/Ascend`.

The service uses:

- `ASR_DEVICE=npu:0`
- `POSTPROCESS_DEVICE=npu:0`
- local ASR models in `./models`
- local Qwen model in `./models/Qwen2.5-1.5B-Instruct`
