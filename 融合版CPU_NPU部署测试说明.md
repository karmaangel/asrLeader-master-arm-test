# 融合版 ASR + Qwen 后处理部署测试说明

本文档用于交接同事编译 NPU 镜像，并分别测试 CPU、NPU 版本。

当前版本已将“ASR 转写 + 领导声纹识别 + Qwen 上下文纠错”融合到同一个服务进程中，不再依赖单独的 Ollama 容器，也不需要额外开放内部端口。

## 1. 代码目录

项目目录：

```bash
D:/CodeSpace/Asr/asr-002/asrLeader-master
```

关键文件：

```text
app/main.py                 接口入口，/transcribe 最后追加 post_process 后处理
app/post_processor.py       Qwen 后处理逻辑
app/config.py               ASR 和后处理配置
docker/Dockerfile.cpu       CPU 镜像
docker/Dockerfile.npu       昇腾 NPU 镜像
requirements.txt            CPU/GPU 依赖
requirements-npu.txt        NPU 依赖，torch/torch_npu 由基础镜像提供
```

## 2. 模型目录要求

容器内统一使用：

```text
/models
```

ASR 模型目录中至少包含：

```text
/models/paraformer-zh
/models/fsmn-vad
/models/ct-punc
/models/cam++
```

Qwen 后处理模型建议放到：

```text
/models/Qwen2.5-1.5B-Instruct
```

也可以用环境变量显式指定：

```bash
POSTPROCESS_MODEL_DIR=/models/Qwen2.5-1.5B-Instruct
```

离线环境必须设置 `POSTPROCESS_MODEL_DIR`，否则程序会尝试通过 ModelScope 下载：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

## 3. 主要环境变量

```bash
ASR_DEVICE=cpu              # CPU 测试
ASR_DEVICE=npu:0            # NPU 测试

POSTPROCESS_ENABLED=true
POSTPROCESS_PRELOAD=true    # 建议测试时开启，启动阶段直接加载 Qwen，能提前发现问题
POSTPROCESS_MODEL=Qwen/Qwen2.5-1.5B-Instruct
POSTPROCESS_MODEL_DIR=/models/Qwen2.5-1.5B-Instruct
POSTPROCESS_DEVICE=cpu      # CPU 后处理
POSTPROCESS_DEVICE=npu:0    # NPU 后处理
POSTPROCESS_MAX_CHARS=1200
POSTPROCESS_MAX_NEW_TOKENS=512
```

说明：

- `post_process=true` 是 `/transcribe` 的默认行为。
- 若后处理失败，会保留原 ASR 文本，不影响主转写链路。
- 如果某段文本被修正，会保留 `raw_text`，并增加 `post_processed: true`。

## 4. CPU 镜像编译

在项目根目录执行：

```bash
cd /path/to/asrLeader-master
docker build -f docker/Dockerfile.cpu -t funasr-leader-asr:cpu .
```

如果在 Windows 本机测试，也可以使用 compose：

```powershell
cd D:\CodeSpace\Asr\asr-002\asrLeader-master
docker compose build asr-leader-cpu
```

## 5. CPU 容器运行

Linux 示例：

```bash
docker run -d \
  --name asr-leader-cpu \
  -p 8000:8000 \
  -e ASR_DEVICE=cpu \
  -e MODELS_DIR=/models \
  -e DATA_DIR=/data \
  -e POSTPROCESS_ENABLED=true \
  -e POSTPROCESS_PRELOAD=true \
  -e POSTPROCESS_DEVICE=cpu \
  -e POSTPROCESS_MODEL_DIR=/models/Qwen2.5-1.5B-Instruct \
  -v /path/to/models:/models:ro \
  -v /path/to/data:/data \
  funasr-leader-asr:cpu
```

CPU 版可用于验证业务逻辑，但 Qwen 后处理会比较慢。若只测 ASR 主链路，可在请求时传：

```bash
-F "post_process=false"
```

## 6. NPU 镜像编译

NPU Dockerfile 使用：

```dockerfile
FROM ascendai/pytorch:2.1.0-ubuntu22.04
```

该基础镜像需要包含：

```text
CANN
torch==2.1.0
torch_npu==2.1.0
```

在昇腾 ARM64 机器上构建：

```bash
cd /path/to/asrLeader-master
docker build -f docker/Dockerfile.npu -t funasr-leader-asr:npu .
```

如果在非 ARM64 机器上交叉构建：

```bash
docker buildx build \
  --platform linux/arm64 \
  -f docker/Dockerfile.npu \
  -t funasr-leader-asr:npu \
  --load .
```

注意：交叉构建只能证明镜像构建成功，不能证明 NPU 运行成功。最终必须在真实昇腾机器上验证。

## 7. NPU 容器运行

昇腾机器运行示例：

```bash
docker run -d \
  --name asr-leader-npu \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -p 8000:8000 \
  -e ASR_DEVICE=npu:0 \
  -e MODELS_DIR=/models \
  -e DATA_DIR=/data \
  -e POSTPROCESS_ENABLED=true \
  -e POSTPROCESS_PRELOAD=true \
  -e POSTPROCESS_DEVICE=npu:0 \
  -e POSTPROCESS_MODEL_DIR=/models/Qwen2.5-1.5B-Instruct \
  -v /path/to/models:/models:ro \
  -v /path/to/data:/data \
  funasr-leader-asr:npu
```

如果 Qwen 在 NPU 上暂时不稳定，可以先让 ASR 走 NPU，Qwen 后处理走 CPU：

```bash
-e ASR_DEVICE=npu:0
-e POSTPROCESS_DEVICE=cpu
```

这样可以先验证融合版业务链路。

## 8. 健康检查

启动后检查：

```bash
curl http://localhost:8000/health
```

预期返回类似：

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "npu:0",
  "leader_count": 5,
  "models_dir": "/models",
  "postprocess_enabled": true,
  "postprocess_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "postprocess_device": "npu:0"
}
```

CPU 测试时 `device` 和 `postprocess_device` 应为 `cpu`。

## 9. 接口测试

只测 ASR 主链路：

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/path/to/test.m4a" \
  -F "post_process=false"
```

测试 ASR + Qwen 后处理融合链路：

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/path/to/test.m4a" \
  -F "post_process=true"
```

预期 HTTP 返回：

```json
{
  "code": 200,
  "message": "success",
  "data": [
    {
      "speaker": "0",
      "start_time": 0.0,
      "end_time": 3.2,
      "text": "修正后的文本",
      "raw_text": "原始 ASR 文本",
      "post_processed": true
    }
  ]
}
```

不是每段都会被修正，未修正的段可能没有 `raw_text` 和 `post_processed`。

## 10. 快速验证后处理规则

可进入容器直接验证上下文兜底规则：

```bash
docker exec asr-leader-npu python - <<'PY'
from post_processor import TranscriptPostProcessor

segments = [
    {"text": "其实这个我是从四月七年前就开始被调试。"},
    {"text": "我记得啊清明清明前最后一个晚上讲过了个清明就要有些朋友上线。"},
]

processor = TranscriptPostProcessor()
print(processor.correct_segments(segments))
PY
```

若模型可用，Qwen 会做上下文纠错；若模型不可用，兜底规则也会把有清明上下文的“四月七年前”修成“清明前”。

## 11. 离线部署注意事项

当前 Dockerfile 会执行：

```text
apt-get install
pip install
```

完全离线环境有两种推荐方式：

1. 在有网络或有内网镜像源的环境构建完整镜像，再导出：

```bash
docker save -o funasr-leader-asr-npu.tar funasr-leader-asr:npu
```

目标环境导入：

```bash
docker load -i funasr-leader-asr-npu.tar
```

2. 准备内网 apt/pip 镜像源或 wheelhouse，然后改 Dockerfile 使用本地依赖。

模型也必须提前放好，推荐统一放在宿主机：

```text
/path/to/models/paraformer-zh
/path/to/models/fsmn-vad
/path/to/models/ct-punc
/path/to/models/cam++
/path/to/models/Qwen2.5-1.5B-Instruct
```

然后容器挂载：

```bash
-v /path/to/models:/models:ro
```

## 12. 常见问题

### 首次请求很慢

Qwen 模型首次加载会慢。测试时建议设置：

```bash
POSTPROCESS_PRELOAD=true
```

这样启动时就加载模型，问题会提前暴露。

### NPU 上 Qwen 加载失败

先确认：

```bash
docker logs asr-leader-npu
```

重点看：

```text
torch_npu
CANN
POSTPROCESS_DEVICE=npu:0
Qwen model path
```

如果只想先验证业务流程，可以临时切到 CPU 后处理：

```bash
POSTPROCESS_DEVICE=cpu
```

### 离线环境还在访问 ModelScope

说明没有设置模型本地路径。请设置：

```bash
POSTPROCESS_MODEL_DIR=/models/Qwen2.5-1.5B-Instruct
```

并确认目录内有：

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

### 只想关闭纠错

运行时设置：

```bash
POSTPROCESS_ENABLED=false
```

或者单次请求：

```bash
-F "post_process=false"
```

## 13. 本地已验证结果

本地 Windows Docker GPU 版已验证：

```text
health: ok
ASR 主链路 post_process=false: HTTP 200
融合后处理 post_process=true: HTTP 200
冷启动首次加载 Qwen 约 59s
热启动后约 19s
```

说明：本机没有昇腾 NPU，NPU 镜像仅做了 Dockerfile 和代码层准备，最终运行必须在真实昇腾环境验证。
