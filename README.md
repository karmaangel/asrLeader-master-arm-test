# FunASR Leader ASR

本项目是新的语音服务镜像，支持：

- 语音识别
- 说话人分离
- 领导声纹注册、删除、查询
- 转写时自动识别某一段话是否来自已注册领导

默认复用同级旧项目的模型目录：

```text
D:\Docker\asr\models
```

所以不需要再复制几 GB 模型文件。

## 1. 停掉原来的服务

如果你之前是在 `D:\Docker\asr` 里启动的旧镜像，先停掉：

```powershell
cd D:\Docker\asr
docker compose down
```

如果不想停旧服务，也可以给新服务换端口：

```powershell
$env:ASR_PORT="8001"
```

## 2. 构建新镜像

```powershell
cd D:\Docker\asr-funasr-leader
docker compose build asr-leader-cpu
```

有 NVIDIA GPU 的机器可以构建 GPU 版：

```powershell
cd D:\Docker\asr-funasr-leader
docker compose build asr-leader-gpu
```

## 3. 启动服务

CPU 版：

```powershell
cd D:\Docker\asr-funasr-leader
docker compose up -d asr-leader-cpu
```

容器启动入口是：

```text
D:\Docker\asr-funasr-leader\app\server.py
```

GPU 版：

```powershell
cd D:\Docker\asr-funasr-leader
docker compose up -d asr-leader-gpu
```

查看日志：

```powershell
docker compose logs -f
```

健康检查：

```powershell
curl http://localhost:8000/health
```

交互式接口文档：

```text
http://localhost:8000/docs
```

## 4. 注册领导声纹

准备一段该领导本人比较干净的音频，建议 10 秒以上。

```powershell
curl -X POST http://localhost:8000/leaders/enroll `
  -F "file=@D:\audio\leader_zhang.wav" `
  -F "leader_id=leader_zhang" `
  -F "name=张总"
```

可以给同一个领导多注册几段样本，准确率通常会更稳：

```powershell
curl -X POST http://localhost:8000/leaders/enroll `
  -F "file=@D:\audio\leader_zhang_2.wav" `
  -F "leader_id=leader_zhang" `
  -F "name=张总"
```

查看已注册领导：

```powershell
curl http://localhost:8000/leaders
```

删除某个领导：

```powershell
curl -X DELETE http://localhost:8000/leaders/leader_zhang
```

声纹数据保存在：

```text
D:\Docker\asr-funasr-leader\data\leaders.json
```

## 5. 转写并识别领导

```powershell
curl -X POST http://localhost:8000/transcribe `
  -F "file=@D:\audio\meeting.m4a" `
  -F "num_speakers=4" `
  -F "identify_leaders=true"
```

返回的每个分段里会带：

```json
{
  "speaker": "0",
  "start": 1.23,
  "end": 5.67,
  "text": "大家看一下这个方案。",
  "is_leader": true,
  "leader": {
    "leader_id": "leader_zhang",
    "name": "张总",
    "score": 0.72
  }
}
```

## 6. 常用排错

如果端口被旧服务占用：

```powershell
cd D:\Docker\asr
docker compose down
cd D:\Docker\asr-funasr-leader
docker compose up -d asr-leader-cpu
```

或者新服务使用 8001：

```powershell
cd D:\Docker\asr-funasr-leader
$env:ASR_PORT="8001"
docker compose up -d asr-leader-cpu
curl http://localhost:8001/health
```

如果构建下载依赖慢，可以设置代理后再 build。
