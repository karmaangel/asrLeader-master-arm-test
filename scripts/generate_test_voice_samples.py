from __future__ import annotations

import asyncio
import json
from pathlib import Path

import edge_tts


OUT_DIR = Path.home() / "Desktop" / "asr_voiceprint_test"
SEG_DIR = OUT_DIR / "segments"

LEADER_VOICE = "zh-CN-YunyangNeural"
OTHER_A_VOICE = "zh-CN-XiaoxiaoNeural"
OTHER_B_VOICE = "zh-CN-YunjianNeural"


ITEMS = {
    "leader_voiceprint_01.mp3": (
        LEADER_VOICE,
        "今天我们主要确认项目上线前的几个关键事项，包括接口稳定性、权限边界和异常处理。",
    ),
    "leader_voiceprint_02.mp3": (
        LEADER_VOICE,
        "后续测试环境需要持续观察日志，尤其关注语音识别结果和声纹匹配是否稳定。",
    ),
    "meeting_01_s01_leader.mp3": (
        LEADER_VOICE,
        "我们先看第一个问题，接口返回结构必须保持稳定，不能影响前端已经对接好的字段。",
    ),
    "meeting_01_s02_other_a.mp3": (
        OTHER_A_VOICE,
        "我这边主要担心页面展示，如果字段变化，测试人员会看不出哪句话是谁说的。",
    ),
    "meeting_01_s03_other_b.mp3": (
        OTHER_B_VOICE,
        "后端可以加一个兼容层，把旧字段和新字段都保留一段时间。",
    ),
    "meeting_01_s04_leader.mp3": (
        LEADER_VOICE,
        "可以，但是最终给客户的版本要简洁，只返回是否领导和匹配到的领导编号。",
    ),
    "meeting_02_s01_other_b.mp3": (
        OTHER_B_VOICE,
        "第二个场景我们模拟多人插话，看看说话人分离会不会把声音混到一起。",
    ),
    "meeting_02_s02_leader.mp3": (
        LEADER_VOICE,
        "这里重点不是每个字都完美，而是领导声纹不要误判，也不要漏掉明显的领导发言。",
    ),
    "meeting_02_s03_other_a.mp3": (
        OTHER_A_VOICE,
        "如果有人只说一两个字，系统最好不要强行判断，避免短句误识别。",
    ),
    "meeting_02_s04_leader.mp3": (
        LEADER_VOICE,
        "对，短句可以不标记，长句和多段发言才更适合做声纹判断。",
    ),
    "meeting_03_s01_other_a.mp3": (
        OTHER_A_VOICE,
        "第三个场景换一个话题，讨论客户现场部署和镜像更新流程。",
    ),
    "meeting_03_s02_leader.mp3": (
        LEADER_VOICE,
        "测试环境拉取新镜像以后，先检查健康接口，再上传两段领导声纹做注册。",
    ),
    "meeting_03_s03_other_b.mp3": (
        OTHER_B_VOICE,
        "如果客户需要重新注册领导，直接删除旧编号，再重新上传两段以上音频。",
    ),
    "meeting_03_s04_leader.mp3": (
        LEADER_VOICE,
        "没错，声纹数据要挂载到数据目录，容器重启以后不能丢失。",
    ),
}


async def synthesize(name: str, voice: str, text: str) -> None:
    output = SEG_DIR / name
    communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
    await communicate.save(str(output))


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    for name, (voice, text) in ITEMS.items():
        await synthesize(name, voice, text)

    manifest = {
        "leader_voice": LEADER_VOICE,
        "other_voices": [OTHER_A_VOICE, OTHER_B_VOICE],
        "leader_voiceprint_files": [
            "leader_voiceprint_01.wav",
            "leader_voiceprint_02.wav",
        ],
        "meeting_files": [
            "meeting_01_dialogue.wav",
            "meeting_02_dialogue.wav",
            "meeting_03_dialogue.wav",
        ],
        "note": "Synthetic Chinese dialogue generated for ASR and voiceprint testing.",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
